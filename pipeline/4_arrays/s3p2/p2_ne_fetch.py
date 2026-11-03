"""Downloads two Natural Earth archives (the states and provinces with lakes removed, and the populated places) by paced, retried requests, tests each archive, extracts its shapefile parts, compares them with earlier copies, and writes their provenance rows."""
import hashlib
import http.cookiejar
import io
import os
import sys
import time
import zipfile
sys.path.insert(0, '/content/emul/run/code')
from choi_dl import common, prov
from s3p2 import p2common as P
HOST = 'naturalearth.s3.amazonaws.com'
BASE_URL = 'https://naturalearth.s3.amazonaws.com/10m_cultural/'
ZIPS = (('ne_10m_admin_1_states_provinces_lakes.zip', 15057498), ('ne_10m_populated_places.zip', 2811199))
USER_AGENT = 'emul Natural Earth research download; python-requests/2.34.2'
RETRY_WAITS = (10.0, 30.0, 90.0)
MIN_GAP = 1.0
CHUNK = 1 << 16
EXTRACT_EXT = ('.shp', '.shx', '.dbf', '.prj', '.cpg')
NOTE = "public domain; Made with Natural Earth; fetched raw by the run's own code (requests 2.34.2)"
_LAST = [None]

def session():
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    if requests.__version__ != '2.34.2':
        raise SystemExit('requests %s, not the pinned 2.34.2' % requests.__version__)
    s = requests.Session()
    s.trust_env = False
    s.cookies.set_policy(http.cookiejar.DefaultCookiePolicy(allowed_domains=[]))
    s.mount('https://', HTTPAdapter(max_retries=Retry(total=0, read=False, redirect=False, raise_on_status=False)))
    s.headers.clear()
    s.headers.update({'User-Agent': USER_AGENT, 'Accept': '*/*', 'Accept-Encoding': 'identity', 'Connection': 'close'})
    return s

def gate():
    if _LAST[0] is not None:
        w = _LAST[0] + MIN_GAP - time.monotonic()
        if w > 0:
            time.sleep(w)
    _LAST[0] = time.monotonic()

def get_once(url, dest):
    import requests
    part = dest + '.partial'
    res = {'url': url, 'status': None, 'bytes': 0, 'outcome': None, 'reason': '', 'sha256': None, 'etag': None, 'last_modified': None, 'content_length': None}
    res['utc'], res['local'] = common.now_pair()
    gate()
    s = session()
    try:
        try:
            r = s.get(url, stream=True, allow_redirects=False, timeout=(60.0, 300.0))
        except (requests.ConnectionError, requests.Timeout) as e:
            res.update(outcome='retry', reason='connect: ' + type(e).__name__)
            return res
        res['status'] = r.status_code
        res['etag'] = common.mask(r.headers.get('ETag', '')) or None
        res['last_modified'] = common.mask(r.headers.get('Last-Modified', '')) or None
        res['content_length'] = r.headers.get('Content-Length')
        st = r.status_code
        if st != 200:
            r.close()
            if 500 <= st <= 599:
                res.update(outcome='retry', reason='http %d' % st)
            elif 300 <= st <= 399:
                res.update(outcome='refused', reason='http %d redirect' % st)
            else:
                res.update(outcome='refused', reason='http %d' % st)
            return res
        h = hashlib.sha256()
        n = 0
        head = b''
        try:
            with open(part, 'wb') as f:
                for chunk in r.iter_content(CHUNK):
                    if not chunk:
                        continue
                    if len(head) < 4:
                        head += chunk[:4 - len(head)]
                    f.write(chunk)
                    h.update(chunk)
                    n += len(chunk)
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as e:
            res.update(outcome='retry', reason='body: ' + type(e).__name__, bytes=n)
            return res
        finally:
            r.close()
        res['bytes'] = n
        cl = res['content_length']
        if cl is not None and cl.isdigit() and (int(cl) != n):
            res.update(outcome='retry', reason='incomplete body: %d of %s' % (n, cl))
            return res
        if head != b'PK\x03\x04':
            res.update(outcome='refused', reason='not a zip')
            return res
        if os.path.exists(dest):
            raise FileExistsError(common.rel(dest))
        os.replace(part, dest)
        res.update(outcome='data', sha256=h.hexdigest())
        return res
    finally:
        s.close()

def fetch(name, want_bytes, log):
    url = BASE_URL + name
    dest = os.path.join(P.NE_ZIPS, name)
    if os.path.exists(dest) and prov.verified(P.NE_ZIPS, dest):
        log('%s: present and verified by its row; not fetched' % name)
        return {'name': name, 'url': url, 'outcome': 'present', 'path': common.rel(dest)}
    if os.path.exists(dest):
        raise SystemExit('%s exists without a verified row: stop' % name)
    tries = []
    for k in range(len(RETRY_WAITS) + 1):
        res = get_once(url, dest)
        tries.append({kk: res[kk] for kk in ('utc', 'local', 'status', 'bytes', 'outcome', 'reason')})
        log('%s try %d: status %s, %d bytes, %s %s' % (name, k + 1, res['status'], res['bytes'], res['outcome'], res['reason']))
        if res['outcome'] in ('data', 'refused'):
            break
        if k < len(RETRY_WAITS):
            time.sleep(RETRY_WAITS[k])
    part = dest + '.partial'
    if os.path.exists(part):
        os.remove(part)
    out = {'name': name, 'url': url, 'tries': tries, 'outcome': res['outcome'], 'reason': res['reason']}
    if res['outcome'] != 'data':
        return out
    ver = read_version(dest)
    row = prov.append(P.NE_ZIPS, path=dest, source=url, retrieved_utc=res['utc'], retrieved_local=res['local'], http_status=res['status'], bytes=res['bytes'], sha256=res['sha256'], version="Natural Earth %s (the zip's VERSION.txt)" % ver, etag=res['etag'], last_modified=res['last_modified'], note=NOTE)
    out.update(path=row['path'], bytes=res['bytes'], sha256=res['sha256'], etag=res['etag'], last_modified=res['last_modified'], version_txt=ver, listed_bytes=want_bytes, bytes_equal_listing=res['bytes'] == want_bytes)
    return out

def version_member(names):
    for n in names:
        if n == 'VERSION.txt' or n.endswith('.VERSION.txt'):
            return n
    return None

def read_version(path):
    with zipfile.ZipFile(path) as z:
        vm = version_member([i.filename for i in z.infolist()])
        if vm is None:
            return 'none (no version file)'
        txt = z.read(vm).decode('ascii', 'replace')
    return common.mask(' '.join(txt.split()))[:60]

def fix_rows(log):
    rows = prov.load(P.NE_ZIPS)
    n = 0
    for path, r in sorted(rows.items()):
        if 'none (no VERSION.txt)' not in r.get('version', ''):
            continue
        full = os.path.join(P.ROOT, path)
        name = os.path.basename(full)
        zname = name if name.endswith('.zip') else None
        if zname is None:
            stem = os.path.basename(os.path.dirname(full))
            zname = stem + '.zip'
        ver = read_version(os.path.join(P.NE_ZIPS, zname))
        sha = prov.sha256_file(full)
        if sha != r['sha256'] or int(os.stat(full).st_size) != int(r['bytes']):
            raise SystemExit('%s differs from its row' % path)
        kw = {k: v for k, v in r.items() if k in prov.FIELDS}
        kw['version'] = "Natural Earth %s (the zip's %s)" % (ver, zname[:-4] + '.VERSION.txt')
        kw['note'] = r.get('note', '') + "; row written again at %s to correct its version field (the first row named none: the zip's version file is <stem>.VERSION.txt)" % P.now_local()
        prov.append(P.NE_ZIPS, **kw)
        n += 1
    log('rows written again with the version: %d' % n)
    return n

def safe_name(n):
    return n and '/' not in n and ('\\' not in n) and ('..' not in n) and (':' not in n)

def extract(name, log):
    zpath = os.path.join(P.NE_ZIPS, name)
    stem = name[:-4]
    dst_dir = os.path.join(P.NE_EXTRACTED, stem)
    os.makedirs(dst_dir, exist_ok=True)
    zsha = prov.sha256_file(zpath)
    rows = prov.load(P.NE_ZIPS)
    out = {'zip': common.rel(zpath), 'zip_sha256': zsha, 'members': [], 'extracted': []}
    with zipfile.ZipFile(zpath) as z:
        bad = z.testzip()
        out['testzip'] = 'ok' if bad is None else 'bad member %s' % bad
        if bad is not None:
            raise SystemExit('%s: CRC fails at %s' % (name, bad))
        ver = read_version(zpath)
        for i in z.infolist():
            out['members'].append({'name': common.mask(i.filename), 'bytes': int(i.file_size), 'crc': '%08x' % i.CRC})
            ext = os.path.splitext(i.filename)[1].lower()
            if not (ext in EXTRACT_EXT or i.filename == version_member([i.filename])):
                continue
            if not safe_name(i.filename):
                raise SystemExit('unsafe member name in %s' % name)
            dest = os.path.join(dst_dir, i.filename)
            if os.path.exists(dest):
                r = rows.get(common.rel(dest))
                if r and prov.verified(P.NE_ZIPS, dest):
                    out['extracted'].append({'name': i.filename, 'path': common.rel(dest), 'bytes': int(r['bytes']), 'sha256': r['sha256'], 'written': False})
                    continue
                raise SystemExit('%s exists without a verified row' % dest)
            data = z.read(i)
            part = dest + '.partial'
            with open(part, 'wb') as f:
                f.write(data)
            os.replace(part, dest)
            sha = hashlib.sha256(data).hexdigest()
            prov.append(P.NE_ZIPS, path=dest, source='zip member %s of %s (%s%s)' % (i.filename, common.rel(zpath), BASE_URL, name), bytes=len(data), sha256=sha, version="Natural Earth %s (the zip's VERSION.txt)" % ver, note='extracted by run\\code\\s3p2\\p2_ne_fetch.py; zip sha256 ' + zsha)
            out['extracted'].append({'name': i.filename, 'path': common.rel(dest), 'bytes': len(data), 'sha256': sha, 'written': True})
    for e in out['extracted']:
        old = os.path.join(P.NE_SHAPES, e['name'])
        if not os.path.exists(old):
            e['cartopy_file'] = 'absent'
            continue
        osha = prov.sha256_file(old)
        e['cartopy_file'] = common.rel(old)
        e['cartopy_bytes'] = int(os.stat(old).st_size)
        e['cartopy_sha256'] = osha
        e['equal_to_cartopy'] = osha == e['sha256'] and e['cartopy_bytes'] == e['bytes']
        log("  %s: %d bytes, sha16 %s; Cartopy's %s" % (e['name'], e['bytes'], e['sha256'][:16], 'EQUAL' if e['equal_to_cartopy'] else 'DIFFERS (%d bytes, sha16 %s)' % (e['cartopy_bytes'], osha[:16])))
    return out

def main():
    a = sys.argv[1:]
    common.set_temp()
    os.makedirs(P.NE_ZIPS, exist_ok=True)
    os.makedirs(P.NE_EXTRACTED, exist_ok=True)
    msgs = []

    def log(m):
        line = '%s %s' % (P.now_local(), common.mask(m))
        msgs.append(line)
        print(line, flush=True)
    res = {'start': P.now_local(), 'host': HOST, 'user_agent': USER_AGENT, 'fetch': [], 'extract': []}
    for name, want in ZIPS:
        if '--no-fetch' in a:
            continue
        r = fetch(name, want, log)
        res['fetch'].append(r)
        if r['outcome'] not in ('data', 'present'):
            log('%s: STOPPED (%s), not fetched another way' % (name, r['reason']))
    for name, _want in ZIPS:
        if os.path.exists(os.path.join(P.NE_ZIPS, name)):
            log('extract %s' % name)
            res['extract'].append(extract(name, log))
            res.setdefault('version', {})[name] = read_version(os.path.join(P.NE_ZIPS, name))
            log('%s version file: %s' % (name, res['version'][name]))
    res['rows_written_again'] = fix_rows(log)
    res['end'] = P.now_local()
    res['log'] = msgs
    P.write_json(os.path.join(P.NE_ZIPS, 'NE_ZIPS_P2.json'), res)
    log('written %s' % common.rel(os.path.join(P.NE_ZIPS, 'NE_ZIPS_P2.json')))
if __name__ == '__main__':
    P.run_main(main)
