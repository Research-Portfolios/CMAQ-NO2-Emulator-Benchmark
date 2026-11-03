"""Client of the EPA RSIG web service: builds each monthly request, fetches it by paced streaming requests with three timed retries, treats any refusal or answer that holds no data as a stop, decompresses and checks the netCDF answer, and writes the file with its provenance row."""
import contextlib
import datetime
import gzip
import hashlib
import http.cookiejar
import os
import shutil
import threading
import time
import zlib
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from choi_dl import common, prov
HOST = 'ofmpub.epa.gov'
ENDPOINT = 'https://ofmpub.epa.gov/rsig/rsigserver'
BBOX = '-134.5,20.5,-55.45,56.6'
LAYER_RANGE = ',1,1'
USER_AGENT = 'emul EQUATES research download; python-requests/2.34.2'
RAW_RSIG = os.path.join(common.RAW, 'rsig')
WORK_RSIG = os.path.join(common.WORK, 'rsig')
NULL = -9.999e+36
RETRY_WAITS = (30.0, 120.0, 600.0)
CONNECT_TIMEOUT = 60.0
READ_TIMEOUT = 900.0
WALL_CAP = 3600.0
MIN_GAP = 1.0
CHUNK = 1 << 16
TEXT_CAP = 1 << 20
FINAL_EXTS = ('.nc.gz', '.nc', '.txt', '.bin')
VERSION_FMT = 'EQUATES v1.0; CMAQ v5.3.2; RSIG coverage %s; DescribeCoverage sha256 %s'
PROV_LOCK = threading.Lock()
NC_LOCK = threading.Lock()
NAME_LOCK = threading.Lock()
_C = 'cmaq.equates.conus.'
SERIES = {'T1': ('T1_aconc_NO2_DM', _C + 'aconc.NO2', 'NO2', 'ppbV', 'DAILY_MEAN', False, False), 'M1': ('M1_metcro3d_PRES_L1_DM', _C + 'metcro3d.PRES', 'PRES', 'Pa', 'DAILY_MEAN', True, False), 'M2': ('M2_aconc_SFC_TMP_DM', _C + 'aconc.SFC_TMP', 'SFC_TMP', 'C', 'DAILY_MEAN', False, False), 'M3': ('M3_aconc_WSPD10_DM', _C + 'aconc.WSPD10', 'WSPD10', 'm_s-1', 'DAILY_MEAN', False, False), 'M4': ('M4_metdot3d_UWIND_L1_DM', _C + 'metdot3d.UWIND', 'UWIND', 'm_s-1', 'DAILY_MEAN', True, True), 'M5': ('M5_metdot3d_VWIND_L1_DM', _C + 'metdot3d.VWIND', 'VWIND', 'm_s-1', 'DAILY_MEAN', True, True), 'M6': ('M6_aconc_SOL_RAD_DM', _C + 'aconc.SOL_RAD', 'SOL_RAD', 'W_m-2', 'DAILY_MEAN', False, False), 'M7': ('M7_metcro3d_QV_L1_DM', _C + 'metcro3d.QV', 'QV', 'kg_kg-1', 'DAILY_MEAN', True, False), 'M8': ('M8_aconc_PBLH_DM', _C + 'aconc.PBLH', 'PBLH', 'm', 'DAILY_MEAN', False, False), 'M9': ('M9_aconc_precip_DM', _C + 'aconc.precip', 'precip', 'cm', 'DAILY_MEAN', False, False), 'X6': ('X6_aconc_PBLH_DMAX', _C + 'aconc.PBLH', 'PBLH', 'm', 'DAILY_MAX', False, False), 'T2': ('T2_lstaconc_DAILY_NO2', _C + 'lstaconc.DAILY_NO2', 'DAILY_NO2', 'ppbV', None, False, False), 'H_NO2': ('TEST_aconc_NO2_H', _C + 'aconc.NO2', 'NO2', 'ppbV', None, False, False), 'H_PBLH': ('TEST_aconc_PBLH_H', _C + 'aconc.PBLH', 'PBLH', 'm', None, False, False), 'H_WSPD10': ('TEST_aconc_WSPD10_H', _C + 'aconc.WSPD10', 'WSPD10', 'm_s-1', None, False, False), 'H_WDIR10': ('TEST_aconc_WDIR10_H', _C + 'aconc.WDIR10', 'WDIR10', 'deg', None, False, False)}
LAG_KEYS = ('M8', 'M3', 'M4', 'M5', 'M2')

def yyyyddd(d):
    return d.year * 1000 + d.timetuple().tm_yday

def _days(start, end):
    d0 = datetime.date.fromisoformat(start)
    d1 = datetime.date.fromisoformat(end)
    out = []
    while d0 <= d1:
        out.append(d0)
        d0 += datetime.timedelta(days=1)
    return out

def make_item(key, start, end, label, hourly=False, sid=None):
    s_sid, cov, var, units, agg, layered, dot = SERIES[key]
    sid = sid or s_sid
    days = _days(start, end)
    if hourly:
        agg = None
        dates = [yyyyddd(d) for d in days for _h in range(24)]
        times = [h * 10000 for _d in days for h in range(24)]
        tstep = 10000
    else:
        dates = [yyyyddd(d) for d in days]
        times = None
        tstep = 240000
    return {'id': '%s_%s' % (sid, label), 'key': key, 'sid': sid, 'coverage': cov, 'var': var, 'units': units, 'aggregate': agg, 'layered': layered, 'dot': dot, 'start': start, 'end': end, 'hourly': bool(hourly), 'tstep': tstep, 'steps': len(dates), 'dates': dates, 'times': times}

def month_item(key, year, month):
    d0 = datetime.date(year, month, 1)
    d1 = datetime.date(year + (month == 12), month % 12 + 1, 1) - datetime.timedelta(days=1)
    return make_item(key, d0.isoformat(), d1.isoformat(), '%04d%02d' % (year, month))

def lag_item(key):
    sid = 'LAG_' + key + SERIES[key][0][2:]
    return make_item(key, '2014-12-31', '2014-12-31', '20141231', sid=sid)

def url_for(item):
    bbox = BBOX + (LAYER_RANGE if item['layered'] else '')
    u = ENDPOINT + '?SERVICE=wcs&VERSION=1.0.0&REQUEST=GetCoverage&FORMAT=netcdf-ioapi&TIME=%sT00:00:00Z/%sT23:59:59Z&BBOX=%s&COVERAGE=%s&COMPRESS=1' % (item['start'], item['end'], bbox, item['coverage'])
    if item['aggregate']:
        u += '&AGGREGATE=' + item['aggregate']
    assert ' ' not in u and '\n' not in u
    prepared = requests.Request('GET', u).prepare().url
    if prepared != u:
        raise ValueError('requests would change the URL: ' + prepared)
    return u

class Gate:

    def __init__(self, concurrency=1, min_gap=MIN_GAP):
        self.concurrency = int(concurrency)
        self.sem = threading.Semaphore(self.concurrency)
        self.lock = threading.Lock()
        self.last = None
        self.min_gap = float(min_gap)

    @contextlib.contextmanager
    def slot(self):
        self.sem.acquire()
        try:
            with self.lock:
                if self.last is not None:
                    wait = self.last + self.min_gap - time.monotonic()
                    if wait > 0:
                        time.sleep(wait)
                self.last = time.monotonic()
            yield
        finally:
            self.sem.release()

def _session(user_agent):
    s = requests.Session()
    s.trust_env = False
    s.cookies.set_policy(http.cookiejar.DefaultCookiePolicy(allowed_domains=[]))
    s.mount('https://', HTTPAdapter(max_retries=Retry(total=0, read=False, redirect=False, raise_on_status=False)))
    s.headers.clear()
    s.headers.update({'User-Agent': user_agent, 'Accept': '*/*', 'Accept-Encoding': 'identity', 'Connection': 'close'})
    return s

def kind_of(head):
    if head[:2] == b'\x1f\x8b':
        return 'gzip'
    if head[:3] == b'CDF' and head[3:4] in (b'\x01', b'\x02', b'\x05'):
        return 'netcdf3'
    if head[:4] == b'\x89HDF':
        return 'netcdf4'
    return 'text'
EXT = {'gzip': '.nc.gz', 'netcdf3': '.nc', 'netcdf4': '.nc', 'text': '.txt'}

def _final_exists(base):
    return any((os.path.exists(base + e) for e in FINAL_EXTS))

def next_base(item):
    folder = os.path.join(RAW_RSIG, item['sid'])
    os.makedirs(folder, exist_ok=True)
    base = os.path.join(folder, item['id'])
    n = 1
    while _final_exists(base if n == 1 else '%s_try%d' % (base, n)):
        n += 1
    return base if n == 1 else '%s_try%d' % (base, n)

def raw_files(item):
    folder = os.path.join(RAW_RSIG, item['sid'])
    base = os.path.join(folder, item['id'])
    out = []
    n = 1
    while True:
        b = base if n == 1 else '%s_try%d' % (base, n)
        found = [b + e for e in FINAL_EXTS if os.path.exists(b + e)]
        if not found:
            if n > 1:
                break
        out.extend(found)
        n += 1
        if n > 50:
            break
    return out

def work_path_of(raw_path):
    if not raw_path.endswith('.nc.gz'):
        return None
    sid = os.path.basename(os.path.dirname(raw_path))
    return os.path.join(WORK_RSIG, sid, os.path.basename(raw_path)[:-3])

def check_path_of(raw_path):
    if not raw_path.endswith('.nc.gz'):
        return None
    return os.path.join(common.W_TMP, 'rsig_check', os.path.basename(raw_path)[:-3])

def remove_check_copy(path, tries=5, gap=0.2):
    left = []
    for p in (path, path + '.partial'):
        try:
            for i in range(tries):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                    break
                except OSError:
                    if i + 1 < tries:
                        time.sleep(gap)
            if os.path.exists(p):
                left.append(p)
        except Exception:
            left.append(p)
    return left

def _safe_text(b, n=160):
    t = b[:n].decode('ascii', 'replace')
    t = ''.join((c if 32 <= ord(c) < 127 else ' ' for c in t))
    return common.mask(' '.join(t.split()))

def _row(folder, **kw):
    with PROV_LOCK:
        return prov.append(folder, **kw)

def _get_once(url, base, user_agent):
    partial = base + '.partial'
    res = {'url': url, 'base': base, 'status': None, 'bytes': 0, 'seconds': None, 't_headers': None, 'kind': None, 'outcome': None, 'reason': '', 'path': None, 'sha256': None, 'content_type': None, 'content_encoding': None, 'etag': None, 'last_modified': None}
    t0 = time.monotonic()
    res['start_utc'], res['start_local'] = common.now_pair()
    s = _session(user_agent)
    try:
        try:
            r = s.get(url, stream=True, allow_redirects=False, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
        except (requests.ConnectionError, requests.Timeout) as e:
            res.update(outcome='retry', reason='connect: ' + type(e).__name__, seconds=time.monotonic() - t0)
            return res
        res['t_headers'] = time.monotonic() - t0
        res['status'] = r.status_code
        res['content_type'] = common.mask(r.headers.get('Content-Type', ''))
        res['content_encoding'] = common.mask(r.headers.get('Content-Encoding', ''))
        res['etag'] = common.mask(r.headers.get('ETag', '')) or None
        res['last_modified'] = common.mask(r.headers.get('Last-Modified', '')) or None
        h = hashlib.sha256()
        n = 0
        head = b''
        data_ok = r.status_code == 200
        try:
            with open(partial, 'wb') as f:
                for chunk in r.iter_content(CHUNK):
                    if not chunk:
                        continue
                    if len(head) < 8:
                        head += chunk[:8 - len(head)]
                    f.write(chunk)
                    h.update(chunk)
                    n += len(chunk)
                    if time.monotonic() - t0 > WALL_CAP:
                        raise TimeoutError('wall cap %.0f s' % WALL_CAP)
                    if (not data_ok or kind_of(head) == 'text') and len(head) >= 8 and (n > TEXT_CAP):
                        break
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError, TimeoutError) as e:
            res.update(outcome='retry', reason='body: ' + type(e).__name__, bytes=n, seconds=time.monotonic() - t0)
            return res
        finally:
            r.close()
        res['seconds'] = time.monotonic() - t0
        res['bytes'] = n
        kind = kind_of(head) if n else 'text'
        res['kind'] = kind
        cl = r.headers.get('Content-Length')
        if data_ok and kind != 'text' and (cl is not None) and cl.isdigit() and (int(cl) != n) and (res['content_encoding'] in ('', 'identity')):
            res.update(outcome='retry', reason='incomplete body: %d of %s bytes' % (n, cl))
            return res
        with NAME_LOCK:
            final = base + EXT[kind]
            if os.path.exists(final):
                raise FileExistsError(common.rel(final))
            os.rename(partial, final)
        res['path'] = final
        res['sha256'] = h.hexdigest()
        st = r.status_code
        if st == 200 and kind != 'text':
            res['outcome'] = 'data'
        elif 500 <= st <= 599:
            res.update(outcome='retry', reason='http %d' % st)
        elif st in (403, 429):
            res.update(outcome='refused', reason='http %d' % st)
        elif 300 <= st <= 399:
            res.update(outcome='refused', reason='http %d redirect' % st)
        elif st != 200:
            res.update(outcome='refused', reason='http %d' % st)
        else:
            with open(final, 'rb') as f:
                first = f.read(200)
            res.update(outcome='refused', reason='not data: %s' % _safe_text(first))
        return res
    finally:
        s.close()

def _decompress(raw_path, work_path):
    os.makedirs(os.path.dirname(work_path), exist_ok=True)
    if os.path.exists(work_path):
        raise FileExistsError(common.rel(work_path))
    partial = work_path + '.partial'
    h = hashlib.sha256()
    n = 0
    with gzip.open(raw_path, 'rb') as g, open(partial, 'wb') as f:
        while True:
            b = g.read(CHUNK * 16)
            if not b:
                break
            f.write(b)
            h.update(b)
            n += len(b)
    os.rename(partial, work_path)
    return (h.hexdigest(), n)

def _attr(ds, name):
    try:
        v = getattr(ds, name)
    except AttributeError:
        return None
    try:
        return v.item() if hasattr(v, 'item') else v
    except (ValueError, AttributeError):
        return v.tolist() if hasattr(v, 'tolist') else v
GRID_ATTRS = ('NCOLS', 'NROWS', 'NLAYS', 'NVARS', 'TSTEP', 'SDATE', 'STIME', 'GDTYP', 'P_ALP', 'P_BET', 'P_GAM', 'XCENT', 'YCENT', 'XORIG', 'YORIG', 'XCELL', 'YCELL', 'VGTYP', 'VGTOP', 'FTYPE')

def check_nc(item, path):
    import netCDF4
    import numpy as np
    info = {}
    with NC_LOCK:
        ds = netCDF4.Dataset(path, 'r')
        try:
            info['file_format'] = ds.file_format
            info['variables'] = [str(v) for v in ds.variables]
            info['dims'] = {str(k): len(d) for k, d in ds.dimensions.items()}
            info['attrs'] = {}
            for a in GRID_ATTRS:
                v = _attr(ds, a)
                if v is not None:
                    info['attrs'][a] = v if not isinstance(v, bytes) else v.decode('ascii', 'replace')
            if item['var'] not in ds.variables:
                return (False, 'variable %s not in %s' % (item['var'], ','.join(info['variables'])), info)
            v = ds.variables[item['var']]
            v.set_auto_maskandscale(False)
            units = str(getattr(v, 'units', '')).strip()
            info['var'] = item['var']
            info['units'] = units
            info['var_dims'] = [str(d) for d in v.dimensions]
            info['shape'] = [int(x) for x in v.shape]
            data = np.asarray(v[:])
            tf = None
            if 'TFLAG' in ds.variables:
                t = ds.variables['TFLAG']
                t.set_auto_maskandscale(False)
                tf = np.asarray(t[:])
        finally:
            ds.close()
    a = info['attrs']
    rows, cols = (a.get('NROWS'), a.get('NCOLS'))
    info['rows'], info['cols'] = (rows, cols)
    null = int(np.count_nonzero(data <= -9e+36))
    nan = int(np.count_nonzero(np.isnan(data)))
    good = data[(data > -9e+36) & np.isfinite(data)]
    info['null_count'] = null
    info['nan_count'] = nan
    info['values'] = int(data.size)
    if good.size:
        info['min'] = float(good.min())
        info['max'] = float(good.max())
    if tf is not None and tf.ndim == 3:
        info['tflag_dates'] = sorted(set((int(x) for x in tf[:, 0, 0])))
        info['tflag_times'] = sorted(set((int(x) for x in tf[:, 0, 1])))
        info['tflag_first'] = [int(tf[0, 0, 0]), int(tf[0, 0, 1])]
        info['tflag_last'] = [int(tf[-1, 0, 0]), int(tf[-1, 0, 1])]
    if units.replace(' ', '_') != item['units']:
        return (False, 'units %r, expected %r' % (units, item['units']), info)
    want = [(299, 459)] + ([(300, 460)] if item['dot'] else [])
    if (rows, cols) not in want:
        return (False, 'grid %sx%s, expected %s' % (rows, cols, ' or '.join(('%dx%d' % w for w in want))), info)
    if len(info['shape']) != 4 or tuple(info['shape'][2:]) != (rows, cols):
        return (False, 'variable shape %s' % info['shape'], info)
    if a.get('NLAYS') != 1 or info['shape'][1] != 1:
        return (False, 'NLAYS %s, layer dim %s' % (a.get('NLAYS'), info['shape'][1]), info)
    if a.get('TSTEP') != item['tstep']:
        return (False, 'TSTEP %s, expected %d' % (a.get('TSTEP'), item['tstep']), info)
    if info['shape'][0] != item['steps']:
        return (False, 'steps %d, expected %d' % (info['shape'][0], item['steps']), info)
    if tf is None or tf.ndim != 3 or tf.shape[0] != item['steps']:
        return (False, 'TFLAG missing or of another length', info)
    dates = [int(x) for x in tf[:, 0, 0]]
    if dates != item['dates']:
        return (False, 'TFLAG dates %s..%s, expected %s..%s' % (dates[0], dates[-1], item['dates'][0], item['dates'][-1]), info)
    if item['times'] is not None:
        times = [int(x) for x in tf[:, 0, 1]]
        if times != item['times']:
            return (False, 'TFLAG hours %s, expected 0..230000' % sorted(set(times))[:4], info)
    return (True, '', info)

def fetch(item, gate, describe_sha256, user_agent=USER_AGENT, log=None):
    url = url_for(item)
    version = VERSION_FMT % (item['coverage'], describe_sha256)
    result = {'id': item['id'], 'url': url, 'status': None, 'attempts': [], 'seconds': None, 't_headers': None, 'bytes': 0, 'kind': None, 'raw': None, 'work': None, 'note': '', 'info': None, 'http_status': None}
    for attempt in range(1 + len(RETRY_WAITS)):
        if attempt:
            wait = RETRY_WAITS[attempt - 1]
            if log:
                log('%s: retry %d after %.0f s' % (item['id'], attempt, wait))
            time.sleep(wait)
        with NAME_LOCK:
            base = next_base(item)
        with gate.slot():
            res = _get_once(url, base, user_agent)
        result['attempts'].append({k: res.get(k) for k in ('start_local', 'status', 'bytes', 'seconds', 't_headers', 'kind', 'outcome', 'reason', 'content_type', 'content_encoding')})
        result['http_status'] = res['status']
        if res['path'] and res['outcome'] != 'data':
            note = ('check failed: %s: %s' % (res['outcome'], res['reason']))[:300]
            _row(RAW_RSIG, path=res['path'], source=url, retrieved_utc=common.now_pair()[0], retrieved_local=common.now_pair()[1], http_status=res['status'], bytes=res['bytes'], sha256=res['sha256'], version=version, etag=res['etag'], last_modified=res['last_modified'], note=note)
            result['raw'] = res['path']
        if res['outcome'] == 'retry':
            continue
        if res['outcome'] == 'refused':
            result.update(status='refused', note=res['reason'], seconds=res['seconds'], bytes=res['bytes'], kind=res['kind'])
            return result
        result.update(seconds=res['seconds'], t_headers=res['t_headers'], bytes=res['bytes'], kind=res['kind'], raw=res['path'])
        raw_utc, raw_local = common.now_pair()
        nc_path = res['path']
        inner = res['kind']
        work = check_path_of(res['path'])
        ok, reason, info = (False, '', {})
        wsha = wbytes = None
        left = []
        try:
            if work:
                for stale in (work, work + '.partial'):
                    if os.path.exists(stale):
                        os.remove(stale)
                wsha, wbytes = _decompress(res['path'], work)
                with open(work, 'rb') as f:
                    inner = kind_of(f.read(8))
                nc_path = work
            if inner not in ('netcdf3', 'netcdf4'):
                reason = 'the answer holds %s, not netCDF' % inner
            else:
                ok, reason, info = check_nc(item, nc_path)
        except (OSError, EOFError, zlib.error, gzip.BadGzipFile) as e:
            reason = '%s: %s' % (type(e).__name__, common.mask(str(e))[:120])
        except Exception as e:
            reason = '%s: %s' % (type(e).__name__, common.mask(str(e))[:120])
        finally:
            if work:
                left = remove_check_copy(work)
        if left and log:
            try:
                log('%s: the check copy was not deleted (%d file(s) left in W\\work/rsig_check)' % (item['id'], len(left)))
            except Exception:
                pass
        kind_txt = res['kind'] + ('>' + inner if work else '')
        if ok:
            note = 'check ok: %s %s %sx%s %d steps; kind %s' % (info['var'], info['units'], info['rows'], info['cols'], info['shape'][0], kind_txt)
        else:
            note = ('check failed: %s; kind %s' % (reason, kind_txt))[:300]
        info = info or {}
        info['inner_kind'] = inner
        mnote = None
        if wsha is not None:
            mnote = 'sha256 of the answer decompressed for its check in W\\tmp, %d bytes; %s' % (wbytes, 'the copy deleted' if not left else 'the copy NOT deleted, %d file(s) left' % len(left))
        _row(RAW_RSIG, path=res['path'], source=url, retrieved_utc=raw_utc, retrieved_local=raw_local, http_status=res['status'], bytes=res['bytes'], sha256=res['sha256'], version=version, etag=res['etag'], last_modified=res['last_modified'], member_sha256=wsha, member_note=mnote, note=note)
        result.update(status='done' if ok else 'failed', note=note, info=info)
        return result
    result.update(status='failed', note='check failed: %d tries: %s' % (len(result['attempts']), result['attempts'][-1]['reason']))
    return result

def item_done(item):
    with PROV_LOCK:
        rows = prov.load(RAW_RSIG)
        for p in raw_files(item):
            r = rows.get(common.rel(p))
            if not r or not str(r.get('note', '')).startswith('check ok'):
                continue
            if not prov.verified(RAW_RSIG, p):
                continue
            return True
        return False

def free_gb():
    return shutil.disk_usage(common.W).free / 1000000000.0
