"""Runs the S3 queue against the public CMAS bucket in order: the 2016 MCIP files, the daily means of the merged gridded NO, NO2, HONO and VOC emissions for 2017 and 2019, the stack groups archives, and the point and marine sectors summed into cells, each written as a compressed NumPy file with its provenance row."""
import io
import os
import shutil
import sys
import tarfile
import time
CODE_DIR = '/content/emul/run/code'
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
import numpy as np
from choi_dl import common, prov, s3io
PARTS = ('mcip', 'emis_merged', 'stackgroups', 'pt_marine')
MCIP_DIR = os.path.join(common.RAW, 'mcip')
S3META = os.path.join(common.RAW, 's3meta')
SG_DIR = os.path.join(common.RAW, 'emis', 'stackgroups')
EM_DIR = os.path.join(common.EXTRACT, 'emis_merged')
PT_DIR = os.path.join(common.EXTRACT, 'pt_marine')
FOLDER = {'mcip': MCIP_DIR, 'emis_merged': EM_DIR, 'stackgroups': SG_DIR, 'pt_marine': PT_DIR}
COPY_NAME = {'mcip': 's3_mcip', 'emis_merged': 's3_emis_merged', 'stackgroups': 's3_stackgroups', 'pt_marine': 's3_pt_marine'}
VERSION_EMIS = 'EQUATES v1.0 (the key names)'
VERSION_MCIP = 'EQUATES v1.0; MCIP v5.1; WRF v4.1.1 (the key names)'
DISK_FLOOR_GB = 30.0
COPY_EVERY = 20
MAX_REFUSALS = 3
MAX_HANGS = 3
GRID = (299, 459)
REFUSED_FILE = os.path.join(common.WORK, 'dl_refused_s3.txt')
HELD_SECTOR = 'held: sector dropped, stack count mismatch'
HELD_HANGS = 'held: the merged part stopped after %d hangs in a row of the selective reader' % MAX_HANGS

class StackCountMismatch(s3io.Failed):
    pass

def read_refused(path=None):
    path = REFUSED_FILE if path is None else path
    out = []
    seen = set()
    if not os.path.exists(path):
        return out
    with open(path, 'rb') as f:
        text = f.read().decode('utf-8-sig', 'replace')
    for line in text.replace('\r', '\n').split('\n'):
        iid = line.split('\t')[0].strip()
        if iid and iid not in seen:
            seen.add(iid)
            out.append(iid)
    return out

def refusal_kind(e):
    st = getattr(e, 'status', None)
    if st:
        return str(st)
    code = getattr(e, 'code', None)
    return str(code) if code else 'refused'

def write_refused(iid, kind, path=None):
    path = REFUSED_FILE if path is None else path
    u, loc = common.now_pair()
    fields = []
    for x in (iid, kind, u, loc):
        x = str(x)
        for c in ('\t', '\r', '\n'):
            x = x.replace(c, ' ')
        fields.append(x.strip())
    line = common.mask('\t'.join(fields))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'ab') as f:
        f.write((line + '\n').encode('ascii', 'replace'))
        f.flush()
        os.fsync(f.fileno())
    return line

class Rows:

    def __init__(self):
        self.cache = {}

    def rows(self, folder):
        if folder not in self.cache:
            self.cache[folder] = prov.load(folder)
        return self.cache[folder]

    def verified(self, folder, path):
        r = self.rows(folder).get(common.rel(path))
        if not r or 'sha256' not in r or 'bytes' not in r:
            return False
        if not os.path.exists(path) or os.stat(path).st_size != int(r['bytes']):
            return False
        return prov.sha256_file(path) == r['sha256']

    def append(self, folder, **row):
        r = prov.append(folder, **row)
        self.rows(folder)[r['path']] = r
        return r

def build_items(plan):
    out = []
    for m in plan['mcip']['members']:
        out.append(('mcip', m['id'], m))
    em = plan['emis_merged']
    for d in em['dates']:
        spec = dict(em['members'][d])
        spec['date'] = d
        out.append(('emis_merged', 'emis_%s' % d, spec))
    for y in sorted(plan['stackgroups']):
        spec = dict(plan['stackgroups'][y])
        spec['year'] = y
        out.append(('stackgroups', 'sg_%s' % y, spec))
    for m in plan['pt_marine']['members']:
        out.append(('pt_marine', m['id'], m))
    return out

def item_path(part, spec):
    if part == 'mcip':
        return os.path.join(MCIP_DIR, spec['dest'])
    if part == 'emis_merged':
        d = spec['date']
        return os.path.join(EM_DIR, d[:4], 'emis_merged_%s.npz' % d)
    if part == 'stackgroups':
        return os.path.join(SG_DIR, spec['dest'])
    d = spec['date']
    return os.path.join(PT_DIR, spec['sector'], d[:4], 'pt_%s_%s.npz' % (spec['sector'], d))

def free_gb():
    return shutil.disk_usage(common.W).free / 1000000000.0

def write_npz(path, arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    part = path + '.partial'
    with open(part, 'wb') as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(part, path)
    return (os.stat(path).st_size, prov.sha256_file(path))

def yyyyddd(d):
    import datetime as dt
    x = dt.date(int(d[:4]), int(d[4:6]), int(d[6:]))
    return x.year * 1000 + x.timetuple().tm_yday

class Ctx:

    def __init__(self, plan, progress, rows):
        self.plan = plan
        self.progress = progress
        self.rows = rows
        self.client = s3io.S3Client(log=progress.note, endpoint_url=plan['s3'].get('endpoint_url'))
        self.reader = None
        self.tix = {}
        self.groups = {}
        self.hangs = 0
        self.hangs_row = 0
        self.child_requests = 0
        self.child_bytes = 0

    def head(self, key):
        h = self.plan['keys'][key]
        return {'etag': h['etag'], 'size': int(h['size']), 'last_modified': h['last_modified']}

    def contents(self, tar_key):
        ck = tar_key[:-4] + '.contents.txt'
        path = os.path.join(S3META, ck.split('/')[-1])
        if not self.rows.verified(S3META, path):
            h = self.head(ck)
            info = self.client.stream_to_file(ck, 0, h['size'], path, if_match=h['etag'])
            u, loc = common.now_pair()
            self.rows.append(S3META, path=path, source='s3://%s/%s' % (s3io.BUCKET, ck), byte_range='0-%d' % (h['size'] - 1), retrieved_utc=u, retrieved_local=loc, http_status=info.get('status'), bytes=os.stat(path).st_size, sha256=info['sha256'], version=VERSION_EMIS if '/emis/' in ck else VERSION_MCIP, etag=h['etag'], last_modified=h['last_modified'], note='contents file of the tar (tar -tv lines); kept raw; parsed for member names, types and sizes only')
        with open(path, 'rb') as f:
            members, bad = s3io.parse_contents(f.read().decode('utf-8', 'replace'))
        return members

    def tindex(self, key):
        if key not in self.tix:
            h = self.head(key)
            self.tix[key] = s3io.TarIndex(self.client, key, h['etag'], h['size'], self.contents(key))
        return self.tix[key]

    def get_reader(self):
        if self.reader is None:
            self.reader = s3io.ReaderProcess(self.client.limiter, timeout=1800, endpoint_url=self.plan['s3'].get('endpoint_url'))
        return self.reader

    def groups_rowcol(self, sector, year):
        from choi import stacks
        import netCDF4
        kk = (sector, str(year))
        if kk not in self.groups:
            sg = self.plan['stackgroups'][str(year)]
            path = os.path.join(SG_DIR, sg['dest'])
            gm = self.plan['pt_marine']['groups_members']['%s_%s' % (sector, year)]
            with tarfile.open(path, 'r:') as tf:
                b = tf.extractfile(tf.getmember(gm)).read()
            ds = netCDF4.Dataset('g.nc', mode='r', memory=b)
            try:
                g = stacks.groups_row_col(ds)
            finally:
                ds.close()
            if g.get('reason'):
                raise s3io.Failed('stack groups %s %s: %s' % (sector, year, g['reason']))
            self.groups[kk] = g
        return self.groups[kk]

    def close(self):
        if self.reader is not None:
            self.reader.close()
            self.reader = None

def _check_nc4(path):
    import netCDF4
    ds = netCDF4.Dataset(path, 'r')
    try:
        dm = ds.data_model
    finally:
        ds.close()
    if not dm.startswith('NETCDF4'):
        raise ValueError('data model %s, not netCDF-4' % dm)
    return dm

def do_mcip(ctx, iid, spec, path):
    key = spec['key']
    p, m, info = s3io.save_member(ctx.client, key, ctx.head(key), ctx.tindex(key), spec['member'], MCIP_DIR, VERSION_MCIP, dest_name=spec['dest'], check=_check_nc4, note='MCIP member kept raw; header checked against the contents file; opened as netCDF-4')
    ctx.rows.cache.pop(MCIP_DIR, None)
    if info.get('skipped'):
        return (0, 'row re-verifies')
    return (int(info['bytes']), 'member at %d (%s); %s' % (m['data_offset'], m['how'], info.get('check')))

def _read_member(ctx, key, m, species, kind, route):
    h = ctx.head(key)
    if route == 'selective':
        res = ctx.get_reader().run({'key': key, 'member': m, 'etag': h['etag'], 'species': list(species), 'kind': kind, 'n_hours': 24, 'layer': 0})
        ctx.child_requests += int(res.get('requests_total', 0) or 0)
        ctx.child_bytes += int(res.get('bytes_total', 0) or 0)
        if not res.get('ok'):
            ek = res.get('error_kind')
            if ek == 'refused':
                raise s3io.Refused(res.get('error', ''), res.get('status'), res.get('code') or None)
            raise s3io.Failed(res.get('error', ''))
        ctx.hangs_row = 0
        w = res['window']
        note = 'selective reader: sha256 of the member bytes read, in offset order (%d bytes in %d cached segments, by %d ranged GETs; member of %d bytes)' % (w['offset_order_bytes'], w['segments'], w['requests'], m['size'])
        return (res, w['sha256_offset_order'], note, int(w['bytes']))
    need = m['size'] * 1.6 + 1000000000.0
    t0 = time.time()
    while True:
        try:
            import psutil
            avail = psutil.virtual_memory().available
        except Exception:
            avail = need
        if avail >= need or time.time() - t0 > 600:
            break
        time.sleep(30)
    if avail < need:
        raise s3io.Failed('memory: %.2f GB available, %.2f GB needed for a whole member of %d bytes' % (avail / 1000000000.0, need / 1000000000.0, m['size']))
    res, data = s3io.whole_read(ctx.client, key, m, h['etag'], species, kind, zero_copy=True)
    del data
    ctx.hangs_row = 0
    tr = res['transfer']
    note = 'whole member: sha256 of the %d member bytes streamed by one ranged GET' % tr['bytes']
    return (res, tr['sha256'], note, int(tr['bytes']))

def do_emis(ctx, iid, spec, path):
    em = ctx.plan['emis_merged']
    key, name, d = (spec['key'], spec['name'], spec['date'])
    species = em['species']
    m = ctx.tindex(key).locate(name)
    res, msha, mnote, nbytes = _read_member(ctx, key, m, species, 'grid', em['route'])
    hc = res['hours']
    if not hc.get('ok') or hc.get('date') != yyyyddd(d):
        raise s3io.Failed('hours: %s (date %s, expected %d)' % (hc.get('reason'), hc.get('date'), yyyyddd(d)))
    miss = [sp for sp in species if sp not in res['means']]
    if miss:
        raise s3io.Failed('species missing: %s' % ', '.join(miss))
    arrays = {}
    for sp in species:
        a = np.asarray(res['means'][sp])
        if a.shape != GRID or a.dtype != np.float32:
            raise s3io.Failed('%s: shape %s dtype %s' % (sp, a.shape, a.dtype))
        arrays[sp] = a
    units = [str(res['units'].get(sp)) for sp in species]
    arrays.update({'species': np.array(species), 'units': np.array(units), 'hours': np.array('00-23 UTC'), 'date': np.array(d), 'member': np.array(name), 'tflag_date': np.array(hc['date'], dtype=np.int64)})
    nb, sha = write_npz(path, arrays)
    u, loc = common.now_pair()
    h = ctx.head(key)
    diff = [sp for sp, un in zip(species, units) if em.get('units', {}).get(sp) not in (None, un)]
    ctx.rows.append(EM_DIR, path=path, source='s3://%s/%s (member %s)' % (s3io.BUCKET, key, name), byte_range='%d-%d' % (m['data_offset'], m['data_offset'] + m['size'] - 1), retrieved_utc=u, retrieved_local=loc, http_status=206, bytes=nb, sha256=sha, version=VERSION_EMIS, etag=h['etag'], last_modified=h['last_modified'], species=' '.join(species), hours='00-23 UTC', member_sha256=msha, member_note=mnote, note='daily means of records 0 to 23 (float64 sums stored as float32) at layer 1, %s route; units as the file states: %s; TFLAG %d hours 00-23 ok%s' % (em['route'], ', '.join(('%s %s' % x for x in zip(species, units))), hc['date'], '; UNITS DIFFER FROM THE PLAN for %s' % diff if diff else ''))
    return (nbytes, '%s; extract %d bytes%s' % (em['route'], nb, '; units differ: %s' % diff if diff else ''))

def do_stackgroups(ctx, iid, spec, path):
    key = spec['key']
    h = ctx.head(key)
    info = ctx.client.stream_to_file(key, 0, h['size'], path, if_match=h['etag'])
    u, loc = common.now_pair()
    ctx.rows.append(SG_DIR, path=path, source='s3://%s/%s' % (s3io.BUCKET, key), byte_range='0-%d' % (h['size'] - 1), retrieved_utc=u, retrieved_local=loc, http_status=info.get('status'), bytes=os.stat(path).st_size, sha256=info['sha256'], version=VERSION_EMIS, etag=h['etag'], last_modified=h['last_modified'], note='stack groups tar fetched whole, kept raw (the point and marine route)')
    return (int(info['bytes']), 'whole tar')

def do_pt(ctx, iid, spec, path):
    from choi import stacks
    pm = ctx.plan['pt_marine']
    key, name, d, sec = (spec['key'], spec['name'], spec['date'], spec['sector'])
    species = pm['species']
    g = ctx.groups_rowcol(sec, spec['year'])
    m = ctx.tindex(key).locate(name)
    res, msha, mnote, nbytes = _read_member(ctx, key, m, species, 'stack', pm['route'])
    hc = res['hours']
    if not hc.get('ok') or hc.get('date') != yyyyddd(d):
        raise s3io.Failed('hours: %s (date %s, expected %d)' % (hc.get('reason'), hc.get('date'), yyyyddd(d)))
    if res['n_stacks'] != g['n_stacks']:
        raise StackCountMismatch('stack count mismatch: emission file %s, stack groups file %s' % (res['n_stacks'], g['n_stacks']))
    miss = [sp for sp in species if sp not in res['means']]
    if miss:
        raise s3io.Failed('species missing: %s' % ', '.join(miss))
    arrays = {}
    info = None
    for sp in species:
        grid, info = stacks.sum_to_cells(res['means'][sp], g['row'], g['col'], GRID[0], GRID[1], base=int(pm['base']))
        arrays[sp] = grid.astype(np.float32)
    units = [str(res['units'].get(sp)) for sp in species]
    arrays.update({'species': np.array(species), 'units': np.array(units), 'hours': np.array('00-23 UTC'), 'date': np.array(d), 'sector': np.array(sec), 'member': np.array(name), 'n_stacks': np.array(info['n_stacks'], dtype=np.int64), 'n_summed': np.array(info['n_summed'], dtype=np.int64), 'n_outside': np.array(info['n_outside'], dtype=np.int64), 'base': np.array(int(pm['base']), dtype=np.int64), 'tflag_date': np.array(hc['date'], dtype=np.int64)})
    nb, sha = write_npz(path, arrays)
    u, loc = common.now_pair()
    h = ctx.head(key)
    ctx.rows.append(PT_DIR, path=path, source='s3://%s/%s (member %s)' % (s3io.BUCKET, key, name), byte_range='%d-%d' % (m['data_offset'], m['data_offset'] + m['size'] - 1), retrieved_utc=u, retrieved_local=loc, http_status=206, bytes=nb, sha256=sha, version=VERSION_EMIS, etag=h['etag'], last_modified=h['last_modified'], species=' '.join(species), hours='00-23 UTC', member_sha256=msha, member_note=mnote, note='per-stack daily means of records 0 to 23 (float64) summed into cells by choi.stacks (ROW and COL %d-based, stack groups file %s), stored as float32; %d stacks, %d summed, %d outside the grid; %s route; units as the file states: %s; TFLAG %d hours 00-23 ok' % (int(pm['base']), pm['groups_members']['%s_%s' % (sec, spec['year'])].split('/')[-1], info['n_stacks'], info['n_summed'], info['n_outside'], pm['route'], ', '.join(('%s %s' % x for x in zip(species, units))), hc['date']))
    return (nbytes, '%s; %d stacks, %d outside; extract %d bytes' % (pm['route'], info['n_stacks'], info['n_outside'], nb))
HANDLERS = {'mcip': do_mcip, 'emis_merged': do_emis, 'stackgroups': do_stackgroups, 'pt_marine': do_pt}

def _copy(part):
    try:
        prov.copy_to_run(FOLDER[part], COPY_NAME[part])
        prov.copy_to_run(S3META, 's3_s3meta')
    except Exception:
        pass

def run(plan, progress, dry_run=False, limit=None, parts=None, item_ids=None):
    common.set_temp()
    items = build_items(plan)
    rows = Rows()
    result = {'done': 0, 'skipped': 0, 'failed': [], 'refused': [], 'bytes': 0, 'held': [], 'held_reason': {}}
    sel = [it for it in items if (not parts or it[0] in parts) and (not item_ids or it[1] in item_ids)]
    earlier = read_refused()
    earlier_set = set(earlier)
    result['refused'] = list(earlier)
    if dry_run:
        out = []
        for part in PARTS:
            its = [it for it in sel if it[0] == part]
            ref = [it[1] for it in its if it[1] in earlier_set]
            done = [it[1] for it in its if it[1] not in earlier_set and rows.verified(FOLDER[part], item_path(part, it[2]))]
            out.append({'part': part, 'items': len(its), 'already_done': len(done), 'refused_earlier': len(ref), 'to_fetch': len(its) - len(done) - len(ref), 'first': its[0][1] if its else None, 'last': its[-1][1] if its else None, 'done_ids': done[:40]})
        result['parts'] = out
        result['skipped'] = sum((p['already_done'] for p in out))
        result['dry_run'] = True
        result['stop'] = 'three refusals' if len(earlier) >= MAX_REFUSALS else None
        return result
    ctx = Ctx(plan, progress, rows)
    stop = None
    refusals = len(earlier)
    dropped = {}

    def hold(iid, why):
        result['held'].append(iid)
        result['held_reason'][iid] = why
    try:
        if earlier:
            progress.note('REFUSED EARLIER: %d item(s) listed in %s are not asked: %s' % (len(earlier), common.rel(REFUSED_FILE), ', '.join(earlier[:20])))
        if refusals >= MAX_REFUSALS:
            stop = 'three refusals'
            progress.note('STOP S3 queue: three refusals (%d recorded earlier); no item is asked' % refusals)
        for part in PARTS:
            its = [it for it in sel if it[0] == part]
            if not its:
                continue
            progress.stage('s3_' + part, len(its))
            fetched = 0
            since = 0
            for p, iid, spec in its:
                if iid in earlier_set:
                    progress.item('s3_' + p, iid, 'refused', 0, 0.0, 'refused earlier')
                    continue
                if stop:
                    hold(iid, 'held: the S3 queue stopped (%s)' % stop)
                    continue
                if part == 'emis_merged' and ctx.hangs_row >= MAX_HANGS:
                    hold(iid, HELD_HANGS)
                    continue
                path = item_path(p, spec)
                if rows.verified(FOLDER[p], path):
                    result['skipped'] += 1
                    progress.item('s3_' + p, iid, 'skipped', 0, 0.0, 'row re-verifies')
                    continue
                if p == 'pt_marine' and spec.get('sector') in dropped:
                    hold(iid, HELD_SECTOR)
                    continue
                if limit is not None and fetched >= limit:
                    hold(iid, 'held: limit (tests only)')
                    continue
                gb = free_gb()
                if gb < DISK_FLOOR_GB:
                    result['failed'].append(iid)
                    progress.item('s3_' + p, iid, 'failed', 0, 0.0, 'disk: %.2f GB free on the work disk (floor %.0f); the S3 queue stops' % (gb, DISK_FLOOR_GB))
                    stop = 'disk'
                    continue
                t0 = time.time()
                fetched += 1
                try:
                    nb, note = HANDLERS[p](ctx, iid, spec, path)
                    result['done'] += 1
                    result['bytes'] += int(nb)
                    progress.item('s3_' + p, iid, 'done', nb, time.time() - t0, note)
                except s3io.Refused as e:
                    refusals += 1
                    result['refused'].append(iid)
                    try:
                        write_refused(iid, refusal_kind(e))
                    except Exception as we:
                        progress.note('WARNING: the refused-items file %s was not written for %s: %s' % (common.rel(REFUSED_FILE), iid, type(we).__name__))
                    progress.item('s3_' + p, iid, 'refused', 0, time.time() - t0, str(e)[:300])
                    if refusals >= MAX_REFUSALS and (not stop):
                        stop = 'three refusals'
                        progress.note('STOP S3 queue: three refusals (%d recorded earlier)' % len(earlier))
                except Exception as e:
                    hang = isinstance(e, s3io.Hang)
                    if hang:
                        ctx.hangs += 1
                        ctx.hangs_row += 1
                    result['failed'].append(iid)
                    progress.item('s3_' + p, iid, 'failed', 0, time.time() - t0, '%s: %s' % (type(e).__name__, str(e)[:300]))
                    if isinstance(e, StackCountMismatch) and spec.get('sector') not in dropped:
                        dropped[spec['sector']] = iid
                        progress.note('DROP sector %s: stack count mismatch on %s; every member of %s not yet read, in both years, is held, not fetched' % (spec['sector'], iid, spec['sector']))
                    if hang and part == 'emis_merged' and (ctx.hangs_row >= MAX_HANGS):
                        progress.note("STOP merged part: %d hangs in a row of the selective reader; every merged day not yet read is held (the result's held list, for the later refetch)" % ctx.hangs_row)
                since += 1
                if since >= COPY_EVERY:
                    _copy(p)
                    since = 0
            _copy(part)
    finally:
        ctx.close()
        result['s3'] = ctx.client.summary()
        result['s3']['child_requests'] = ctx.child_requests
        result['s3']['child_bytes'] = ctx.child_bytes
        result['hangs'] = ctx.hangs
        result['stop'] = stop
    return result
