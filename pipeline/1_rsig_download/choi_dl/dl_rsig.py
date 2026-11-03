"""Runs the queue of EPA RSIG requests in a fixed order: the CMAQ NO2 target and the meteorological series month by month for 2015 to 2019, the lag requests for 31 December 2014 and a second daily NO2 series for 2016. It skips items whose saved answer verifies, records and limits refusals, stops when disk runs short, and copies the provenance records."""
import os
import threading
import time
import traceback
from choi_dl import common, prov, rsig
from choi_dl.progress import append_bytes, retry
QUEUE = 'rsig'
DISK_FLOOR_GB = 30.0
REFUSAL_PAUSE_S = 600.0
MAX_REFUSALS = 3
COPY_EVERY = 20
ORDER = ('T1', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'X6')
YEARS = (2015, 2016, 2017, 2018, 2019)
REFUSED_FILE = os.path.join(common.WORK, 'dl_refused_rsig.txt')
REFUSED_LOCK = threading.Lock()

def build(plan=None):
    items = []
    for key in ORDER:
        for y in YEARS:
            for m in range(1, 13):
                items.append(rsig.month_item(key, y, m))
    for key in rsig.LAG_KEYS:
        items.append(rsig.lag_item(key))
    for m in range(1, 13):
        items.append(rsig.month_item('T2', 2016, m))
    return items

def held_ids(plan, items):
    keys = set(plan.get('held_series') or [])
    if plan.get('m3_held'):
        keys.add('M3')
    return [it['id'] for it in items if it['key'] in keys]

def copy_records():
    with rsig.PROV_LOCK:
        a = prov.copy_to_run(rsig.RAW_RSIG, 'rsig')
        b = prov.copy_to_run(rsig.WORK_RSIG, 'rsig_work')
    return a + b

def _quiet(fn, *args, **kw):
    try:
        return fn(*args, **kw)
    except Exception:
        return None

def read_refused(path=None):
    path = REFUSED_FILE if path is None else path
    if not os.path.exists(path):
        return []
    box = {}

    def _read():
        with open(path, 'rb') as f:
            box['b'] = f.read()
    if not retry(_read):
        raise OSError('the refused-items file %s could not be read' % common.rel(path))
    ids = []
    for line in box['b'].decode('ascii', 'replace').split('\n'):
        iid = line.split('\t')[0].strip()
        if iid and iid not in ids:
            ids.append(iid)
    return ids

def refusal_kind(r):
    st = r.get('http_status')
    if st is not None and str(st) != '200':
        return str(st)
    return 'not-data'

def write_refused(item_id, kind, path=None):
    path = REFUSED_FILE if path is None else path
    try:
        u, loc = common.now_pair()
        fields = []
        for v in (item_id, kind, u, loc):
            s = common.mask(v)
            for c in ('\t', '\r', '\n'):
                s = s.replace(c, ' ')
            fields.append(s)
        data = ('\t'.join(fields) + '\n').encode('ascii', 'replace')
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        return False
    with REFUSED_LOCK:
        return append_bytes(path, data)

def dry(plan, items, held):
    refused = read_refused()
    rs = set(refused)
    by = {}
    done_ids = []
    hs = set(held)
    for it in items:
        s = by.setdefault(it['sid'], {'items': 0, 'done': 0, 'held': 0})
        s['items'] += 1
        if it['id'] in hs:
            s['held'] += 1
            continue
        if it['id'] in rs:
            continue
        if rsig.item_done(it):
            s['done'] += 1
            done_ids.append(it['id'])
    active = len(items) - len(hs)
    refused_active = len([it for it in items if it['id'] in rs and it['id'] not in hs])
    return {'dry_run': True, 'total': active, 'held_count': len(hs), 'already_done': len(done_ids), 'to_fetch': active - len(done_ids) - refused_active, 'by_series': by, 'done_items': done_ids, 'held': list(held), 'first': items[0]['id'], 'last': items[-1]['id'], 'done': 0, 'skipped': len(done_ids), 'failed': [], 'refused': refused, 'refused_earlier': len(refused), 'stop_at_start': 'three refusals' if len(refused) >= MAX_REFUSALS else None, 'bytes': 0, 'stop': None}

def run(plan, progress, dry_run=False, limit=None):
    items = build(plan)
    held = held_ids(plan, items)
    if dry_run:
        return dry(plan, items, held)
    hs = set(held)
    pending = [it for it in items if it['id'] not in hs]
    conc = int(plan.get('concurrency', 1))
    if conc not in (1, 2):
        raise ValueError('concurrency must be 1 or 2, not %r' % conc)
    ua = plan.get('user_agent') or rsig.USER_AGENT
    if chr(64) in ua or '%' + '40' in ua.lower():
        raise ValueError('the User-Agent holds an at sign')
    dsha = plan['describe_sha256']
    out = {'done': 0, 'skipped': 0, 'failed': [], 'refused': [], 'bytes': 0, 'held': list(held), 'stop': None, 'done_ids': []}
    state = {'refusals': 0, 'pause_until': 0.0, 'processed': 0, 'fetches': 0}
    lock = threading.Lock()
    gate = rsig.Gate(conc)
    _quiet(progress.stage, QUEUE, len(pending))
    _quiet(progress.note, 'RSIG queue: %d items (%d held), concurrency %d, day kind %s, limit %s' % (len(pending), len(held), conc, plan.get('daily_mean_kind'), limit))
    if held:
        keys = sorted(set((it['key'] for it in items if it['id'] in hs)))
        _quiet(progress.note, 'HELD %d items of the series %s (each with its lag request), left out for a decision: m3_held %s, held_series %s (the download plan)' % (len(held), ','.join(keys), plan.get('m3_held'), plan.get('held_series')))

    def counted():
        state['processed'] += 1
        if state['processed'] % COPY_EVERY == 0:
            _quiet(copy_records)

    def worker():
        while True:
            with lock:
                if out['stop'] or not pending:
                    return
                it = pending.pop(0)
            try:
                if rsig.item_done(it):
                    with lock:
                        out['skipped'] += 1
                        counted()
                    _quiet(progress.item, QUEUE, it['id'], 'skipped', 0, None, 'verified')
                    continue
                with lock:
                    if limit is not None and state['fetches'] >= limit:
                        out['stop'] = out['stop'] or 'limit %d' % limit
                        pending.insert(0, it)
                        return
                    state['fetches'] += 1
                    wait = state['pause_until'] - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                with lock:
                    if out['stop']:
                        pending.insert(0, it)
                        return
                    gb = rsig.free_gb()
                    if gb < DISK_FLOOR_GB:
                        out['failed'].append(it['id'])
                        out['stop'] = 'disk'
                        counted()
                if gb < DISK_FLOOR_GB:
                    _quiet(progress.item, QUEUE, it['id'], 'failed', 0, None, 'disk: %.2f GB free on the work disk (floor %.0f)' % (gb, DISK_FLOOR_GB))
                    return
                r = rsig.fetch(it, gate, dsha[it['coverage']], user_agent=ua, log=progress.note)
                st = r['status']
                nref = 0
                with lock:
                    out['bytes'] += int(r['bytes'] or 0)
                    if st == 'done':
                        out['done'] += 1
                        out['done_ids'].append(it['id'])
                    elif st == 'refused':
                        out['refused'].append(it['id'])
                        state['refusals'] += 1
                        nref = state['refusals']
                        state['pause_until'] = time.monotonic() + REFUSAL_PAUSE_S
                        if state['refusals'] >= MAX_REFUSALS:
                            out['stop'] = 'three refusals'
                    else:
                        out['failed'].append(it['id'])
                    counted()
                if st == 'refused' and (not write_refused(it['id'], refusal_kind(r))):
                    _quiet(progress.note, 'REFUSED-ITEMS FILE NOT WRITTEN for %s (%s): a restart would ask it again' % (it['id'], common.rel(REFUSED_FILE)))
                _quiet(lambda: progress.item(QUEUE, it['id'], st, r.get('bytes'), r.get('seconds'), str(r.get('note') or '')[:160]))
                if st == 'refused':
                    _quiet(progress.note, 'REFUSAL %d of %d: %s; pause %.0f s' % (nref, MAX_REFUSALS, it['id'], REFUSAL_PAUSE_S))
            except Exception as e:
                with lock:
                    out['failed'].append(it['id'])
                    counted()
                try:
                    tb = traceback.format_exc().splitlines()[-3:]
                    progress.item(QUEUE, it['id'], 'failed', 0, None, 'error %s: %s' % (type(e).__name__, common.mask(str(e))[:100]))
                    for t in tb:
                        progress.note('  ' + t.strip()[:150])
                except Exception:
                    pass
    try:
        try:
            refused_before = read_refused()
        except Exception as e:
            refused_before = None
            out['failed'].extend((it['id'] for it in pending))
            out['stop'] = 'refused-items file not readable; %d items not tried' % len(pending)
            _quiet(progress.note, 'STOP RSIG queue: %s not readable (%s: %s); %d items not tried, counted failed' % (common.rel(REFUSED_FILE), type(e).__name__, common.mask(str(e))[:100], len(pending)))
        if refused_before:
            rb = set(refused_before)
            pending[:] = [it for it in pending if it['id'] not in rb]
            for iid in refused_before:
                out['refused'].append(iid)
                _quiet(progress.item, QUEUE, iid, 'refused', 0, None, 'refused earlier')
            state['refusals'] = len(refused_before)
            _quiet(progress.note, 'REFUSED EARLIER %d items (%s): not asked again; refusals %d of %d' % (len(refused_before), common.rel(REFUSED_FILE), state['refusals'], MAX_REFUSALS))
            if state['refusals'] >= MAX_REFUSALS:
                out['stop'] = 'three refusals'
                _quiet(progress.note, 'STOP RSIG queue: three refusals (earlier); no request made')
        if not out['stop']:
            if conc == 1:
                worker()
            else:
                ts = [threading.Thread(target=worker, name='rsig-%d' % i) for i in range(conc)]
                for t in ts:
                    t.start()
                for t in ts:
                    t.join()
            if pending and (not out['stop']):
                out['failed'].extend((it['id'] for it in pending))
                out['stop'] = 'workers ended; %d items not tried' % len(pending)
    finally:
        _quiet(copy_records)
        _quiet(progress.note, 'RSIG queue end: done %d skipped %d failed %d refused %d bytes %d held %d; stop %s' % (out['done'], out['skipped'], len(out['failed']), len(out['refused']), out['bytes'], len(held), out['stop']))
    return out
