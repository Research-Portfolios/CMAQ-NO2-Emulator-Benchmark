"""Fetches named point and marine items again through the downloader's own S3 route, one item per call, within a two-hour cap kept in a state file and with its own progress and status files; a refusal stops that item and every later one."""
import json
import os
import sys
import time
CODE = '/content/emul/run/code'
if CODE not in sys.path:
    sys.path.insert(0, CODE)
from choi_dl import common, dl_s3
from choi_dl.progress import Progress
PLAN_S = os.path.join(common.RUN_CONFIG, 'S2A_PLAN_S3.json')
PROGRESS = os.path.join(common.RUN_LOGS, 'S2C_REFETCH_ASCII.txt')
OUT = os.path.join(common.W_TMP, 's2c')
STATUS = os.path.join(OUT, 'refetch_status.json')
STATE = os.path.join(OUT, 'refetch_state.json')
RESULT = os.path.join(OUT, 'refetch_result.json')
CAP_S = 2 * 3600.0

def load_state():
    if os.path.exists(STATE):
        with open(STATE, 'rb') as f:
            return json.loads(f.read().decode('ascii'))
    return {}

def save_state(st):
    os.makedirs(OUT, exist_ok=True)
    with open(STATE, 'wb') as f:
        f.write(json.dumps(st, sort_keys=True).encode('ascii'))

def avail_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / 1000000000.0
    except Exception:
        return -1.0

def main(argv):
    common.set_temp()
    os.makedirs(OUT, exist_ok=True)
    ids = [a for a in argv if not a.startswith('--')]
    with open(PLAN_S, 'rb') as f:
        plan = json.loads(f.read().decode('ascii'))
    known = {m['id']: m for m in plan['pt_marine']['members']}
    unknown = [i for i in ids if i not in known]
    if unknown or not ids:
        print('not point and marine items of the plan: %s' % unknown)
        return 2
    st = load_state()
    p = Progress(progress_path=PROGRESS, status_path=STATUS)
    p.note('S2C REFETCH start (attempt 2); items %s; route %s; cap 2 h from the first fetch%s' % (', '.join(ids), plan['pt_marine']['route'], '; first fetch at %s' % st['first_fetch_local'] if st.get('first_fetch_local') else ''))
    results = {}
    if os.path.exists(RESULT):
        with open(RESULT, 'rb') as f:
            results = json.loads(f.read().decode('ascii'))
    stop = None
    for iid in ids:
        m = known[iid]
        now = time.time()
        if stop:
            p.note('NOT TRIED %s: %s' % (iid, stop))
            results.setdefault(iid, []).append({'outcome': 'not tried', 'reason': stop})
            continue
        if st.get('first_fetch') and now - st['first_fetch'] > CAP_S:
            p.note('NOT TRIED %s: the 2-hour cap from the first fetch (%s) has passed' % (iid, st['first_fetch_local']))
            results.setdefault(iid, []).append({'outcome': 'not tried', 'reason': 'cap'})
            continue
        if not st.get('first_fetch'):
            st['first_fetch'] = now
            st['first_fetch_local'] = common.now_pair()[1]
            st['first_fetch_utc'] = common.now_pair()[0]
            save_state(st)
        g0 = avail_gb()
        p.note('ATTEMPT %s (member %s, %d bytes, key %s); memory %.2f GB available; free disk %.2f GB' % (iid, m['name'].split('/')[-1], int(m['size']), m['key'].split('/')[-1], g0, dl_s3.free_gb()))
        t0 = time.time()
        t_local = common.now_pair()[1]
        try:
            r = dl_s3.run(plan, p, parts=('pt_marine',), item_ids=[iid])
            err = None
        except Exception as e:
            r = None
            err = '%s: %s' % (type(e).__name__, common.mask(str(e))[:200])
        sec = time.time() - t0
        if r is None:
            rec = {'outcome': 'error', 'error': err, 'start_local': t_local, 'seconds': round(sec, 1)}
        else:
            s3s = r.get('s3', {})
            outcome = 'done' if r['done'] else 'refused' if iid in r['refused'] else 'failed' if iid in r['failed'] else 'held' if iid in r['held'] else 'skipped' if r['skipped'] else 'unknown'
            rec = {'outcome': outcome, 'start_local': t_local, 'seconds': round(sec, 1), 'done': r['done'], 'skipped': r['skipped'], 'failed': r['failed'], 'refused': r['refused'], 'held': r['held'], 'held_reason': r.get('held_reason'), 'stop': r.get('stop'), 'bytes': r.get('bytes'), 'requests': s3s.get('requests'), 'request_bytes': s3s.get('bytes'), 'by_op': s3s.get('by_op'), 'retries': s3s.get('retries'), 'hosts': s3s.get('hosts'), 'hangs': r.get('hangs'), 'memory_gb_before': round(g0, 2), 'memory_gb_after': round(avail_gb(), 2)}
            if outcome == 'refused':
                stop = "a refusal of %s stops the stage's fetches" % iid
        results.setdefault(iid, []).append(rec)
        p.note('RESULT %s: %s in %.1f s; requests %s (%s), bytes %s; hosts %s; memory %.2f GB available after' % (iid, rec['outcome'], sec, rec.get('requests'), json.dumps(rec.get('by_op'), sort_keys=True), rec.get('request_bytes'), json.dumps(rec.get('hosts'), sort_keys=True), avail_gb()))
        with open(RESULT, 'wb') as f:
            f.write(json.dumps(results, sort_keys=True, indent=1).encode('ascii'))
    p.note('S2C REFETCH end; %s' % '; '.join(('%s %s' % (k, v[-1]['outcome']) for k, v in sorted(results.items()))))
    print(json.dumps(results, sort_keys=True, indent=1))
    return 0
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
