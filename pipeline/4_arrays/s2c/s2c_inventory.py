"""Builds the inventory of the downloaded inputs from the verification and refetch results: drops any point or marine sector not read in full for both years, stops if a required target, input, lag, MCIP or merged-emission file is missing or failed or the disk is short, and writes the inventory JSON."""
import collections
import json
import os
import re
import shutil
import sys
CODE = '/content/emul/run/code'
if CODE not in sys.path:
    sys.path.insert(0, CODE)
from choi_dl import common, prov
OUT = os.path.join(common.W_TMP, 's2c')
INV = os.path.join(common.RUN_CONFIG, 'S2C_INVENTORY.json')
PLAN_S = os.path.join(common.RUN_CONFIG, 'S2A_PLAN_S3.json')
DL_PROGRESS = [os.path.join(common.RUN_LOGS, n) for n in ('DL_PROGRESS_C5_ASCII.txt', 'DL_PROGRESS_RSIG_ASCII.txt')]
NEED_GB = {'arrays_s3_addendum_1b': 14.54, 'f6_emission_arrays': 2.0, 'predictions': 1.3, 'checkpoints_and_logs': 0.2}
MARGIN_GB = 30.0
SERIES_ROLE = {'T1': 'T-1 target', 'T2': 'T-2 check only', 'X6': 'X-6 input (daily maximum PBLH)'}
FOLDERS = (('rsig', 'raw\\rsig'), ('rsig_work', 'work\\rsig'), ('s3_mcip', 'raw\\mcip'), ('s3_s3meta', 'raw\\s3meta'), ('s3_stackgroups', 'raw\\emis\\stackgroups'), ('s3_emis_merged', 'extract\\emis_merged'), ('s3_pt_marine', 'extract\\pt_marine'), ('ne_natural_earth', 'natural_earth'))
SOURCE_OF = {'rsig': 'rsig', 'rsig_test': 'rsig', 'rsig_work': 'rsig', 'mcip': 's3', 's3meta': 's3', 'stackgroups': 's3', 'emis_merged': 's3', 'pt_marine': 's3', 'ne': 'natural_earth'}

def jload(p):
    with open(p, 'rb') as f:
        return json.loads(f.read().decode('ascii'))

def series_role(e):
    s = e['series']
    if e['sid'].startswith('LAG_'):
        return 'X-1 to X-5 lag input (2014-12-31, %s)' % s
    if s in SERIES_ROLE:
        return SERIES_ROLE[s]
    return '%s-%s input' % (s[0], s[1:])

def main():
    common.set_temp()
    vr = jload(os.path.join(OUT, 'verify_results_post.json'))
    plan = jload(PLAN_S)
    ref = jload(os.path.join(OUT, 'refetch_result.json'))
    state = jload(os.path.join(OUT, 'refetch_state.json'))
    files = vr['files']
    ent = vr['entries']
    prog = ''
    for pp in DL_PROGRESS:
        with open(pp, 'rb') as f:
            prog += f.read().decode('ascii', 'replace') + '\n'
    s09 = [ln for ln in prog.split('\n') if re.search('stack count mismatch|sector dropped|DROP sector', ln, re.I)]
    s07 = [ln for ln in prog.split('\n') if re.search('\\bHang\\b|hangs in a row|STOP merged', ln)]
    pm = plan['pt_marine']
    sec_years = collections.defaultdict(lambda: {'expected': 0, 'verified': 0})
    for e in ent:
        if e['kind'] == 'pt_marine':
            k = (e['sector'], e['year'])
            sec_years[k]['expected'] += 1
            sec_years[k]['verified'] += int(bool(e['verified']))
    kept, dropped = ([], {})
    for sec in pm['sectors_kept']:
        full = all((sec_years[sec, y]['expected'] > 0 and sec_years[sec, y]['verified'] == sec_years[sec, y]['expected'] for y in ('2017', '2019')))
        mism = [ln for ln in s09 if sec in ln]
        if full and (not mism):
            kept.append(sec)
        else:
            dropped[sec] = {'reason': 'not read in full for both years' if not full else 'stack count mismatch', 'members': {y: sec_years[sec, y] for y in ('2017', '2019')}, 'lines': mism}
    for sec, why in (pm.get('sectors_dropped') or {}).items():
        dropped.setdefault(sec, {'reason': 'dropped by the download plan: %s' % why})
    em_days = collections.defaultdict(list)
    pt_days = collections.defaultdict(lambda: collections.defaultdict(list))
    for e in ent:
        if e['kind'] == 'emis_merged' and e['verified']:
            em_days[e['year']].append(e['date'])
        if e['kind'] == 'pt_marine' and e['verified']:
            pt_days[e['sector']][e['year']].append(e['date'])
    em_plan = collections.defaultdict(list)
    for d in plan['emis_merged']['dates']:
        em_plan[d[:4]].append(d)
    stop = []
    for e in ent:
        if e['kind'] == 'rsig' and e['series'] != 'T2' and (not e['verified']):
            stop.append('RSIG %s missing or failing' % e['id'])
        if e['kind'] == 'mcip' and (not e['verified']):
            stop.append('MCIP member %s missing or failing' % e['id'])
        if e['kind'] == 'emis_merged' and (not e['verified']):
            stop.append('merged emissions of %s missing or failing (its month %s)' % (e['date'], e['date'][:6]))
    t2_bad = [e['id'] for e in ent if e['kind'] == 'rsig' and e['series'] == 'T2' and (not e['verified'])]
    du = shutil.disk_usage(common.W)
    need = round(sum(NEED_GB.values()), 2)
    free = du.free / 1000000000.0
    disk_ok = free >= need + MARGIN_GB
    if not disk_ok:
        stop.append('disk: %.2f GB free on the work disk, the disk rule asks %.2f GB' % (free, need + MARGIN_GB))
    verdict = 'STOP' if stop else 'PASS'
    by_kind = collections.OrderedDict()
    for e in ent:
        s = by_kind.setdefault(e['kind'], {'expected': 0, 'present': 0, 'verified': 0, 'files': 0, 'bytes': 0})
        s['expected'] += 1
        s['present'] += int(bool(e['present']))
        s['verified'] += int(bool(e['verified']))
    item_of = {}
    for e in ent:
        for p in e['files']:
            item_of[p] = e
    flist = []
    for p in sorted(files):
        t = files[p]
        e = item_of[p]
        k = t['kind']
        by_kind[k]['files'] += 1
        by_kind[k]['bytes'] += int(t.get('size') or 0)
        row = {'path': p, 'kind': k, 'sha256': t.get('sha256'), 'bytes': t.get('size'), 'verified': bool(t.get('verified'))}
        if k == 'rsig':
            row['item'] = e['id']
            row['series'] = e['series']
            row['period'] = e['id'].rsplit('_', 1)[-1]
            if p == e.get('use'):
                row['role'] = series_role(e)
                row['s3_input'] = True
            else:
                row['role'] = 'another try of %s, not used (%s)' % (e['id'], t.get('note'))
                row['s3_input'] = False
            row['decompressed_sha256'] = t.get('member_sha256_now')
            row['decompressed_bytes'] = t.get('member_bytes_now')
            row['steps'] = e['steps']
        elif k == 'rsig_test':
            row['role'] = 'test answer (not an input); note: %s' % t.get('note')
            row['s3_input'] = False
            row['decompressed_sha256'] = t.get('member_sha256_now')
            row['decompressed_bytes'] = t.get('member_bytes_now')
        elif k == 'rsig_work':
            row['role'] = 'a decompressed copy, kept with its row; S3 decompresses the raw answer instead'
            row['s3_input'] = False
        elif k == 'mcip':
            row['role'] = 'MCIP member (L-1 to L-3, L-5, geometry)'
            row['s3_input'] = True
        elif k == 's3meta':
            row['role'] = 'contents file of a tar (raw record)'
            row['s3_input'] = False
        elif k == 'stackgroups':
            row['role'] = 'stack groups tar (raw, kept; the point and marine extracts already hold the cell sums)'
            row['s3_input'] = False
        elif k == 'emis_merged':
            row['role'] = 'merged-emission daily means (E-1 to E-4)'
            row['s3_input'] = True
            row['date'] = e['date']
        elif k == 'pt_marine':
            row['sector'] = e['sector']
            row['date'] = e['date']
            row['s3_input'] = e['sector'] in kept
            row['role'] = 'point and marine cell sums of NO, NO2 and HONO (E-1 to E-3)' if row['s3_input'] else 'point and marine sector dropped'
            z = t.get('npz') or {}
            for kk in ('n_stacks', 'n_summed', 'n_outside'):
                if kk in z:
                    row[kk] = z[kk]
        elif k == 'ne':
            row['role'] = 'Natural Earth (the land mask; the city maps)'
            row['s3_input'] = True
        flist.append(row)
    by_source = collections.OrderedDict()
    for k, s in by_kind.items():
        src = by_source.setdefault(SOURCE_OF[k], {'expected': 0, 'present': 0, 'verified': 0, 'files': 0, 'bytes': 0})
        for f in ('expected', 'present', 'verified', 'files', 'bytes'):
            src[f] += s[f]
    records = []
    for name, folder in FOLDERS:
        d = os.path.join(common.W, *folder.split('\\'))
        for fn in ('PROVENANCE.jsonl', 'PROVENANCE_ASCII.txt'):
            p = os.path.join(d, fn)
            if not os.path.isfile(p):
                records.append({'path': common.rel(p), 'absent': True, 'note': 'no such record on this machine (nothing was written there)'})
                print('record %s absent: listed as absent' % common.rel(p))
                continue
            records.append({'path': common.rel(p), 'bytes': os.stat(p).st_size, 'sha256': prov.sha256_file(p), 'rows': len(prov.load(d)) if fn.endswith('jsonl') else None, 'copy': common.rel(os.path.join(common.RUN_PROV, '%s_%s' % (name, fn)))})
    u, loc = common.now_pair()
    inv = collections.OrderedDict()
    inv['run'] = 'emul'
    inv['item'] = 'the inventory'
    inv['stage'] = 'the inventory (attempt 2)'
    inv['made'] = [u, loc]
    inv['root'] = common.ROOT
    inv['w'] = common.rel(common.W)
    inv['design'] = 'the study design'
    inv['made_by'] = 'run\\code\\s2c\\s2c_inventory.py from W\\work/verify_results_post.json (s2c_verify.py --tag post)'
    inv['verified_at'] = vr['made']
    inv['stop'] = {'verdict': verdict, 'reasons': stop, 't2_missing_or_failing': t2_bad, 'disk': {'free_gb': round(free, 2), 'free_bytes': du.free, 'need_gb': need, 'need_parts_gb': NEED_GB, 'margin_gb': MARGIN_GB, 'asked_gb': round(need + MARGIN_GB, 2), 'met': disk_ok}}
    inv['counts_by_kind'] = by_kind
    inv['counts_by_source'] = by_source
    inv['sectors'] = {'kept': kept, 'dropped': dropped, 'rule': 'The point and marine route: a sector not read in full for both years, or with a stack count mismatch, is dropped for every day and year', 'members': {'%s_%s' % k: v for k, v in sorted(sec_years.items())}, 's09_lines': s09, 's07_lines': s07, 'stack_counts_plan': pm['stack_counts']}
    inv['day_sets'] = {'emis_merged': {y: {'days': len(v), 'plan_days': len(em_plan[y]), 'dates': sorted(v)} for y, v in sorted(em_days.items())}, 'emis_merged_day_set': plan['emis_merged']['day_set'], 'pt_marine': {s: {y: {'days': len(v), 'dates': sorted(v)} for y, v in sorted(ys.items())} for s, ys in sorted(pt_days.items())}, 'pt_marine_classes': "the plan's pt_marine.holiday_members, federal_holidays and members_per_month (run\\config\\S2A_PLAN_S3.json) give the weekday classes"}
    inv['refetch'] = {'items': ref, 'first_fetch': [state.get('first_fetch_utc'), state.get('first_fetch_local')], 'progress_file': 'run\\logs\\S2C_REFETCH_ASCII.txt'}
    inv['records'] = records
    inv['notes'] = ["RSIG answers are gzip'd netCDF-3 kept as received; S3 decompresses each as it builds the arrays; decompressed_sha256 and decompressed_bytes let S3 check the bytes it reads", "M4_metdot3d_UWIND_L1_DM_201607 uses its _try2 answer; the first try's row says 'check failed' (units spelling, before the fix of 18 Oct 03:40); it is listed with s3_input false", 'HONO is zero in every cell of every extract of ptegu (2017, 2019), pt_oilgas (2017, 2019) and ptnonipm 2017, as read from the files (s2c_analyze.py --zeros)', 'no stack lies outside the grid (n_outside 0 in every extract)']
    inv['files'] = flist
    data = json.dumps(inv, indent=1).encode('ascii')
    if b'@' in data:
        raise ValueError('the inventory would hold an at sign')
    tmp = INV + '.partial'
    with open(tmp, 'wb') as f:
        f.write(data)
    os.replace(tmp, INV)
    print('wrote %s: %d bytes, sha16 %s; %d files' % (common.rel(INV), os.stat(INV).st_size, prov.sha256_file(INV)[:16], len(flist)))
    print('STOP verdict %s; reasons %s; T-2 missing %s' % (verdict, stop, t2_bad))
    print('disk free %.2f GB; need %.2f + %.0f = %.2f; met %s' % (free, need, MARGIN_GB, need + MARGIN_GB, disk_ok))
    print('sectors kept %s; dropped %s; stack-mismatch or dropped-sector lines %d; hang lines %d' % (kept, dropped, len(s09), len(s07)))
    for k, s in by_kind.items():
        print('KIND %s: %s' % (k, json.dumps(s, sort_keys=True)))
    for k, s in by_source.items():
        print('SOURCE %s: %s' % (k, json.dumps(s, sort_keys=True)))
    print('s3_input files %d; others %d' % (sum((1 for r in flist if r.get('s3_input'))), sum((1 for r in flist if not r.get('s3_input')))))
    for y, v in sorted(em_days.items()):
        print('EMIS %s: %d days (plan %d)' % (y, len(v), len(em_plan[y])))
    for s, ys in sorted(pt_days.items()):
        print('PT %s: %s' % (s, ', '.join(('%s %d' % (y, len(v)) for y, v in sorted(ys.items())))))
    return 0
if __name__ == '__main__':
    sys.exit(main())
