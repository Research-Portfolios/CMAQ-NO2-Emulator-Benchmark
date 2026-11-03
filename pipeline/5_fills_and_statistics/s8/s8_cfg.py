"""Updates the index files after a year's emission channels are filled: marks the year built, records its round-trip results and the new array's size and sha256, and keeps the earlier entries under new keys, after checking the year's verification results and the configuration backups."""
import copy
import os
import sys
sys.path.insert(0, '/content/emul/run/code')
sys.path.insert(0, '/content/emul/run/code/s8')
import s8common as C
from s3 import s3common as S

def backup_ok(path, reason):
    stem, ext = os.path.splitext(path)
    b = '%s_Backup_PriorTo%s%s' % (stem, reason, ext)
    if not os.path.isfile(b):
        raise SystemExit('no backup ' + S.common.rel(b))
    if os.stat(b).st_size != os.stat(path).st_size or C.sha256_file(b) != C.sha256_file(path):
        raise SystemExit('the backup differs from the file now: ' + S.common.rel(b))
    return {'path': S.common.rel(b), 'bytes': os.stat(b).st_size, 'sha256': C.sha256_file(b)}

def main():
    y = sys.argv[1]
    yi = int(y)
    p = C.paths(y)
    after = C.jload(p['work']('after_s8_%s.json'))
    ver = C.jload(p['work']('verify_s8_%s.json'))
    rtd = C.jload(p['work']('roundtrip_s8_%s.json'))
    if after.get('verdict') != 'PASS' or ver.get('verdict') != 'PASS':
        raise SystemExit('after %s, verify %s: no config is changed' % (after.get('verdict'), ver.get('verdict')))
    sha = C.sha256_file(p['array'])
    nbytes = os.stat(p['array']).st_size
    if sha != after['sha256'] or nbytes != after['bytes']:
        raise SystemExit("the array differs from after_s8's record")
    reason = 'S8_%s' % y
    chp = os.path.join(C.CFG_P2, 'S3_CHANNELS.json')
    arp = os.path.join(C.CFG_P2, 'S3_ARRAYS.json')
    bk = {'S3_CHANNELS.json': backup_ok(chp, reason), 'S3_ARRAYS.json': backup_ok(arp, reason)}
    made = C.now()
    report = S.common.rel(p['report'])
    ch = C.jload(chp)
    changed = []
    for e in ch['channels']:
        if e['name'] not in C.EMIS:
            continue
        if yi in e['built_years'] or yi not in e['waiting_years']:
            raise SystemExit('%s: %s already built or not waiting' % (e['name'], y))
        e['built_years'] = sorted(set(e['built_years']) | {yi})
        e['waiting_years'] = [w for w in e['waiting_years'] if w != yi]
        by = dict(e['roundtrip_by_year'])
        if y in by:
            raise SystemExit('%s: roundtrip_by_year.%s exists' % (e['name'], y))
        r = copy.deepcopy(rtd['channels'][e['name']])
        want = list(next(iter(by.values())).keys())
        if list(r.keys()) != want:
            raise SystemExit('%s: round-trip keys differ from the form there: %s' % (e['name'], list(r.keys())))
        by[y] = r
        e['roundtrip_by_year'] = {k: by[k] for k in sorted(by)}
        changed.append(e['name'])
    if sorted(changed) != sorted(C.EMIS):
        raise SystemExit('channels changed: %s' % changed)
    py = ch['partial_years']
    if y not in py['unfilled_channels']:
        raise SystemExit('partial_years.unfilled_channels lacks ' + y)
    before_unf = py['unfilled_channels'].pop(y)
    py.setdefault('filled', {})[y] = {'channels': before_unf, 'by': "the fills stage; E-1 to E-4 from each day's merged extract, P-1 from the extracts' P1_NOX by the ONE DEFINITION, the representative-day sectors by the weekday classes", 'made': made, 'inputs_f32_sha256': sha, 'report': report}
    ch_info = C.write_json(chp, ch)
    ar = C.jload(arp)
    f = ar['years'][y]['files']['inputs_f32']
    if not f.get('partial') or f.get('sha256') != after['sha256_before']:
        raise SystemExit("S3_ARRAYS.json's inputs_f32 of %s is not the partial state this stage began from" % y)
    f['partial_before'] = {'partial': True, 'sha256': f['sha256'], 'bytes': f['bytes'], 'unfilled_channels': f['unfilled_channels'], 'unfilled_indices': f['unfilled_indices'], 'unfilled_fill': f['unfilled_fill'], 'copy': 'none kept on Colab: the partial array was PROJECT_DATA/EMUL/W_P2/arrays_p2/inputs_f32_%s.npy on the Drive (the arrays stage, stage (1), rebuilds it), replaced there by the completed array after its after and verify steps (no extra copy)' % y}
    f['sha256'] = sha
    f['bytes'] = int(nbytes)
    f['partial'] = False
    f['unfilled_channels'] = []
    f['unfilled_indices'] = []
    f['unfilled_fill'] = 'none: stage (8) filled E-1 to E-4 and P-1 on every day (partial_before holds the partial state)'
    f['completed'] = {'stage': 'the fills stage, completed in place (as decided for stage (8))', 'made': made, 'report': report}
    ar['channels_file_before_s8_%s' % y] = ar['channels_file']
    ar['channels_file'] = ch_info
    ar_info = C.write_json(arp, ar)
    lst = {'run': 'emul', 'stage': "the fills stage, %s, for the later stages' copy" % y, 'made': made, 'design': 'the list of files that the later stages copy; the completed main float32 array of %s only' % y, 'files': [{'path': after['array'], 'bytes': int(nbytes), 'sha256': sha, 'kind': 'inputs_f32', 'sha256_checked_against_record': True, 'year': yi, 'shape': f['shape'], 'dtype': f['dtype'], 'partial': False}], 'n_files': 1, 'bytes': int(nbytes), 'partial': [], 'configs_by_outbox': [{'path': ch_info['path'], 'bytes': ch_info['bytes'], 'sha256': ch_info['sha256']}, {'path': ar_info['path'], 'bytes': ar_info['bytes'], 'sha256': ar_info['sha256']}]}
    li = C.write_json(os.path.join(C.CFG_P2, 'S3C_LIST_8_%s.json' % y), lst)
    for k, v in bk.items():
        print('backup of %s: %s (%d bytes, sha16 %s)' % (k, v['path'], v['bytes'], v['sha256'][:16]))
    for k, v in (('S3_CHANNELS.json', ch_info), ('S3_ARRAYS.json', ar_info), ('S3C_LIST_8', li)):
        print('written %s (%d bytes, sha16 %s)' % (v['path'], v['bytes'], v['sha256'][:16]))
if __name__ == '__main__':
    S.run_main(main)
