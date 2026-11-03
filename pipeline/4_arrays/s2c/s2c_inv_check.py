"""Checks the inventory file (read only): 7-bit ASCII with LF line ends, valid JSON, a unique path, a sha256 and a byte count for every entry, and every input verified; prints its stop section and its counts by kind."""
import collections
import json
import os
import re
P = '.\\run\\config\\S2C_INVENTORY.json'

def main():
    with open(P, 'rb') as f:
        b = f.read()
    print('bytes %d; ascii %s; CR %d; at signs %d' % (len(b), all((x < 128 for x in b)), b.count(b'\r'), b.count(b'@')))
    d = json.loads(b.decode('ascii'))
    print('keys: %s' % ', '.join(d))
    print('stop: %s' % json.dumps(d['stop'], sort_keys=True))
    for k, v in d['counts_by_kind'].items():
        print('kind %s: %s' % (k, json.dumps(v, sort_keys=True)))
    fl = d['files']
    paths = [r['path'] for r in fl]
    hexok = all((re.match('^[0-9a-f]{64}$', str(r.get('sha256', ''))) for r in fl))
    nb = all((isinstance(r.get('bytes'), int) and r['bytes'] > 0 for r in fl))
    unver = [r['path'] for r in fl if r.get('s3_input') and (not r.get('verified'))]
    print('files %d; unique paths %d; sha256 all hex %s; bytes all > 0 %s; s3_input %d; s3_input not verified %d' % (len(fl), len(set(paths)), hexok, nb, sum((1 for r in fl if r.get('s3_input'))), len(unver)))
    seen = collections.OrderedDict()
    for r in fl:
        seen.setdefault(r['kind'], r)
    for k, r in seen.items():
        print('sample %s: %s' % (k, json.dumps(r, sort_keys=True)[:400]))
    for r in fl:
        if 'M4_metdot3d_UWIND_L1_DM_201607' in r['path'] or r['path'].endswith(('pt_othpt_20170109.npz', 'pt_othpt_20170114.npz')):
            print('entry: %s' % json.dumps(r, sort_keys=True))
    print('sectors kept %s; dropped %s' % (d['sectors']['kept'], d['sectors']['dropped']))
    ds = d['day_sets']
    print('emis days %s; pt days %s' % ({y: v['days'] for y, v in ds['emis_merged'].items()}, {s: {y: v['days'] for y, v in ys.items()} for s, ys in ds['pt_marine'].items()}))
    print('refetch first fetch %s' % d['refetch']['first_fetch'])
    return 0
if __name__ == '__main__':
    main()
