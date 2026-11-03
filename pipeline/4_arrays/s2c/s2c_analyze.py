"""Summarizes the verification results (read only): files and bytes by kind, RSIG items tried more than once, failed checks, decompressed-member checks, stack counts and stacks outside the grid, and, on request, the count of extract files whose field is zero everywhere."""
import collections
import json
import os
import re
import sys
CODE = '.\\run\\code'
if CODE not in sys.path:
    sys.path.insert(0, CODE)
from choi_dl import common
OUT = os.path.join(common.W_TMP, 's2c')

def main(argv):
    common.set_temp()
    tag = ''
    if '--tag' in argv:
        tag = '_' + re.sub('[^A-Za-z0-9]', '', argv[argv.index('--tag') + 1])
    with open(os.path.join(OUT, 'verify_results%s.json' % tag), 'rb') as f:
        d = json.loads(f.read().decode('ascii'))
    files = d['files']
    ent = d['entries']
    print('made %s; seconds %s; workers %s' % (d['made'], d['seconds'], d['workers']))
    nb = collections.Counter()
    nn = collections.Counter()
    for t in files.values():
        if t.get('size') is not None:
            nn[t['kind']] += 1
            nb[t['kind']] += t['size']
    for k in sorted(nn):
        print('BYTES %s: %d files hashed, %d bytes' % (k, nn[k], nb[k]))
    print('TOTAL hashed: %d files, %d bytes' % (sum(nn.values()), sum(nb.values())))
    gz = [t for t in files.values() if t.get('gz')]
    mb = sum((t.get('member_bytes_now') or 0 for t in gz))
    src = collections.Counter((t.get('want_member_from', 'none').split(' of ')[0] for t in gz))
    print('GZIP answers %d; decompressed bytes %d; member check source %s' % (len(gz), mb, dict(src)))
    for e in ent:
        if e['kind'] == 'rsig' and e.get('tries', 1) > 1:
            print('TRIES %s: uses %s' % (e['id'], e.get('use')))
            for p in e['files']:
                print('  %s: note %s' % (p.split('\\')[-1], files[p].get('note')))
    for t in files.values():
        if str(t.get('note', '')).startswith('check failed'):
            print('CHECK FAILED ROW %s (%s): %s' % (t['path'].split('\\')[-1], t['kind'], t.get('note')))
    st = collections.defaultdict(set)
    out = collections.defaultdict(set)
    for e in ent:
        if e['kind'] != 'pt_marine':
            continue
        t = files[e['files'][0]]
        z = t.get('npz') or {}
        k = '%s_%s' % (e['sector'], e['year'])
        if 'n_stacks' in z:
            st[k].add(z['n_stacks'])
        if 'n_outside' in z:
            out[k].add(z['n_outside'])
    for k in sorted(st):
        print('STACKS %s: n_stacks %s; n_outside %s' % (k, sorted(st[k]), sorted(out[k])))
    if '--zeros' in argv:
        import numpy as np
        zc = collections.Counter()
        tot = collections.Counter()
        for e in ent:
            if e['kind'] not in ('emis_merged', 'pt_marine'):
                continue
            t = files[e['files'][0]]
            if not t.get('verified'):
                continue
            grp = '%s %s' % (e['kind'], e['year']) if e['kind'] == 'emis_merged' else 'pt_marine %s %s' % (e['sector'], e['year'])
            with np.load(os.path.join(common.ROOT, t['path']), allow_pickle=False) as z:
                for sp in [str(x) for x in z['species']]:
                    tot[grp, sp] += 1
                    if not np.any(z[sp] != 0):
                        zc[grp, sp] += 1
        for k in sorted(tot):
            print('ZERO FIELDS %s %s: %d of %d files' % (k[0], k[1], zc[k], tot[k]))
    return 0
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
