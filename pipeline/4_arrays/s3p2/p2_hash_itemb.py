"""Hashes the first pass's arrays and configuration files by sha256 and size, writes the list, and compares it with an earlier list to show that they stayed unchanged (read only)."""
import glob
import os
import sys
sys.path.insert(0, '.\\run\\code')
from s3p2 import p2common as P

def listing():
    files = []
    for top in (P.ARRAYS_B, os.path.join(P.W, 'work', 's3')):
        for dp, _dn, fn in os.walk(top):
            for f in fn:
                files.append(os.path.join(dp, f))
    files += [p for p in glob.glob(os.path.join(P.CFG_B, '*.json')) if os.path.isfile(p)]
    return sorted(files)

def main():
    a = sys.argv[1:]
    tag = a[0]
    P.ensure_dirs()
    out = {'tag': tag, 'start': P.now_local(), 'files': {}}
    for p in listing():
        st = os.stat(p)
        out['files'][P.common.rel(p)] = {'bytes': int(st.st_size), 'sha256': P.S.sha256_file(p)}
    out['end'] = P.now_local()
    out['n_files'] = len(out['files'])
    out['bytes'] = sum((v['bytes'] for v in out['files'].values()))
    dst = os.path.join(P.TMP_P2, 'itemb_hashes_%s.json' % tag)
    info = P.write_json(dst, out)
    print('%s: %d files, %d bytes, hashed %s to %s; %s sha16 %s' % (tag, out['n_files'], out['bytes'], out['start'][11:], out['end'][11:], info['path'], info['sha256'][:16]))
    if '--compare' in a:
        t0 = a[a.index('--compare') + 1]
        old = P.jload(os.path.join(P.TMP_P2, 'itemb_hashes_%s.json' % t0))
        fa, fb = (old['files'], out['files'])
        same = sorted((k for k in fa if k in fb and fa[k] == fb[k]))
        diff = sorted((k for k in fa if k in fb and fa[k] != fb[k]))
        gone = sorted((k for k in fa if k not in fb))
        new = sorted((k for k in fb if k not in fa))
        print('compare %s with %s: equal %d, differ %d, gone %d, new %d' % (t0, tag, len(same), len(diff), len(gone), len(new)))
        for k in diff + gone + new:
            print('  ' + P.common.mask(k))
        print('EQUAL' if not (diff or gone or new) else 'NOT EQUAL')
if __name__ == '__main__':
    P.run_main(main)
