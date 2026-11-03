"""Lists what is on disk under the working folder's raw, extract, work and Natural Earth subfolders (read only): files, bytes and counts by extension per folder, and every partial, failed or temporary file."""
import collections
import os
ROOT = '.'
W = os.path.join(ROOT, 'work')
TOPS = ('raw', 'extract', 'work', 'natural_earth')

def rel(p):
    return p[len(ROOT) + 1:] if p.lower().startswith(ROOT.lower() + '\\') else '<outside>'

def ext_of(name):
    n = name.lower()
    for e in ('.nc.gz', '.contents.txt', '.jsonl', '.partial', '.nc4', '.ncf', '.npz', '.tar', '.zip', '.txt', '.nc', '.json', '.failed_check'):
        if n.endswith(e):
            return e
    return os.path.splitext(n)[1] or '(none)'

def main():
    odd = []
    for top in TOPS:
        base = os.path.join(W, top)
        if not os.path.isdir(base):
            print('%s: NOT PRESENT' % rel(base))
            continue
        per = collections.OrderedDict()
        for dp, dn, fn in os.walk(base):
            dn.sort()
            n = 0
            b = 0
            ex = collections.Counter()
            for f in sorted(fn):
                p = os.path.join(dp, f)
                st = os.stat(p)
                n += 1
                b += st.st_size
                ex[ext_of(f)] += 1
                low = f.lower()
                if low.endswith(('.partial', '.failed_check', '.tmp')) or '_try' in low:
                    odd.append((rel(p), st.st_size))
            if n:
                per[rel(dp)] = (n, b, ex)
        tn = sum((v[0] for v in per.values()))
        tb = sum((v[1] for v in per.values()))
        print('== %s: %d files, %d bytes' % (rel(base), tn, tb))
        for k, (n, b, ex) in per.items():
            print('  %s: %d files, %d bytes; %s' % (k, n, b, ', '.join(('%s %d' % kv for kv in sorted(ex.items())))))
    tmp = os.path.join(W, 'tmp')
    if os.path.isdir(tmp):
        names = sorted(os.listdir(tmp))
        print('== %s: %d entries at the top' % (rel(tmp), len(names)))
    print('== odd names (.partial, .failed_check, .tmp, _try): %d' % len(odd))
    for p, s in odd:
        print('  %s %d' % (p, s))
if __name__ == '__main__':
    main()
