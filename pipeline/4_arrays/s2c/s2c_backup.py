"""Copies each named file beside itself as a backup, never overwriting an earlier backup, and checks that the copy is not empty and that its size and sha256 equal the original's."""
import hashlib
import os
import shutil
import sys
ROOT = '.'

def rel(p):
    p = os.path.abspath(p)
    return p[len(ROOT) + 1:] if p.lower().startswith(ROOT.lower() + '\\') else '<outside>'

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def backup_name(p, reason):
    d, n = os.path.split(p)
    if n.lower().endswith('.nc.gz'):
        stem, ext = (n[:-6], n[-6:])
    else:
        stem, ext = os.path.splitext(n)
    base = os.path.join(d, '%s_Backup_PriorTo%s' % (stem, reason))
    cand = base + ext
    k = 2
    while os.path.exists(cand):
        cand = '%s_%d%s' % (base, k, ext)
        k += 1
    return cand

def main(argv):
    if len(argv) < 2:
        print('usage: <Reason> <file> ...')
        return 2
    reason = argv[0]
    if not reason.isalnum():
        print('the reason must be letters and digits only')
        return 2
    bad = 0
    for p in argv[1:]:
        if not os.path.isfile(p):
            print('NOT FOUND %s' % rel(p))
            bad += 1
            continue
        dst = backup_name(p, reason)
        shutil.copyfile(p, dst)
        s0, s1 = (os.stat(p).st_size, os.stat(dst).st_size)
        h0, h1 = (sha(p), sha(dst))
        ok = s1 > 0 and s0 == s1 and (h0 == h1)
        if not ok:
            bad += 1
        print('%s %s %d bytes sha16 %s (original %d bytes)' % ('OK' if ok else 'BAD', rel(dst), s1, h1[:16], s0))
    return 1 if bad else 0
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
