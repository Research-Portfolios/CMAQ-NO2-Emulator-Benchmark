"""Copies each named file beside itself as a backup, never overwriting an earlier backup, and checks that the copy has the same size and sha256."""
import hashlib
import os
import shutil
import sys
ROOT = '/content/emul'

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    if len(sys.argv) < 3:
        sys.exit('usage: REASON PATH...')
    reason = sys.argv[1]
    bad = 0
    for p in sys.argv[2:]:
        src = p if len(p) > 2 and p[1] == ':' else os.path.join(ROOT, p)
        if not os.path.isfile(src):
            print('MISSING %s' % p)
            bad += 1
            continue
        stem, ext = os.path.splitext(src)
        base = '%s_Backup_PriorTo%s' % (stem, reason)
        dest, k = (base + ext, 1)
        while os.path.exists(dest):
            k += 1
            dest = '%s_%d%s' % (base, k, ext)
        shutil.copyfile(src, dest)
        s1, s2 = (os.stat(src).st_size, os.stat(dest).st_size)
        h1, h2 = (sha256(src), sha256(dest))
        if s2 == 0 or s1 != s2 or h1 != h2:
            print('FAILED %s' % os.path.relpath(dest, ROOT))
            bad += 1
            continue
        print('backup %s (%d bytes, sha16 %s)' % (os.path.relpath(dest, ROOT), s2, h2[:16]))
    sys.exit(1 if bad else 0)
if __name__ == '__main__':
    main()
