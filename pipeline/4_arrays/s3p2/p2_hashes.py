"""Prints the path, size and sha256 of the files named or matched by patterns (read only)."""
import glob
import os
import sys
sys.path.insert(0, '.\\run\\code')
from choi_dl import common
from s3 import s3common as S

def main():
    a = sys.argv[1:]
    full = '--full' in a
    pats = [x for x in a if x != '--full']
    n = 0
    for pat in pats:
        for p in sorted(glob.glob(pat)):
            if not os.path.isfile(p):
                continue
            h = S.sha256_file(p)
            print('%s %d %s' % (common.rel(p), os.stat(p).st_size, h if full else h[:16]))
            n += 1
    print('files: %d' % n)
if __name__ == '__main__':
    main()
