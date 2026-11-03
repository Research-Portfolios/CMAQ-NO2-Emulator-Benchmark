"""Counts the preprocessing's text files whose text the masking function would change (a home folder, a sign-in name, an at sign); prints counts only."""
import glob
import os
import sys
sys.path.insert(0, '.\\run\\code')
from s3 import s3common as S

def main():
    pats = []
    for top in (S.WORK_S3, S.TMP_S3, S.ARRAYS):
        for ext in ('json', 'txt'):
            pats += glob.glob(os.path.join(top, '**', '*.' + ext), recursive=True)
    hits = 0
    for p in sorted(set(pats)):
        with open(p, 'rb') as f:
            t = f.read().decode('utf-8', 'replace')
        if S.common.mask(t) != t:
            hits += 1
            print('HIT in %s' % S.common.rel(p))
    print('text files scanned %d; files with a match %d' % (len(set(pats)), hits))
if __name__ == '__main__':
    S.run_main(main)
