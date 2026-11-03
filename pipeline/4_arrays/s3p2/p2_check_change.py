"""Prints the differences of the changed first-pass files against their backups, their sha256 and the channel lists that the changed code gives (read only)."""
import difflib
import hashlib
import os
import sys
sys.path.insert(0, '.\\run\\code')
from s3 import s3_assemble as A
from s3 import s3_index as I
from s3 import s3common as S
from s3p2 import p2common as P
D = '.\\run\\code\\s3'

def sha16(p):
    with open(p, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]

def main():
    for n in ('s3common', 's3_assemble', 's3_index'):
        cur = os.path.join(D, n + '.py')
        bak = os.path.join(D, n + '_Backup_PriorToDluseP2.py')
        with open(bak, 'r', encoding='ascii') as f:
            a = f.read().split('\n')
        with open(cur, 'r', encoding='ascii') as f:
            b = f.read().split('\n')
        print('== %s: backup %s, now %s, lines %d -> %d' % (n, sha16(bak), sha16(cur), len(a), len(b)))
        for line in difflib.unified_diff(a, b, 'backup', 'now', n=0, lineterm=''):
            print('  ' + line.replace('@@', '=='))
    names = [c[0] for c in S.CHANNELS]
    print('s3common.CHANNELS (%d): %s' % (len(names), ' '.join(names)))
    print('PRIMARY (%d): %s' % (len(I.PRIMARY), ' '.join(I.PRIMARY)))
    print('ARM18 (%d): %s' % (len(I.ARM18), ' '.join(I.ARM18)))
    adv = [c for c in P.CH_NAMES if c in I.PRIMARY or c in I.LAGS]
    print('advanced (%d): %s' % (len(adv), ' '.join(adv)))
    print('p2common.CHANNELS (%d): %s' % (len(P.CH_NAMES), ' '.join(P.CH_NAMES)))
    missing = [c for c in set(I.PRIMARY) | set(I.ARM18) | set(I.LAGS) if c not in P.CH_NAMES]
    print('named but not in the table: %s' % (missing or 'none'))
    arm_as_built = [c for c in I.ARM18 if c in P.CH_NAMES]
    print('ARM18 as built (L-4 not built): %d' % len(arm_as_built))
if __name__ == '__main__':
    main()
