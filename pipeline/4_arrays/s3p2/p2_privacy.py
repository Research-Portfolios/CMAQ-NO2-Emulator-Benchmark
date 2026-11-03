"""Counts the second pass's text files whose text the masking function or the privacy patterns would change; prints counts only."""
import glob
import os
import re
import sys
sys.path.insert(0, '.\\run\\code')
from choi_dl import common
from s3p2 import p2common as P
HOME = os.path.expanduser('~')
PATTERNS = {'email': re.compile('[A-Za-z0-9._%+-]+\\x40[A-Za-z0-9.-]+\\.[A-Za-z]{2,}'), 'login-hash': re.compile('(email|org)_sha256_12\\s*[=:]\\s*[0-9a-f]{12}'), 'users-path': re.compile('\\\\Users\\\\[A-Za-z]', re.IGNORECASE), 'home': re.compile(re.escape(HOME), re.IGNORECASE)}

def main():
    tops = [os.path.join(P.RUN, 'code', 's3p2'), P.CFG_P2, P.TMP_P2, P.ARRAYS_P2, P.NE_ZIPS]
    pats = []
    for top in tops:
        for ext in ('py', 'json', 'jsonl', 'txt', 'csv', 'cpg'):
            pats += glob.glob(os.path.join(top, '**', '*.' + ext), recursive=True)
    pats += [P.REPORT, P.D4_RECORD]
    hits = 0
    n = 0
    for p in sorted(set(pats)):
        if not os.path.isfile(p):
            continue
        n += 1
        with open(p, 'rb') as f:
            t = f.read().decode('utf-8', 'replace')
        cats = [c for c, rx in PATTERNS.items() if rx.search(t)]
        if common.mask(t) != t:
            cats.append('mask')
        if cats:
            hits += 1
            print('HIT in %s: %s' % (common.rel(p), ', '.join(cats)))
    print('text files scanned %d; files with a match %d' % (n, hits))
if __name__ == '__main__':
    P.run_main(main)
