"""Checks each named evaluation output against its folder's EVAL_DONE.json entry (sha256 and size) and stops on a missing file or a difference."""
import hashlib
import json
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEALED = os.path.join(ROOT, 'results')
CV = os.path.join(SEALED, 'evaluation_test_years')
LB = os.path.join(SEALED, 'evaluation_held_back')

def _sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def verify(folder, names):
    done = os.path.join(folder, 'EVAL_DONE.json')
    if not os.path.isfile(done):
        raise SystemExit('STOP: EVAL_DONE.json missing in %s' % folder)
    with open(done, encoding='utf-8') as f:
        files = json.load(f)['files']
    out = []
    for n in names:
        p = os.path.join(folder, n)
        tag = os.path.basename(folder) + '\\' + n
        if not os.path.isfile(p):
            raise SystemExit('STOP: sealed file missing: %s' % tag)
        if n not in files:
            raise SystemExit('STOP: %s not listed in EVAL_DONE.json' % tag)
        ent = files[n]
        if os.path.getsize(p) != int(ent['bytes']):
            raise SystemExit('STOP: bytes differ: %s' % tag)
        if _sha(p) != ent['sha256']:
            raise SystemExit('STOP: sha256 differs: %s' % tag)
        print('sealed check PASS:', tag)
        out.append(p)
    return out
if __name__ == '__main__':
    verify(CV, ['family.json', 'band_ratios.json', 'residual_acf.json', 'd4_to_d7.json', 'an02.json', 'exploratory.json'])
    verify(LB, ['family.json', 'an02.json'])
    sys.exit(0)
