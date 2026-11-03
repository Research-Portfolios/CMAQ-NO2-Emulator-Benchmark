"""Copies the cell-sum reducer from the Drive extraction code into its own module byte for byte, with a header, its imports and the two constants it uses, and prints the hashes of the source and the copy."""
import hashlib
import os
SRC = '.\\work/c7a_reduce.py'
SRC_COMMON = '.\\work/c7a_common.py'
VEND = '.\\work/stacks.py'
OWN = '.\\run\\code\\choi\\stacks.py'
DST = '.\\run\\code\\s3p2\\p2_cellsums.py'
WANT = '6bf9341cf37eec52'
LINES = (202, 216)

def sha(b):
    return hashlib.sha256(b).hexdigest()

def main():
    with open(SRC, 'rb') as f:
        b = f.read()
    if sha(b)[:16] != WANT:
        raise SystemExit('c7a_reduce.py sha16 %s, not %s' % (sha(b)[:16], WANT))
    lines = b.split(b'\n')
    part = b'\n'.join(lines[LINES[0] - 1:LINES[1]]) + b'\n'
    if not part.startswith(b'def cell_sums(res, g, base, p1):'):
        raise SystemExit('the vendored source holds no cell_sums at the expected place')
    with open(SRC_COMMON, 'rb') as f:
        cm = f.read().split(b'\n')
    if cm[48] != b'GRID = (299, 459)' or cm[58] != b'PT_SPECIES = ("NO", "NO2", "HONO")':
        raise SystemExit('c7a_common.py differ')
    with open(VEND, 'rb') as f:
        v = sha(f.read())
    with open(OWN, 'rb') as f:
        o = sha(f.read())
    if v != o:
        raise SystemExit("the vendored stacks.py differs from the run's")
    head = ('r"""The ONE DEFINITION\'s reducer cell_sums, copied by code by\nrun\\code\\s3p2\\p2_copy_cellsums.py from work/\nc7a_reduce.py (sha256 first 16 %s) lines %d-%d, byte for byte\n(their sha256 %s);\nchoi.stacks is the run\'s own, byte-equal to the reducer\'s vendored\ncopy (sha256 first 16 %s); C holds the two names the lines use,\nas work/c7a_common.py give them.\n"""\nimport sys\n\nsys.path.insert(0, r".\\run\\code")\n\nimport numpy as np # noqa: E402,F401\n\nfrom choi import stacks # noqa: E402\n\n\nclass C:\n    GRID = (299, 459)\n    PT_SPECIES = ("NO", "NO2", "HONO")\n\n\n' % (WANT, LINES[0], LINES[1], sha(part)[:16], v[:16])).encode('ascii')
    out = head + part
    if os.path.exists(DST):
        with open(DST, 'rb') as f:
            if f.read() == out:
                print('p2_cellsums.py exists and is equal; not written')
                return
        raise SystemExit('p2_cellsums.py exists and differs: stop')
    with open(DST + '.partial', 'wb') as f:
        f.write(out)
    os.replace(DST + '.partial', DST)
    print("source sha16 %s; lines %d-%d sha16 %s; vendored stacks = run's stacks sha16 %s; written %s sha16 %s, %d bytes" % (WANT, LINES[0], LINES[1], sha(part)[:16], v[:16], os.path.basename(DST), sha(out)[:16], len(out)))
if __name__ == '__main__':
    main()
