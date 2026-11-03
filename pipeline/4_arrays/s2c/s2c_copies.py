"""Compares each source folder's provenance records with their copies in the record folder by size and sha256 (read only)."""
import hashlib
import os
ROOT = '.'
W = os.path.join(ROOT, 'work')
PROV = os.path.join(ROOT, 'run\\provenance')
PAIRS = (('rsig', 'raw\\rsig'), ('rsig_work', 'work\\rsig'), ('s3_mcip', 'raw\\mcip'), ('s3_s3meta', 'raw\\s3meta'), ('s3_stackgroups', 'raw\\emis\\stackgroups'), ('s3_emis_merged', 'extract\\emis_merged'), ('s3_pt_marine', 'extract\\pt_marine'), ('ne_natural_earth', 'natural_earth'))

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def main():
    for name, folder in PAIRS:
        for fn in ('PROVENANCE_ASCII.txt', 'PROVENANCE.jsonl'):
            src = os.path.join(W, folder, fn)
            dst = os.path.join(PROV, '%s_%s' % (name, fn))
            a = (os.stat(src).st_size, sha(src)) if os.path.exists(src) else (None, None)
            b = (os.stat(dst).st_size, sha(dst)) if os.path.exists(dst) else (None, None)
            print('%s_%s: source %s %s; copy %s %s; %s' % (name, fn, a[0], (a[1] or '')[:16], b[0], (b[1] or '')[:16], 'EQUAL' if a == b and a[0] is not None else 'DIFFER'))
if __name__ == '__main__':
    main()
