"""Prints the provenance rows that the refetch added for the named dates, masked and shortened, with the record's line count and ASCII test (read only)."""
import json
import os
import sys
W = '.\\work'
D = os.path.join(W, 'extract', 'pt_marine')

def mask(s):
    return str(s).replace('@', '<at>').replace('%40', '<pct40>')

def main(argv):
    with open(os.path.join(D, 'PROVENANCE.jsonl'), 'rb') as f:
        lines = [x for x in f.read().decode('ascii').split('\n') if x.strip()]
    print('jsonl rows %d' % len(lines))
    for ln in lines:
        r = json.loads(ln)
        if any((r['path'].endswith('pt_othpt_%s.npz' % d) for d in argv)):
            print('ROW')
            for k in sorted(r):
                print('  %s: %s' % (k, mask(r[k])[:110]))
    with open(os.path.join(D, 'PROVENANCE_ASCII.txt'), 'rb') as f:
        t = f.read()
    tl = t.split(b'\n')
    print('PROVENANCE_ASCII.txt: %d lines, longest %d, ascii %s, CR %d' % (len(tl), max((len(x) for x in tl)), all((b < 128 for b in t)), t.count(b'\r')))
    return 0
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
