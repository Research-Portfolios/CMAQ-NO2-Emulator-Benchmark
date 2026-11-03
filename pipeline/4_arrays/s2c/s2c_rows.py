"""Summarizes the provenance record of every source folder (read only): line counts, distinct paths, paths with several rows, field sets, note words and a masked sample row, with a check of each text twin."""
import collections
import json
import os
ROOT = '.'
W = os.path.join(ROOT, 'work')
FOLDERS = ('raw\\rsig', 'raw\\mcip', 'raw\\s3meta', 'raw\\emis\\stackgroups', 'extract\\emis_merged', 'extract\\pt_marine', 'work\\rsig', 'natural_earth')

def mask(s):
    s = str(s).replace('@', '<at>')
    return s.replace('%40', '<pct40>')

def main():
    for f in FOLDERS:
        d = os.path.join(W, f)
        jl = os.path.join(d, 'PROVENANCE.jsonl')
        print('== %s' % f)
        if not os.path.exists(jl):
            print('  NO PROVENANCE.jsonl')
            continue
        with open(jl, 'rb') as fh:
            raw = fh.read()
        lines = [x for x in raw.decode('ascii').split('\n') if x.strip()]
        rows = collections.OrderedDict()
        multi = collections.Counter()
        fsets = collections.Counter()
        notes = collections.Counter()
        for ln in lines:
            r = json.loads(ln)
            multi[r['path']] += 1
            rows[r['path']] = r
        for r in rows.values():
            fsets[tuple(sorted(r))] += 1
            notes[' '.join(str(r.get('note', '')).split()[:2])] += 1
        at = raw.count(b'@')
        print('  lines %d; distinct paths %d; paths with >1 row %d; at signs %d' % (len(lines), len(rows), sum((1 for v in multi.values() if v > 1)), at))
        for k, v in multi.items():
            if v > 1:
                print('    multi: %s x%d' % (mask(k), v))
        for fs, n in fsets.most_common():
            print('  fields (%d rows): %s' % (n, ', '.join(fs)))
        for k, n in notes.most_common(8):
            print('  note starts %r: %d' % (mask(k), n))
        r0 = next(iter(rows.values()))
        print('  sample:')
        for k in sorted(r0):
            print('    %s: %s' % (k, mask(r0[k])[:100]))
        txt = os.path.join(d, 'PROVENANCE_ASCII.txt')
        if os.path.exists(txt):
            with open(txt, 'rb') as fh:
                t = fh.read()
            tl = t.split(b'\n')
            print('  PROVENANCE_ASCII.txt: %d lines, longest %d, ascii %s, CR %d' % (len(tl), max((len(x) for x in tl)), all((b < 128 for b in t)), t.count(b'\r')))
        else:
            print('  NO PROVENANCE_ASCII.txt')
if __name__ == '__main__':
    main()
