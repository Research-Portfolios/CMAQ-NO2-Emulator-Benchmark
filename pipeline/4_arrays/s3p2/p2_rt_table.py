"""Writes record lines of the round-trip results: each channel's default and final storage type and its yearly errors and counts."""
import os
import sys
sys.path.insert(0, '.\\run\\code')
from s3p2 import p2common as P

def g(x, f='%.3g'):
    return 'none' if x is None else f % x

def main():
    out = sys.argv[1]
    rt = P.jload(os.path.join(P.WORK_P2, 'roundtrip_p2.json'))
    L = ['  chan year test            p99 rel   max rel  n made0  n subn  nonfin', "  (p99 rel, max rel: over the values above the year's 1st percentile of", '  positive values; made0, subn: finite values the cast made 0 or', '  subnormal; nonfin: non-finite after the round trip)']
    for c in P.CH_NAMES:
        for y in P.YEARS:
            r = rt['years'][str(y)].get(c)
            if r is None:
                L.append('  %-4s %4d not built (stage (8))' % (c, y))
                continue
            L.append('  %-4s %4d %-14s %9s %9s %8d %7d %7d' % (c, y, r['test'], g(r['p99_rel_error']), g(r['max_rel_error']), r['n_made_zero'], r['n_made_subnormal'], r['n_nonfinite_roundtrip']))
        L.append('  %-4s storage %s (%s)' % (c, rt['storage'][c], P.DEFAULT_STORAGE[c]))
        for line in P.S.wrap('       why: ', rt['why'][c], '       '):
            L.append(line)
    L.append('  FOR THE RECORD, the plain float16 cast of the float32 channels:')
    for c in P.CH_NAMES:
        if P.DEFAULT_STORAGE[c] != 'float32':
            continue
        for y in P.YEARS:
            r = rt['years'][str(y)].get(c)
            if r is None:
                continue
            q = r['float16_plain_cast_for_the_record']
            L.append('  %-4s %4d f16 p99 %9s max %9s made0 %8d subn %8d nf %d' % (c, y, g(q['p99_rel_error']), g(q['max_rel_error']), q['n_made_zero'], q['n_made_subnormal'], q['n_nonfinite_roundtrip']))
    bad = P.S.check_record_text('\n'.join(L))
    if bad:
        raise ValueError('; '.join(bad[:5]))
    with open(out, 'w', encoding='ascii', newline='\n') as f:
        f.write('\n'.join(L) + '\n')
    print('lines %d; moved %s' % (len(L), rt['moved_to_float32']))
if __name__ == '__main__':
    P.run_main(main)
