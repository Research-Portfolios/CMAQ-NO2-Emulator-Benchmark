"""Writes the input-check record of one filled year from the JSON results of the fill."""
import os
import sys
sys.path.insert(0, '/content/emul/run/code')
sys.path.insert(0, '/content/emul/run/code/s8')
import s8common as C
from s3 import s3common as S

def wr(lines, first, text, cont='  '):
    lines.extend(S.common.wrap(first, text, 78, cont))

def fmt(v):
    if v is None:
        return 'none'
    if isinstance(v, float):
        return '%.6g' % v
    return str(v)

def main():
    y = sys.argv[1]
    p = C.paths(y)
    d4 = C.jload(p['work']('d4_s8_%s.json'))
    ver = C.jload(p['work']('verify_s8_%s.json'))
    pre = C.jload(p['work']('pre_s8_%s.json'))
    fill = C.jload(p['work']('fill_s8_%s.json'))
    W = 'work'
    L = []
    L.append('S8 INPUT CHECKS OF %s (the fills stage;' % y)
    wr(L, '', "input checks). Root /content/emul (the Colab VM of the fills stage, the local machine's layout below it); W is %s. Made by run\\code\\s8\\s8_d4_record.py from W\\arrays_p2\\work\\d4_s8_%s.json (the inputs, read by s8_run.py d4 at %s, before the first write of the array, which began after pre_s8_%s.json of %s) and verify_s8_%s.json (the five channels' values, %s)." % (W, y, d4['made'][:19], y, pre['made'][:19], y, ver['made'][:19]))
    wr(L, '', "Inputs: the extracts of W\\extract_c7a\\%s\\ that MANIFEST_%s.csv lists (%d rows; sha16 %s), each read whole. Masked cells (W\\arrays_p2\\mask_conus_land.npy, sha16 %s, equal to S3_ARRAYS.json's: %s): %d." % (y, y, d4['manifest']['rows'], d4['manifest']['sha256'][:16], d4['mask']['sha256'][:16], d4['mask']['equal_to_S3_ARRAYS'], d4['mask']['cells']))
    L.append('')
    L.append('1. HOW EACH FILE WAS READ')
    pc = d4['preconditions']
    wr(L, '- ', 'Preconditions: %s complete %s (completed_at %s), files %d, status %s; %s begins "C-7a (4) %s: PASS": %s.' % (pc['copylog'], pc['complete'], pc['completed_at'], pc['files'], pc['status'], pc['line_file'], y, pc['line_begins_pass']))
    wr(L, '- ', 'Each extract read whole into memory; its bytes and sha256 equal to its entry in the copy log (status ok), its MANIFEST row and its SHA256SUMS_%s.txt line (%d lines, sha16 %s) before its first use, else the read stops; loaded from memory by numpy (no pickle). Files verified %d; reads %d (each later pass reads them again, each time checked the same way).' % (y, d4['sums']['lines'], d4['sums']['sha256'][:16], d4['files_verified'], d4['reads']))
    wr(L, '- ', 'Fill values: a value at or below -9e36 counts as -9.999e+36; NaN and infinities counted apart as non-finite.')
    L.append('')
    L.append('2. UNITS AGAINST THE CHANNEL TABLE (the channels and their inputs)')
    g = d4['groups']
    wr(L, '  ', "E-1 NO, E-2 NO2, E-3 HONO: the table's mol/s; merged extracts state %s." % ', '.join(('%s (%d files)' % (k, v) for k, v in g['merged']['units'].items())))
    wr(L, '  ', 'E-4 VOC_INV: the table\'s "as its file states, likely g/s"; the same merged files: the fourth unit, g/s.')
    for s in C.DAILY + C.REP:
        wr(L, '  ', 'P-1 input %s (NO NO2 HONO, and P1_NOX formed from them in mol/s): %s.' % (s, ', '.join(('%s (%d files)' % (k, v) for k, v in g[s]['units'].items()))))
    wr(L, '  ', 'Accepted as mol/s: "moles/s" or "mol/s" (c7a_common.py). No conversion: stage (1) applied none to 2017 and 2019 (their extracts state the same units).')
    L.append('')
    L.append('3. COVERAGE, FORM AND DUPLICATES (per kind and sector)')
    for name, rec in g.items():
        hrs = ', '.join(('%s %d' % (k, v) for k, v in rec['hours'].items()))
        sp = ', '.join(('%s %d' % (k, v) for k, v in rec['species'].items()))
        t = '%s %s: %d files, %d dates, duplicates %d, other years %d; species %s; hours %s; key sets %d; file problems %d' % (name, y, rec['files'], rec['dates'], len(rec['duplicates']), len(rec['not_this_year']), sp, hrs, len(rec['keys']), len(rec['problems']))
        if 'missing_days' in rec:
            t += '; days missing %d' % len(rec['missing_days'])
        if name != 'merged':
            t += '; stacks %s; summed %s; stacks outside the grid %d (in %d files)' % (sorted(rec['stacks']), sorted(rec['summed']), rec['outside_total'], rec['outside_files'])
        if 'members_a_month' in rec:
            mm = rec['members_a_month']
            t += 'members a month %s (the holiday rule: %s); members on a federal holiday %s' % ('/'.join((str(mm[k]['n']) for k in sorted(mm))), 'all months ok' if all((mm[k]['k6_ok'] for k in mm)) else 'a month fails', ', '.join(rec['members_on_holidays']) or 'none')
        wr(L, '  ', t + '.')
    wr(L, '  ', 'Manifest rows outside merged and the six sectors: %d.' % d4['rows_outside_the_six_sectors_and_merged'])
    L.append('')
    L.append('4. FILL VALUES, NON-FINITE, NEGATIVE AND ALL-ZERO (inputs)')
    wr(L, '  ', 'Per kind, sector and array, over all cells of every file (masked cells in brackets); min and max over the finite values.')
    for name, rec in g.items():
        for k, c in rec['counts'].items():
            wr(L, '  ', '%s %s: non-finite %d [%d], fill %d [%d], negative %d [%d], zero cells %d, all-zero files %d; min %s, max %s.' % (name, k, c.get('nonfinite', 0), c.get('nonfinite_mask', 0), c.get('fill', 0), c.get('fill_mask', 0), c.get('negative', 0), c.get('negative_mask', 0), c.get('zero', 0), c.get('all_zero_files', 0), fmt(c.get('min_v')), fmt(c.get('max_v'))))
    L.append('')
    L.append('5. THE FIVE CHANNELS AS FORMED AND STORED (verify_s8_%s.json)' % y)
    wr(L, '  ', "Each day's float64 fields formed from the extracts (E-1 to E-4 the merged extract of the day; P-1 the six sectors' P1_NOX by the ONE DEFINITION and the weekday classes) and the stored float32 of %s's inputs_f32 checked against them bitwise: equal on %d of %d days." % (y, ver['stored_days_equal_to_fields'], ver['stored_days_equal_to_fields'] + len(ver['stored_days_differing'])))
    for c, v in ver['d4_counts'].items():
        ex = ver['extremes'][c]
        wr(L, '  ', '%s: cell-days %d [masked %d]; non-finite %d [%d]; fill %d [%d]; negative %d [%d]; zero %d; stored non-finite %d, negative %d; masked cell-days missing %d (fraction %s); stored min %s, max %s.' % (c, v['n'], v['n_mask'], v['nonfinite'], v['nonfinite_mask'], v['fill'], v['fill_mask'], v['negative'], v['negative_mask'], v['zero'], v['stored_nonfinite'], v['stored_negative'], v['masked_missing'], fmt(v['masked_missing_fraction']), fmt(ex['min']), fmt(ex['max'])))
    wr(L, '  ', "the study design's ranges for emissions: not below 0 (counted above, kept, never altered).")
    L.append('')
    L.append('6. WEEKDAY CLASSES, HOLIDAYS AND FALLBACKS')
    hol = d4['holidays']
    wr(L, '  ', "Federal holidays of %s (the rule of the plan's 2017 and 2019 lists, found equal to them: %s): %s." % (y, d4['plan_holidays_equal_s2a'], '; '.join(('%s %s' % (d, ', '.join(v)) for d, v in sorted(hol.items())))))
    wr(L, '  ', 'Members dated on a holiday, left out: %d: %s.' % (len(d4.get('holiday_members_left_out', [])), ', '.join(d4.get('holiday_members_left_out', [])) or 'none'))
    wr(L, '  ', 'Classes with no member, taking the Tuesday-to-Friday class: %d: %s.' % (len(d4.get('fallbacks', [])), ', '.join(d4.get('fallbacks', [])) or 'none'))
    wr(L, '  ', 'The fill used the same lists (fill_s8_%s.json keys holiday_members_left_out and fallbacks: %s).' % (y, 'equal' if fill.get('holiday_members_left_out') == d4.get('holiday_members_left_out') and fill.get('fallbacks') == d4.get('fallbacks') else 'DIFFERENT'))
    L.append('')
    L.append('7. WHAT IS EXCLUDED, AND WHY')
    wr(L, '- ', 'The fire sectors and rwc: not read; not in the extracts.')
    wr(L, '- ', 'E-1 to E-4 take the merged files only; the six point and marine sectors enter as P-1.')
    wr(L, '- ', "The representative-day sectors' members dated on a federal holiday (legal or observed): left out of the class means, listed in section 6.")
    wr(L, '- ', 'Stacks outside the grid: left out at the extraction (counts in section 3).')
    wr(L, '- ', 'No value of any input is excluded or altered for its range or for a fill value: the counts above are kept as read.')
    L.append('')
    L.append('8. STOP TEST (more than 0.1 percent of masked cell-days missing)')
    stop = list(d4.get('problems', [])) + list(ver.get('stop', []))
    wr(L, '  ', "%s: the inputs' problems %d (d4 verdict %s); the channels' stops %d (verify verdict %s)%s." % ('PASS' if not stop else 'STOP', len(d4.get('problems', [])), d4['verdict'], len(ver.get('stop', [])), ver['verdict'], ': ' + '; '.join(stop) if stop else ''))
    text = '\n'.join(L) + '\n'
    bad = S.check_record_text(text.rstrip('\n'))
    if bad:
        raise SystemExit('record lines: ' + '; '.join(bad[:5]))
    out = p['d4record']
    part = out + '.partial'
    with open(part, 'w', encoding='ascii', newline='\n') as f:
        f.write(text)
    os.replace(part, out)
    info = S.file_info(out)
    print('written %s (%d bytes, sha16 %s, %d lines)' % (info['path'], info['bytes'], info['sha256'][:16], len(L)))
if __name__ == '__main__':
    S.run_main(main)
