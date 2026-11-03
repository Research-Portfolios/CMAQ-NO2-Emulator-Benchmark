"""Writes the log sections of the round trip, the build, the splits, the statistics and the second reading from the stage's outputs, so that each number is copied by code."""
import json
import os
import sys
sys.path.insert(0, '.\\run\\code')
from s3 import s3common as S
STAGE = os.path.join(S.WORK_S3, 'stage')

def jload(p):
    with open(p, 'r', encoding='ascii') as f:
        return json.load(f)

def e3(x):
    return 'none' if x is None else '%.3g' % x

def part_roundtrip():
    r = jload(os.path.join(S.WORK_S3, 'roundtrip.json'))
    L = ['', '7. STEP (5) THE ROUND TRIP: DONE', '(run\\code\\s3\\s3_roundtrip.py, %s to %s; log' % (r['start'][11:], r['end'][11:]), '  W\\work/roundtrip_stdout.txt; result W\\work\\s3\\roundtrip.json)']
    L += S.wrap('- ', 'Days: the first of each month of 2016 (%s to %s, %d days); %d RSIG answers read (no T-1); every channel on all 137241 cells, so %d values a channel; X-1 to X-5 from the day before (2015-12-31 for 2016-01-01, from the 2015-12 answers). The test is choi.f16.roundtrip_error (threshold %g), unchanged.' % (r['days'][0], r['days'][-1], len(r['days']), r['reads'], r['channels'][0]['roundtrip']['n'], r['threshold']))
    L.append('  chan test           p99 rel   max rel   finite     storage')
    for c in r['channels']:
        rt = c['roundtrip']
        L.append('  %-4s %-14s %9s %9s %8d   %s' % (c['name'], c['test_storage'], e3(rt['p99_rel_error']), e3(rt['max_rel_error']), c['n_finite_stored'], c['storage']))
    L += S.wrap('  ', "(p99 rel and max rel: the 99th percentile and the largest relative error over the values above the channel's 1st percentile of positive values; finite: finite values after the round trip, of %d; the float32 channels' round trip through float32 is exact.)" % r['channels'][0]['roundtrip']['n'])
    mv = r['moved_to_float32']
    for c in r['channels']:
        if c['name'] in mv:
            L += S.wrap('- ', 'MOVED to float32: %s (%s): %s. Its positive values number %d, its 1st percentile of them is %s and %d values lie above it.' % (c['name'], c['label'], c['why'], c['roundtrip']['n_positive'], e3(c['roundtrip']['p1_positive']), c['roundtrip']['n_used']))
    L += S.wrap('- ', 'For the record, the plain float16 cast of the five float32 channels (the storage rule keeps them float32 whatever this gives):')
    for c in r['channels']:
        if 'float16_plain_cast_for_the_record' in c:
            f = c['float16_plain_cast_for_the_record']
            L += S.wrap('    ', '%s: p99 %s, max %s, non-finite %d, verdict %s' % (c['name'], e3(f['p99_rel_error']), e3(f['max_rel_error']), f['n_nonfinite_roundtrip'], f['verdict']))
    L += S.wrap('- ', 'Storage fixed for step (6): float16 %d channels, float32 %d channels.' % (sum((1 for c in r['channels'] if c['storage'] != 'float32')), sum((1 for c in r['channels'] if c['storage'] == 'float32'))))
    return L

def part_build():
    ch = jload(os.path.join(S.CFG, 'S3_CHANNELS.json'))
    ar = jload(os.path.join(S.CFG, 'S3_ARRAYS.json'))
    b = {y: jload(os.path.join(S.WORK_S3, 'build_%d.json' % y)) for y in S.YEARS}
    L = ['', '8. STEP (6) THE ARRAYS: DONE', '(run\\code\\s3\\s3_build.py, second run %s to %s; log' % (b[2015]['start'][11:], b[2019]['end'][11:]), '  W\\work/build_stdout.txt; per year W\\work\\s3\\build_YYYY.json; the index', '  files by s3_index.py)']
    import re
    with open(os.path.join(S.W, 'tmp', 's3', 'build_stdout.txt'), 'r', encoding='utf-8-sig') as fh:
        first = fh.readline()
    mt = re.search('disk free ([0-9.]+) GB; need ([0-9.]+) GB, asked ([0-9.]+) GB', first)
    L += S.wrap('- ', "Disk first: before the build, free disk %s GB against a remaining need of %s GB (the five years' arrays, and 1.5 GB for the later stages' predictions, checkpoints and logs), so %s GB asked: met (W\\work/build_stdout.txt)." % mt.groups())
    L += S.wrap('- ', "Channels, the table's order with L-4 dropped (%d): %s. Baseline %d: %s; advanced all %d." % (len(ch['order']), ', '.join(ch['order']), len(ch['baseline_channels']), ', '.join(ch['baseline_channels']), len(ch['advanced_channels'])))
    L += S.wrap('- ', "inputs_f16 holds %d channels (%s); inputs_f32 holds %d (%s). M-1 is stored as (x - 50000) / 60000 in float16 (inverse y * 60000 + 50000 Pa). Each channel's array, index, source, unit, storage, affine map and round-trip error: run\\config\\S3_CHANNELS.json." % (len(ch['f16_channels']), ', '.join(ch['f16_channels']), len(ch['f32_channels']), ', '.join(ch['f32_channels'])))
    L += S.wrap('- ', "Sources (run\\code\\s3\\s3_assemble.py, shared with step (5)): M-1 to M-9 and X-6 the day's step of its month's answer (M-4 and M-5 the four-corner mean in float64); L-1 HT, L-2 LWMASK and L-3 PURB / 100 static; L-5 the month's VEG (mean of the 24 records of the 15th of that month of 2016); E-1 to E-4 the emission climatology of the date's (month, weekday) from W\\arrays\\emis_clim_2017_2019.npy; X-1 to X-5 the day before's M-8, M-3, M-4, M-5 and M-2 (2015-01-01 from the lag answers of 2014-12-31; each later year's first day from the previous year's December answers). Inside each month every X lag was checked equal to the previous day's stored M value (build_YYYY.json lag_blocks_checked: %s)." % ', '.join((str(b[y]['lag_blocks_checked']) for y in S.YEARS)))
    L.append('  year days  file          bytes        sha256')
    for y in S.YEARS:
        yr = ar['years'][str(y)]
        for k in ('inputs_f16', 'inputs_f32', 'target'):
            f = yr['files'][k]
            L.append('  %d %4d  %-12s %11d  %s' % (y, yr['days'], k, f['bytes'], f['sha256'][:16]))
    L += S.wrap('  ', "(sha16 shown; the full sha256, shapes and dtypes are in run\\config\\S3_ARRAYS.json, which also lists each year's dates and the mask and emission arrays; npy bytes in all %d.)" % ar['total_bytes_npy'])
    for y in S.YEARS:
        r = b[y]
        L += S.wrap('- ', "%d: %d days (%s to %s), %d RSIG answers read, %.1f s; free memory %.2f GB at the start, at least %.2f GB at each month's start, %.2f GB at the end; free disk %.2f GB after." % (y, r['days'], r['dates_first_last'][0], r['dates_first_last'][1], r['answers_read'], r['seconds'], r['memory_gb_start'], r['memory_gb_min_at_month_starts'], r['memory_gb_end'], r['disk_free_gb_end']))
    return L

def part_splits():
    sp = jload(os.path.join(S.CFG, 'SPLITS.json'))
    info = S.file_info(os.path.join(S.CFG, 'SPLITS.json'))
    L = ['', '9. STEP (7) THE SPLITS: DONE', '(run\\code\\s3\\s3_splits.py;', 'run\\config\\SPLITS.json, %d bytes, sha16 %s)' % (info['bytes'], info['sha256'][:16])]
    L += S.wrap('- ', "Rule: %s; each list as dates and as indices into its year's arrays (%s)." % (sp['validation_rule'], sp['index_rule']))
    L.append('  split               train  validation  held out  tests')
    for k, v in list(sp['folds'].items()) + list(sp['f6'].items()):
        nm = 'fold ' + k if k in sp['folds'] else k
        L.append('  %-19s %6d %11d %9d  %s' % (nm, v['train']['count'], v['validation']['count'], v['held_out']['count'], 'pass' if v['test']['pass'] else 'FAIL'))
    L += S.wrap('- ', "Tests (each split's key test): no held-out day in any training or validation list; training and validation disjoint; every validation day in an ISO week divisible by 5 and no training day in one; training plus validation cover the training years exactly. All pass: %s. Labels: fold 2015 %s; fold 2019 %s." % (sp['all_tests_pass'], sp['folds']['2015']['label'], sp['folds']['2019']['label']))
    L += S.wrap('- ', 'train 2017 test 2019 and train 2019 test 2017, inside the one training year on its extracted days (%d of 2017 and %d of 2019, every day); the arms TRUE, OTHER and OWN share these days and differ in E-1 to E-4 only (SPLITS.json f6.<name>.arms).' % (sp['f6']['train2017_test2019']['extracted_days_train_year'], sp['f6']['train2019_test2017']['extracted_days_train_year']))
    return L

def part_norm():
    idx = jload(os.path.join(S.WORK_S3, 'norm_index.json'))
    L = ['', '10. STEP (8) THE STATISTICS: DONE', '(run\\code\\s3\\s3_norm.py, %s to %s; index' % (idx['start'][11:], idx['end'][11:]), '  W\\work\\s3\\norm_index.json; written before any training, and none ran)']
    L += S.wrap('- ', "Rules: on the training days of config\\SPLITS.json only, on the stored values (M-1 as stored); each channel's minimum and maximum over all cells from the build's per-day values (W\\work\\s3\\daystats_YYYY.npz); for M-9 and E-1 to E-4, c = the 1st percentile of the positive values (numpy's linear method, from the exact order statistics kept at the build, W\\work\\s3\\smallk_YYYY.npz, K = 2000000 a year; the rank needed is tested under K) and the minimum and maximum of log10(x + c); the target's minimum and maximum over the masked cells. The two-year emission tests' arms take E-1 to E-4 from their own arrays (TRUE emis_true_<y>, OWN emis_clim_<y>, OTHER the other year's emis_clim) on the training year's 295 days.")
    L.append('file (in run\\config\\) bytes')
    for name in idx['written']:
        f = idx['files'][name]
        L.append('  %-40s %8d' % (name + '.json', f['bytes']))
        L.append('    sha256 %s' % f['sha256'])
    L.append('')
    L.append("  Values of the folds (of each NORM file's keys channels.<c> and target):")
    L.append('  fold  M-1 min..max (stored)  T-1 min..max (ppb, masked)')
    for name in idx['written']:
        d = jload(os.path.join(S.CFG, name + '.json'))
        if d['kind'] != 'fold':
            continue
        c = d['channels']['M-1']
        L.append('  %d  %.6g..%.6g          %.6g..%.6g' % (d['held_out_year'], c['min'], c['max'], d['target']['min'], d['target']['max']))
    L.append('  c of the log channels (fold files):')
    L.append('  fold  M-9         E-1         E-2         E-3         E-4')
    for name in idx['written']:
        d = jload(os.path.join(S.CFG, name + '.json'))
        if d['kind'] != 'fold':
            continue
        L.append('  %d  %s' % (d['held_out_year'], ' '.join(('%-11.4g' % d['channels'][c]['c'] for c in S.LOG_CHANNELS))))
    L.append('c of E-1 to E-4 in the two-year emission test files:')
    for name in idx['written']:
        d = jload(os.path.join(S.CFG, name + '.json'))
        if d['kind'] != 'f6':
            continue
        L.append('  %-30s %s' % (name[8:], ' '.join(('%-10.4g' % d['channels'][c]['c'] for c in S.EMIS))))
    return L

def part_second():
    r = jload(os.path.join(S.WORK_S3, 'second_reading.json'))
    s = r['summary']
    L = ['', '11. STEP (9) A SECOND READING: %s' % ('PASS' if s['pass'] else 'FAIL'), '(run\\code\\s3\\s3_second.py, %s to %s; result' % (r['start'][11:], r['end'][11:]), '  W\\work\\s3\\second_reading.json)']
    L += S.wrap('- ', 'The 15th of each month of 2015 to 2019 (%d days, %d cells each): T-1, M-1 and M-4 read again from the raw answers by another path (hashlib on the raw bytes; zlib for the gzip stream; scipy.io.netcdf_file for the netCDF-3 file; the four-corner mean summed in another order) and compared with the stored arrays.' % (s['days'], s['cells_per_day']))
    L += S.wrap('- ', 'T-1 (float32): equal in every cell of every day: %s (largest difference %s ppb).' % (s['T1_exact_all'], e3(s['T1_max_abs'])))
    tot = s['days'] * s['cells_per_day']
    L += S.wrap('- ', 'M-1 (float16 under the map): within half the float16 spacing at the stored value in every cell: %s; equal to the float16 cast of the recomputed mapped value in %d of %d cells; largest difference after the inverse map %s Pa.' % (s['M1_within_all'], s['M1_exact_cast_cells'], tot, e3(s['M1_max_abs_pa'])))
    L += S.wrap('- ', 'M-4 (float16 of the four-corner mean): within half the float16 spacing in every cell: %s; equal to the float16 cast of the recomputed mean in %d of %d cells; largest difference %s m/s.' % (s['M4_within_all'], s['M4_exact_cast_cells'], tot, e3(s['M4_max_abs'])))
    return L
PARTS = {'roundtrip': part_roundtrip, 'build': part_build, 'splits': part_splits, 'norm': part_norm, 'second': part_second}

def main():
    part = sys.argv[1]
    L = [x.rstrip() for x in PARTS[part]()]
    os.makedirs(STAGE, exist_ok=True)
    p = os.path.join(STAGE, 'part_%s.txt' % part)
    text = '\n'.join(L) + '\n'
    bad = S.check_record_text(text.rstrip('\n'))
    if bad:
        raise ValueError('; '.join(bad[:5]))
    with open(p, 'w', encoding='ascii', newline='\n') as f:
        f.write(text)
    print('%s: %d lines -> %s' % (part, len(L), p))
if __name__ == '__main__':
    S.run_main(main)
