"""Tests each channel's storage type before the arrays are written: on the first day of each month of 2016 it takes every channel through float16 (pressure through its affine map) and moves to float32 any channel with a non-finite value or a 99th-percentile relative error above 0.001."""
import datetime
import os
import sys
import time
sys.path.insert(0, '.\\run\\code')
import numpy as np
from choi import f16
from s3 import s3_assemble as A
from s3 import s3common as S
LAG_SERIES = ('M2', 'M3', 'M4', 'M5', 'M8')
MET = ('M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'X6')

def main():
    S.common.set_temp()
    t0 = time.time()
    start = S.now_local()
    inputs = S.inventory_inputs()
    reader = A.MonthReader(inputs)
    static = A.load_static(inputs)
    veg, veg_used = A.load_veg(inputs)
    clim = np.load(os.path.join(S.ARRAYS, 'emis_clim_2017_2019.npy'), mmap_mode='r')
    vals = {c: [] for c in S.CH_NAMES}
    reads = []
    prev_met = {}
    for ser in LAG_SERIES:
        v, dates = reader.read_month(ser, 2015, 12)
        if dates[-1] != 2015365:
            raise ValueError('2015-12 %s ends %s' % (ser, dates[-1]))
        ch = [c for c, s in S.MET_SERIES.items() if s == ser][0]
        prev_met[ch] = v[-1].copy()
        reads.append('%s 201512' % ser)
        del v
    days = []
    for m in range(1, 13):
        md = {}
        for ser in MET:
            v, dates = reader.read_month(ser, 2016, m)
            md[ser] = (v, dates)
            reads.append('%s 2016%02d' % (ser, m))
        d = datetime.date(2016, m, 1)
        for ser in MET:
            if md[ser][1][0] != S.yyyyddd(d):
                raise ValueError('%s 2016-%02d starts %s' % (ser, m, md[ser][1][0]))
        met = A.met_of_day(md, 0)
        src = A.day_sources(d, met, prev_met, static, veg, clim)
        for c in S.CH_NAMES:
            vals[c].append(np.asarray(src[c], dtype=np.float64).copy())
        days.append(S.ymd(d))
        last = A.met_of_day(md, -1)
        prev_met = {c: np.asarray(last[c]).copy() for c in ('M-2', 'M-3', 'M-4', 'M-5', 'M-8')}
        del md, met, src, last
        print('%s month %d read, %.0f s, memory %.2f GB' % (S.now_local(), m, time.time() - t0, S.mem_gb()), flush=True)
    res = {'start': start, 'days': days, 'reads': len(reads), 'veg_records_used': {str(k): v for k, v in veg_used.items()}, 'threshold': f16.REL_ERR_THRESHOLD, 'channels': []}
    for i, (name, label, source, unit, default) in enumerate(S.CHANNELS):
        x = np.concatenate([a.ravel() for a in vals[name]])
        rec = {'index_c2': i, 'name': name, 'label': label, 'unit': unit, 'default_storage': default}
        rt = f16.roundtrip_error(x, A.store_fn(default), A.load_fn(default))
        back = A.load_fn(default)(A.store_fn(default)(x))
        rec['test_storage'] = default
        rec['roundtrip'] = rt
        rec['n_finite_stored'] = int(np.isfinite(back).sum())
        rec['max_abs_error'] = float(np.nanmax(np.abs(back - x)))
        if default == 'float32':
            rec['storage'] = 'float32'
            rec['why'] = 'the storage rule: E-1 to E-4 and M-9 float32'
            info = f16.roundtrip_error(x, f16.f16_store, f16.f16_load)
            rec['float16_plain_cast_for_the_record'] = info
        elif rt['verdict'] == 'float32':
            rec['storage'] = 'float32'
            if rt['n_nonfinite_roundtrip'] > 0:
                why = 'non-finite after the float16 round trip'
            elif rt['p99_rel_error'] is None:
                why = 'no value above its 1st percentile of positive values, so no error is defined (the safe side, choi.f16 and test_f16)'
            else:
                why = '99th percentile of relative error above 1e-3'
            rec['why'] = 'moved by the storage rule: ' + why
        else:
            rec['storage'] = default
            rec['why'] = 'round trip within the storage rule'
        rec['n_nonfinite_source'] = int((~np.isfinite(x)).sum())
        res['channels'].append(rec)
        del x, back
    res['storage'] = {r['name']: r['storage'] for r in res['channels']}
    res['moved_to_float32'] = [r['name'] for r in res['channels'] if r['default_storage'] != 'float32' and r['storage'] == 'float32']
    res['end'] = S.now_local()
    res['seconds'] = round(time.time() - t0, 1)
    S.write_json(os.path.join(S.WORK_S3, 'roundtrip.json'), res)
    for r in res['channels']:
        rt = r['roundtrip']
        print('%-4s %-15s storage %-14s p99 %s max %s finite %d/%d %s' % (r['name'], r['default_storage'], r['storage'], '%.3g' % rt['p99_rel_error'] if rt['p99_rel_error'] is not None else 'none', '%.3g' % rt['max_rel_error'] if rt['max_rel_error'] is not None else 'none', r['n_finite_stored'], rt['n'], r['why'][:40]))
    print('moved:', res['moved_to_float32'])
if __name__ == '__main__':
    main()
