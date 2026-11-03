"""Reads the NO2 target, pressure and a wind component again from the raw answers for the 15th of each month of 2015 to 2019 by a separate path (hashlib, zlib and SciPy's netCDF reader) and compares them with the stored arrays: the target exactly, the others within float16 rounding."""
import datetime
import hashlib
import io
import json
import os
import sys
import zlib
sys.path.insert(0, '/content/emul/run/code')
import numpy as np
from scipy.io import netcdf_file
from s3 import s3common as S

def read2(entry, var):
    p = os.path.join(S.ROOT, entry['path'])
    with open(p, 'rb') as f:
        raw = f.read()
    if hashlib.sha256(raw).hexdigest() != entry['sha256']:
        raise ValueError('raw sha256 differs: ' + entry['path'])
    data = zlib.decompress(raw, 31)
    if hashlib.sha256(data).hexdigest() != entry['decompressed_sha256']:
        raise ValueError('decompressed sha256 differs: ' + entry['path'])
    nc = netcdf_file(io.BytesIO(data), 'r', mmap=False)
    try:
        v = np.array(nc.variables[var].data, dtype=np.float32)
        tf = np.array(nc.variables['TFLAG'].data)
        units = nc.variables[var].units
        units = units.decode() if isinstance(units, bytes) else str(units)
    finally:
        nc.close()
    return (v, tf, units.strip())

def half_ulp16(stored):
    s = np.abs(stored.astype(np.float16))
    return 0.5 * np.spacing(s).astype(np.float64)

def main():
    S.common.set_temp()
    t0 = S.now_local()
    inv = S.load_inventory()
    ent = {}
    for e in inv['files']:
        if e.get('kind') == 'rsig' and e.get('s3_input') is True:
            ent[e['series'], e['period']] = e
    with open(os.path.join(S.CFG, 'S3_CHANNELS.json'), 'r', encoding='ascii') as fh:
        ch = json.load(fh)
    cmap = {c['name']: c for c in ch['channels']}
    j1 = cmap['M-1']['index']
    j4 = cmap['M-4']['index']
    if cmap['M-1']['array'] != 'inputs_f16' or cmap['M-4']['array'] != 'inputs_f16':
        raise ValueError('M-1 or M-4 not in the float16 array')
    out = {'start': t0, 'days': [], 'reader': 'scipy.io.netcdf_file, zlib'}
    ok = True
    for y in S.YEARS:
        f16 = np.load(os.path.join(S.ARRAYS, 'inputs_f16_%d.npy' % y), mmap_mode='r')
        tgt = np.load(os.path.join(S.ARRAYS, 'target_%d.npy' % y), mmap_mode='r')
        for m in range(1, 13):
            d = datetime.date(y, m, 15)
            i = (d - datetime.date(y, 1, 1)).days
            k = 14
            per = '%04d%02d' % (y, m)
            want = d.year * 1000 + d.timetuple().tm_yday
            rec = {'date': d.strftime('%Y%m%d'), 'index': i}
            v, tf, u = read2(ent['T1', per], 'NO2')
            if int(tf[k, 0, 0]) != want:
                raise ValueError('T1 TFLAG %s' % tf[k, 0, 0])
            a = v[k, 0]
            b = np.asarray(tgt[i])
            rec['T1_cells'] = int(a.size)
            rec['T1_exact'] = int((a == b).sum())
            rec['T1_max_abs'] = float(np.abs(a.astype(np.float64) - b).max())
            rec['T1_units'] = u
            v, tf, u = read2(ent['M1', per], 'PRES')
            if int(tf[k, 0, 0]) != want:
                raise ValueError('M1 TFLAG')
            x = v[k, 0].astype(np.float64)
            mv = (x - 50000.0) / 60000.0
            st = np.asarray(f16[i, j1]).astype(np.float64)
            tol = half_ulp16(np.asarray(f16[i, j1]))
            rec['M1_within'] = int((np.abs(st - mv) <= tol).sum())
            rec['M1_exact_cast'] = int((mv.astype(np.float16) == np.asarray(f16[i, j1])).sum())
            rec['M1_max_abs_pa'] = float(np.abs(st * 60000.0 + 50000.0 - x).max())
            v, tf, u = read2(ent['M4', per], 'UWIND')
            if int(tf[k, 0, 0]) != want or v.shape[2:] != (300, 460):
                raise ValueError('M4 TFLAG or shape')
            w = v[k, 0].astype(np.float64)
            cell = (w[:-1, :-1] + w[:-1, 1:] + (w[1:, :-1] + w[1:, 1:])) / 4.0
            st = np.asarray(f16[i, j4]).astype(np.float64)
            tol = half_ulp16(np.asarray(f16[i, j4]))
            rec['M4_within'] = int((np.abs(st - cell) <= tol).sum())
            rec['M4_exact_cast'] = int((cell.astype(np.float16) == np.asarray(f16[i, j4])).sum())
            rec['M4_max_abs'] = float(np.abs(st - cell).max())
            n = rec['T1_cells']
            rec['pass'] = rec['T1_exact'] == n and rec['M1_within'] == n and (rec['M4_within'] == n)
            ok = ok and rec['pass']
            out['days'].append(rec)
        del f16, tgt
        print('%s %d checked' % (S.now_local(), y), flush=True)
    n = S.NROWS * S.NCOLS
    out['summary'] = {'days': len(out['days']), 'cells_per_day': n, 'T1_exact_all': all((r['T1_exact'] == n for r in out['days'])), 'M1_within_all': all((r['M1_within'] == n for r in out['days'])), 'M4_within_all': all((r['M4_within'] == n for r in out['days'])), 'M1_exact_cast_cells': sum((r['M1_exact_cast'] for r in out['days'])), 'M4_exact_cast_cells': sum((r['M4_exact_cast'] for r in out['days'])), 'M1_max_abs_pa': max((r['M1_max_abs_pa'] for r in out['days'])), 'M4_max_abs': max((r['M4_max_abs'] for r in out['days'])), 'T1_max_abs': max((r['T1_max_abs'] for r in out['days'])), 'pass': ok}
    out['end'] = S.now_local()
    S.write_json(os.path.join(S.WORK_S3, 'second_reading.json'), out)
    print(json.dumps(out['summary']))
if __name__ == '__main__':
    main()
