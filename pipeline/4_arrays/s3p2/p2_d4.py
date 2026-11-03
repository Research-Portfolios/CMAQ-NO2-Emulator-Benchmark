"""Checks every input of the second pass before the build, as the first pass did and extended to the dominant land-use field and the point and marine NOx inputs: the RSIG answers against the published 12 km grid, the MCIP files and the emission extracts."""
import multiprocessing as mp
import os
import sys
import time
sys.path.insert(0, '/content/emul/run/code')
import numpy as np
from s3 import s3_d4 as D
from s3 import s3common as S
from s3p2 import p2_grid_mask as GM
from s3p2 import p2common as P

def check_answer_p2(e):
    out = D.check_answer(e)
    if e['series'] in ('M4', 'M5'):
        import netCDF4
        var = S.SERIES[e['series']][0]
        data, _chk = S.decompress_answer(e, check_raw=False)
        ds = netCDF4.Dataset('inmemory.nc', mode='r', memory=data)
        try:
            v = ds.variables[var]
            out['var_attrs'] = {k: P.common.mask(' '.join(str(getattr(v, k)).split())) for k in ('long_name', 'var_desc', 'units') if k in v.ncattrs()}
        finally:
            ds.close()
        del data
    return out

def check_mcip_p2(inputs, mask):
    import netCDF4
    out = {}
    for e in sorted(inputs['mcip'], key=lambda x: x['path']):
        name = os.path.basename(e['path'])
        p = S.path_abs(e['path'])
        rec = {'sha256_ok': S.sha256_file(p) == e['sha256'], 'bytes_ok': os.stat(p).st_size == int(e['bytes'])}
        if name.startswith('LUFRAC'):
            rec['used'] = False
            out[name] = rec
            continue
        ds = netCDF4.Dataset(p, 'r')
        try:
            rec['data_model'] = ds.data_model
            a = {k: S._attr(ds, k) for k in D.GRID_KEYS}
            rec['grid_diff'] = [k for k in D.GRID_KEYS if float(a[k]) != float(D.GRIDCRO[k])]
            t = ds.variables['TFLAG']
            t.set_auto_maskandscale(False)
            tf = np.array(t[:])
            rec['tflag_first'] = tf[0, 0].tolist()
            rec['tflag_last'] = tf[-1, 0].tolist()
            rec['steps'] = int(tf.shape[0])
            gridcro = name.startswith('GRIDCRO')
            names = ('HT', 'LWMASK', 'PURB', 'DLUSE', 'LAT', 'LON') if gridcro else ('VEG',)
            if not gridcro:
                rec['present'] = {n: n in ds.variables for n in ('GSW', 'RGRND', 'ALBEDO')}
            rec['vars'] = {}
            for n in names:
                v = ds.variables[n]
                v.set_auto_maskandscale(False)
                x = np.array(v[:])
                vr = {'units': str(getattr(v, 'units', '')).strip(), 'shape': list(x.shape), 'dtype': str(x.dtype)}
                if n == 'VEG':
                    day = int(tf[0, 0, 0])
                    use = [k for k in range(tf.shape[0]) if int(tf[k, 0, 0]) == day]
                    vr['records_used'] = [use[0], use[-1]]
                    vr['records_used_n'] = len(use)
                    vr['record_hours'] = sorted(set((int(tf[k, 0, 1]) for k in use)))
                    x = x[use]
                f = x[:, 0]
                vr['fill'] = int((f <= S.NULL_LIMIT).sum())
                vr['nan'] = int(np.isnan(f).sum())
                g = f[np.isfinite(f) & (f > S.NULL_LIMIT)]
                vr['min'] = float(g.min())
                vr['max'] = float(g.max())
                fm = f[:, mask]
                vr['fill_or_nan_mask'] = int(((fm <= S.NULL_LIMIT) | np.isnan(fm)).sum())
                if n in ('LWMASK', 'DLUSE'):
                    vals = sorted(set(np.unique(f).tolist()))
                    vr['values'] = vals
                if n == 'DLUSE':
                    vr['n_values'] = len(vr['values'])
                    vr['all_whole_numbers'] = bool(np.all(np.asarray(vr['values']) == np.floor(np.asarray(vr['values']))))
                    vr['values_on_masked_cells'] = sorted(set(np.unique(f[0][mask]).tolist()))
                    vr['float16_exact'] = bool(np.all(f.astype(np.float16).astype(np.float64) == f.astype(np.float64)))
                if n == 'PURB':
                    vr['below_0'] = int((g < 0).sum())
                    vr['above_100'] = int((g > 100).sum())
                if n == 'VEG':
                    vr['below_0'] = int((g < 0).sum())
                    vr['above_1'] = int((g > 1).sum())
                rec['vars'][n] = vr
        finally:
            ds.close()
        out[name] = rec
    return out

def main():
    a = sys.argv[1:]
    workers = int(a[a.index('--workers') + 1]) if '--workers' in a else 3
    S.common.set_temp()
    P.ensure_dirs()
    same = all((float(D.GRIDCRO[k]) == float(v) for k, v in GM.EPA_CROSS.items())) and all((float(D.DOT[k]) == float(v) for k, v in GM.EPA_DOT.items()))
    if not same:
        raise SystemExit("the earlier version's grid values differ from the study design's")
    t_start = P.now_local()
    inv = S.load_inventory()
    inputs = S.inventory_inputs(inv)
    mask_path = os.path.join(P.ARRAYS_P2, 'mask_conus_land.npy')
    mask = np.load(mask_path)
    rs = sorted(inputs['rsig'], key=lambda e: (e['series'], e['period']))
    res = {'start': t_start, 'workers': workers, 'mask_cells': int(mask.sum()), 'mask': P.file_info(mask_path), 'ad55_grid_values_equal_epa': same, 'answers': []}
    t0 = time.time()
    ctx = mp.get_context('spawn')
    with ctx.Pool(workers, initializer=D._init, initargs=(mask_path,)) as pool:
        for k, out in enumerate(pool.imap(check_answer_p2, rs, chunksize=2)):
            res['answers'].append(out)
            if (k + 1) % 50 == 0:
                print('%s answers %d of %d, %.0f s, memory %.2f GB' % (P.now_local(), k + 1, len(rs), time.time() - t0, P.mem_gb()), flush=True)
    res['rsig_seconds'] = round(time.time() - t0, 1)
    res['mcip'] = check_mcip_p2(inputs, mask)
    t1 = time.time()
    ex = {}
    for kind in ('emis_merged', 'pt_marine'):
        for e in sorted(inputs[kind], key=lambda x: x['path']):
            r = D.check_extract(e)
            r['kind'] = kind
            ex[e['path']] = r
    res['extracts'] = ex
    res['extract_seconds'] = round(time.time() - t1, 1)
    res['end'] = P.now_local()
    res['memory_gb_end'] = round(P.mem_gb(), 2)
    info = P.write_json(os.path.join(P.WORK_P2, 'd4_stats.json'), res)
    print('done %s; answers %d; extracts %d; %s bytes' % (res['end'], len(res['answers']), len(ex), info['bytes']))
if __name__ == '__main__':
    P.run_main(main)
