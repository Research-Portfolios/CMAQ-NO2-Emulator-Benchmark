"""Prints the variables, dimensions, grid attributes, units, shapes and time-flag ends of one input of each kind (read only)."""
import os
import sys
import time
sys.path.insert(0, '.\\run\\code')
import numpy as np
from s3 import s3common as S

def show_rsig(idx, series, period):
    e = idx[series, period]
    var = S.SERIES[series][0]
    t0 = time.time()
    r = S.read_answer(e, var, coords=True)
    dt = time.time() - t0
    v = r['values']
    print('RSIG %s %s: %.2f s; format %s; checks %s' % (series, period, dt, r['format'], r['checks']))
    print('  variables %s; dims %s' % (r['variables'], r['dims']))
    print('  attrs %s' % r['attrs'])
    print('  %s units %r dims %s shape %s dtype %s' % (var, r['units'], r['var_dims'], v.shape, v.dtype))
    print('  tflag shape %s first %s last %s' % (r['tflag'].shape, r['tflag'][0, 0].tolist(), r['tflag'][-1, 0].tolist()))
    for c in ('LONGITUDE', 'LATITUDE'):
        if c in r:
            print('  %s shape %s range %.4f..%.4f' % (c, r[c].shape, float(r[c].min()), float(r[c].max())))
    print('  min %.6g max %.6g nulls %d nan %d' % (float(np.nanmin(v)), float(np.nanmax(v)), int((v <= S.NULL_LIMIT).sum()), int(np.isnan(v).sum())))

def show_nc(path, names):
    import netCDF4
    ds = netCDF4.Dataset(path, 'r')
    try:
        print('NC %s: format %s' % (os.path.basename(path), ds.data_model))
        print('  variables %s' % [str(v) for v in ds.variables])
        print('  dims %s' % {str(k): len(d) for k, d in ds.dimensions.items()})
        print('  attrs %s' % {a: S._attr(ds, a) for a in S.GRID_ATTRS if S._attr(ds, a) is not None})
        t = ds.variables['TFLAG']
        t.set_auto_maskandscale(False)
        tf = np.array(t[:])
        print('  tflag shape %s first %s last %s' % (tf.shape, tf[0, 0].tolist(), tf[-1, 0].tolist()))
        for n in names:
            v = ds.variables[n]
            v.set_auto_maskandscale(False)
            a = np.array(v[:])
            print('  %s units %r dims %s shape %s dtype %s min %.6g max %.6g' % (n, str(getattr(v, 'units', '')).strip(), [str(d) for d in v.dimensions], a.shape, a.dtype, float(a.min()), float(a.max())))
    finally:
        ds.close()

def show_npz(path):
    z = np.load(path, allow_pickle=False)
    print('NPZ %s:' % os.path.basename(path))
    for k in z.files:
        a = z[k]
        if a.ndim == 2:
            print('  %s %s %s min %.6g max %.6g' % (k, a.shape, a.dtype, float(a.min()), float(a.max())))
        else:
            print('  %s = %s' % (k, a.tolist()))

def main():
    inputs = S.inventory_inputs()
    print('kinds:', {k: len(v) for k, v in inputs.items()})
    idx = S.rsig_index(inputs)
    show_rsig(idx, 'T1', '201501')
    show_rsig(idx, 'M4', '201501')
    show_rsig(idx, 'M2', '20141231')
    mc = {os.path.basename(e['path']): S.path_abs(e['path']) for e in inputs['mcip']}
    print('mcip:', sorted(mc))
    show_nc(mc['GRIDCRO2D_20160701.nc4'], ['HT', 'LWMASK', 'PURB', 'LAT', 'LON'])
    show_nc(mc['METCRO2D_20160115.nc4'], ['VEG'])
    em = sorted(inputs['emis_merged'], key=lambda e: e['path'])
    show_npz(S.path_abs(em[0]['path']))
    pt = [e for e in inputs['pt_marine'] if e.get('sector') == 'othpt']
    show_npz(S.path_abs(sorted(pt, key=lambda e: e['path'])[0]['path']))
if __name__ == '__main__':
    main()
