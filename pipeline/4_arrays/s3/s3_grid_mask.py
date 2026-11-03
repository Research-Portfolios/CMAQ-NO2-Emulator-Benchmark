"""Checks the grid of the RSIG answers against MCIP's GRIDCRO2D (cell centers within 0.0001 degree, the dot grid offset by half a cell) and builds the land mask: the cells whose center lies inside the 48 contiguous states and the District of Columbia (Natural Earth) and that MCIP marks as land."""
import os
import sys
sys.path.insert(0, '/content/emul/run/code')
import numpy as np
from s3 import s3common as S
STATES49 = sorted(['Alabama', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'District of Columbia', 'Florida', 'Georgia', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming'])
KEYS = ('GDTYP', 'P_ALP', 'P_BET', 'P_GAM', 'XCENT', 'YCENT', 'XORIG', 'YORIG', 'XCELL', 'YCELL', 'NCOLS', 'NROWS')
NE_REL = os.path.join(*'work\\natural_earth\\shapefiles\\natural_earth\\cultural\\ne_10m_admin_1_states_provinces'.split('\\'))

def read_gridcro(inputs):
    import netCDF4
    e = [x for x in inputs['mcip'] if os.path.basename(x['path']) == 'GRIDCRO2D_20160701.nc4'][0]
    p = S.path_abs(e['path'])
    if S.sha256_file(p) != e['sha256']:
        raise ValueError('GRIDCRO2D sha256 differs from the inventory')
    ds = netCDF4.Dataset(p, 'r')
    try:
        out = {'attrs': {a: S._attr(ds, a) for a in KEYS}, 'entry': e}
        for n in ('LAT', 'LON', 'HT', 'LWMASK', 'PURB'):
            v = ds.variables[n]
            v.set_auto_maskandscale(False)
            out[n] = np.array(v[:])[0, 0]
            out[n + '_units'] = str(getattr(v, 'units', '')).strip()
    finally:
        ds.close()
    return out

def lcc(attrs):
    import pyproj
    p4 = '+proj=lcc +lat_1=%s +lat_2=%s +lat_0=%s +lon_0=%s +x_0=0 +y_0=0 +units=m +no_defs +a=6370000 +b=6370000' % (attrs['P_ALP'], attrs['P_BET'], attrs['YCENT'], attrs['P_GAM'])
    return (pyproj.Proj(p4), p4)

def centres(attrs):
    proj, p4 = lcc(attrs)
    nc, nr = (int(attrs['NCOLS']), int(attrs['NROWS']))
    x = attrs['XORIG'] + (np.arange(nc) + 0.5) * attrs['XCELL']
    y = attrs['YORIG'] + (np.arange(nr) + 0.5) * attrs['YCELL']
    xx, yy = np.meshgrid(x, y)
    lon, lat = proj(xx, yy, inverse=True)
    return (np.asarray(lon), np.asarray(lat), p4)

def maxdiff(a, b):
    d = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))
    return float(d.max())

def step1(inputs, g):
    idx = S.rsig_index(inputs)
    cross = S.read_answer(idx['T1', '201501'], 'NO2', coords=True)
    dot = S.read_answer(idx['M4', '201501'], 'UWIND', coords=True)
    ca = {k: cross['attrs'].get(k) for k in KEYS}
    da = {k: dot['attrs'].get(k) for k in KEYS}
    ga = g['attrs']
    res = {'gridcro2d_attrs': ga, 'rsig_cross_attrs': ca, 'rsig_dot_attrs': da}
    res['cross_attrs_equal'] = all((float(ca[k]) == float(ga[k]) for k in KEYS))
    want_dot = dict(ga)
    want_dot['NCOLS'] = ga['NCOLS'] + 1
    want_dot['NROWS'] = ga['NROWS'] + 1
    want_dot['XORIG'] = ga['XORIG'] - ga['XCELL'] / 2.0
    want_dot['YORIG'] = ga['YORIG'] - ga['YCELL'] / 2.0
    res['dot_attrs_half_cell_offset'] = all((float(da[k]) == float(want_dot[k]) for k in KEYS))
    lon, lat, p4 = centres(ca)
    res['proj4'] = p4
    res['centres_from_rsig_attrs_vs_gridcro2d'] = {'max_abs_dlon_deg': maxdiff(lon, g['LON']), 'max_abs_dlat_deg': maxdiff(lat, g['LAT'])}
    res['rsig_lonlat_vs_gridcro2d'] = {'max_abs_dlon_deg': maxdiff(cross['LONGITUDE'], g['LON']), 'max_abs_dlat_deg': maxdiff(cross['LATITUDE'], g['LAT'])}
    dlon, dlat, _ = centres(da)
    res['dot_points_from_attrs_vs_dot_lonlat'] = {'max_abs_dlon_deg': maxdiff(dlon, dot['LONGITUDE']), 'max_abs_dlat_deg': maxdiff(dlat, dot['LATITUDE'])}
    corner_x = da['XORIG'] + 0.5 * da['XCELL']
    res['dot_point_00_is_cross_corner'] = corner_x == ga['XORIG'] and da['YORIG'] + 0.5 * da['YCELL'] == ga['YORIG']
    lim = 0.0001
    vals = [res['centres_from_rsig_attrs_vs_gridcro2d'], res['rsig_lonlat_vs_gridcro2d'], res['dot_points_from_attrs_vs_dot_lonlat']]
    res['within_1e-4'] = all((v['max_abs_dlon_deg'] <= lim and v['max_abs_dlat_deg'] <= lim for v in vals))
    res['shapes'] = {'cross': list(cross['values'].shape), 'dot': list(dot['values'].shape)}
    cell = S.dot_to_cell(dot['values'][:, 0])
    res['dot_to_cell_shape'] = list(cell.shape)
    res['pass'] = bool(res['cross_attrs_equal'] and res['dot_attrs_half_cell_offset'] and res['within_1e-4'] and res['dot_point_00_is_cross_corner'] and (res['dot_to_cell_shape'] == [31, 299, 459]))
    return res

def step2(inputs, g):
    import shapefile
    import shapely
    from shapely.geometry import shape as to_shape
    ne_files = {}
    for e in inputs['ne']:
        if e['path'].startswith(NE_REL + '.'):
            ne_files[e['path'][len(NE_REL):]] = e
    need = ('.shp', '.shx', '.dbf', '.prj', '.cpg')
    miss = [x for x in need if x not in ne_files]
    if miss:
        raise ValueError('Natural Earth files missing: %s' % miss)
    for ext, e in ne_files.items():
        if S.sha256_file(S.path_abs(e['path'])) != e['sha256']:
            raise ValueError('Natural Earth %s sha256 differs' % ext)
    with open(S.path_abs(ne_files['.prj']['path']), 'r', encoding='ascii', errors='replace') as f:
        prj = f.read().strip()
    r = shapefile.Reader(S.path_abs(NE_REL + '.shp'), encoding='utf-8')
    fields = [f[0] for f in r.fields[1:]]
    low = {f.lower(): f for f in fields}
    fa, fn = (low['adm0_a3'], low['name'])
    us, chosen, geoms = ([], [], [])
    for sr in r.iterShapeRecords():
        rec = sr.record.as_dict()
        if rec[fa] != 'USA':
            continue
        us.append(rec[fn])
        if rec[fn] in STATES49:
            chosen.append(rec[fn])
            geoms.append(to_shape(sr.shape.__geo_interface__))
    r.close()
    res = {'file': NE_REL + '.shp', 'prj_first_40': prj[:40], 'us_records': len(us), 'us_other': sorted(set(us) - set(STATES49)), 'chosen': len(chosen), 'chosen_unique': len(set(chosen)), 'missing': sorted(set(STATES49) - set(chosen))}
    if res['missing'] or res['chosen_unique'] != 49:
        raise ValueError('states: %s' % res)
    union = shapely.union_all(geoms)
    shapely.prepare(union)
    lon = g['LON'].astype(np.float64)
    lat = g['LAT'].astype(np.float64)
    inside = shapely.contains_xy(union, lon, lat)
    land = g['LWMASK'] == 1.0
    mask = np.asarray(inside & land, dtype=bool)
    res.update({'lwmask_values': sorted(set(np.unique(g['LWMASK']).tolist())), 'cells_inside_union': int(inside.sum()), 'cells_lwmask_1': int(land.sum()), 'cells_mask': int(mask.sum()), 'cells_inside_not_land': int((inside & ~land).sum()), 'cells_land_not_inside': int((land & ~inside).sum()), 'union_type': union.geom_type, 'union_valid': bool(shapely.is_valid(union))})
    os.makedirs(S.ARRAYS, exist_ok=True)
    out = os.path.join(S.ARRAYS, 'mask_conus_land.npy')
    if os.path.exists(out):
        raise FileExistsError(out)
    part = out + '.partial'
    with open(part, 'wb') as f:
        np.save(f, mask, allow_pickle=False)
    os.replace(part, out)
    back = np.load(out)
    res['mask_file'] = S.file_info(out)
    res['mask_dtype'] = str(back.dtype)
    res['mask_shape'] = list(back.shape)
    res['mask_reload_equal'] = bool(np.array_equal(back, mask))
    return res

def main():
    t0 = S.now_local()
    inputs = S.inventory_inputs()
    g = read_gridcro(inputs)
    res = {'start': t0, 'step1': step1(inputs, g)}
    res['gridcro2d_units'] = {n: g[n + '_units'] for n in ('LAT', 'LON', 'HT', 'LWMASK', 'PURB')}
    res['step2'] = step2(inputs, g)
    res['end'] = S.now_local()
    os.makedirs(S.WORK_S3, exist_ok=True)
    S.write_json(os.path.join(S.WORK_S3, 'grid_mask.json'), res)
    print('step1 pass %s; step2 mask cells %d' % (res['step1']['pass'], res['step2']['cells_mask']))
if __name__ == '__main__':
    main()
