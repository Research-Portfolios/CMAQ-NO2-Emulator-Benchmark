"""Checks the grid of every MCIP file and RSIG answer against the published 12 km grid, records the wind attributes and the MCIP and WRF versions, checks the MCIP variable lists, builds the land mask from Natural Earth's states with lakes removed and MCIP's land flag, and finds the cells of the ten cities."""
import os
import re
import sys
sys.path.insert(0, '/content/emul/run/code')
import numpy as np
from s3 import s3_grid_mask as G1
from s3 import s3common as S
from s3p2 import p2common as P
EPA_COMMON = {'P_ALP': 33.0, 'P_BET': 45.0, 'P_GAM': -97.0, 'XCENT': -97.0, 'YCENT': 40.0, 'XCELL': 12000.0, 'YCELL': 12000.0}
EPA_CROSS = dict(EPA_COMMON, NCOLS=459, NROWS=299, XORIG=-2556000.0, YORIG=-1728000.0)
EPA_DOT = dict(EPA_COMMON, NCOLS=460, NROWS=300, XORIG=-2562000.0, YORIG=-1734000.0)
LAKES = os.path.join(P.NE_EXTRACTED, 'ne_10m_admin_1_states_provinces_lakes', 'ne_10m_admin_1_states_provinces_lakes')
PLACES = os.path.join(P.NE_EXTRACTED, 'ne_10m_populated_places', 'ne_10m_populated_places')
CITIES = ('New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'Austin')
PAPER_NAMES = {'New York': 'New York City'}
VERSION_RE = (('MCIP', re.compile('MCIP\\s*(?:V(?:ERSION)?\\s*)?(\\d+(?:\\.\\d+)+)', re.I)), ('WRF', re.compile('WRF(?:\\s*ARW)?\\s*(?:V(?:ERSION)?\\s*)?(\\d+(?:\\.\\d+)+)', re.I)))
TEXT_ATTRS = ('FILEDESC', 'EXEC_ID', 'UPNAM', 'HISTORY')

def epa_check(attrs, want):
    bad = []
    for k, v in want.items():
        a = attrs.get(k)
        if a is None or float(a) != float(v):
            bad.append(k)
    return bad

def version_tokens(ds):
    out = {}
    for a in TEXT_ATTRS:
        if a not in ds.ncattrs():
            continue
        v = getattr(ds, a)
        if isinstance(v, bytes):
            v = v.decode('ascii', 'replace')
        v = str(v)
        for tag, rx in VERSION_RE:
            for m in rx.finditer(v):
                out.setdefault(tag, set()).add('%s %s in %s' % (tag, m.group(1), a))
    return {k: sorted(v) for k, v in out.items()}

def mcip_meta(inputs):
    import netCDF4
    from choi_dl import prov
    rows = prov.load(os.path.join(P.W, 'raw', 'mcip'))
    res = {}
    for e in sorted(inputs['mcip'], key=lambda x: x['path']):
        e = dict(e)
        row = rows.get(e['path'], {})
        e['source'] = row.get('source', '')
        e['prov_version'] = row.get('version', '')
        p = S.path_abs(e['path'])
        name = os.path.basename(p)
        if S.sha256_file(p) != e['sha256']:
            raise ValueError('%s differs from the inventory' % name)
        ds = netCDF4.Dataset(p, 'r')
        try:
            attrs = {k: S._attr(ds, k) for k in EPA_CROSS}
            r = {'epa_grid_mismatch': epa_check(attrs, EPA_CROSS), 'versions': version_tokens(ds), 'variables': [str(v) for v in ds.variables], 'source': P.common.mask(e.get('source', ''))[:200], 'provenance_version': P.common.mask(e.get('prov_version', ''))[:120]}
        finally:
            ds.close()
        m = re.search('MCIPv(\\d)(\\d)_WRFv(\\d)(\\d)(\\d)', r['source'])
        if m:
            r['tar_name_versions'] = 'MCIP %s.%s, WRF %s.%s.%s' % m.groups()
        res[name] = r
    return res

def wind_attrs(inputs):
    import netCDF4
    idx = S.rsig_index(inputs)
    out = {}
    for ser, var in (('M4', 'UWIND'), ('M5', 'VWIND')):
        data, _chk = S.decompress_answer(idx[ser, '201501'])
        ds = netCDF4.Dataset('inmemory.nc', mode='r', memory=data)
        try:
            v = ds.variables[var]
            out[var] = {k: P.common.mask(' '.join(str(getattr(v, k)).split())) for k in ('long_name', 'var_desc', 'units') if k in v.ncattrs()}
            a = {k: S._attr(ds, k) for k in EPA_DOT}
            out[var]['epa_grid_mismatch'] = epa_check(a, EPA_DOT)
            out[var]['answer'] = idx[ser, '201501']['path']
        finally:
            ds.close()
        del data
    data, _chk = S.decompress_answer(idx['T1', '201501'])
    ds = netCDF4.Dataset('inmemory.nc', mode='r', memory=data)
    try:
        a = {k: S._attr(ds, k) for k in EPA_CROSS}
        out['T1_201501_epa_grid_mismatch'] = epa_check(a, EPA_CROSS)
    finally:
        ds.close()
    return out

def expectation(meta):
    g = meta['GRIDCRO2D_20160701.nc4']['variables']
    mets = {n: r['variables'] for n, r in meta.items() if n.startswith('METCRO2D_')}
    out = {'GRIDCRO2D': {v: v in g for v in ('HT', 'LWMASK', 'PURB', 'DLUSE', 'LAT', 'LON')}}
    allm = {}
    for v in ('VEG', 'GSW', 'RGRND', 'ALBEDO'):
        allm[v] = sorted(set((v in lst for lst in mets.values())))
    out['METCRO2D_all_12'] = allm
    out['METCRO2D_data_variables'] = sorted(set((len([x for x in lst if x != 'TFLAG']) for lst in mets.values())))
    out['as_expected'] = bool(all(out['GRIDCRO2D'].values()) and allm['VEG'] == [True] and (allm['GSW'] == [True]) and (allm['RGRND'] == [True]) and (allm['ALBEDO'] == [False]))
    return out

def mask_step(g):
    import shapefile
    import shapely
    from shapely.geometry import shape as to_shape
    with open(LAKES + '.prj', 'r', encoding='ascii', errors='replace') as f:
        prj = f.read().strip()
    r = shapefile.Reader(LAKES + '.shp', encoding='utf-8')
    fields = [f[0] for f in r.fields[1:]]
    low = {f.lower(): f for f in fields}
    fa, fn = (low['adm0_a3'], low['name'])
    us, chosen, geoms = ([], [], [])
    for sr in r.iterShapeRecords():
        rec = sr.record.as_dict()
        if rec[fa] != 'USA':
            continue
        us.append(rec[fn])
        if rec[fn] in G1.STATES49:
            chosen.append(rec[fn])
            geoms.append(to_shape(sr.shape.__geo_interface__))
    r.close()
    res = {'file': P.common.rel(LAKES + '.shp'), 'prj_first_40': prj[:40], 'us_records': len(us), 'us_other': sorted(set(us) - set(G1.STATES49)), 'chosen': len(chosen), 'chosen_unique': len(set(chosen)), 'missing': sorted(set(G1.STATES49) - set(chosen))}
    if res['missing'] or res['chosen_unique'] != 49 or res['chosen'] != 49:
        raise ValueError('states: %s' % res)
    union = shapely.union_all(geoms)
    shapely.prepare(union)
    lon = g['LON'].astype(np.float64)
    lat = g['LAT'].astype(np.float64)
    inside = shapely.contains_xy(union, lon, lat)
    land = g['LWMASK'] == 1.0
    mask = np.asarray(inside & land, dtype=bool)
    res.update({'lwmask_values': sorted(set(np.unique(g['LWMASK']).tolist())), 'cells_inside_union': int(inside.sum()), 'cells_lwmask_1': int(land.sum()), 'cells_mask': int(mask.sum()), 'cells_inside_not_land': int((inside & ~land).sum()), 'cells_land_not_inside': int((land & ~inside).sum()), 'union_type': union.geom_type, 'union_valid': bool(shapely.is_valid(union))})
    out = os.path.join(P.ARRAYS_P2, 'mask_conus_land.npy')
    if os.path.exists(out):
        if not np.array_equal(np.load(out), mask):
            raise SystemExit('the mask on disk differs from the one computed now: stop')
        res['mask_written_now'] = False
    else:
        part = out + '.partial'
        with open(part, 'wb') as f:
            np.save(f, mask, allow_pickle=False)
        os.replace(part, out)
        res['mask_written_now'] = True
    back = np.load(out)
    res['mask_file'] = P.file_info(out)
    res['mask_dtype'] = str(back.dtype)
    res['mask_shape'] = list(back.shape)
    res['mask_reload_equal'] = bool(np.array_equal(back, mask))
    old_p = os.path.join(P.ARRAYS_B, 'mask_conus_land.npy')
    if os.path.isfile(old_p):
        old = np.load(old_p)
        res['for_the_record_vs_item_b_mask'] = {'cells_item_b': int(old.sum()), 'cells_in_this_not_item_b': int((mask & ~old).sum()), 'cells_in_item_b_not_this': int((old & ~mask).sum())}
    else:
        res['for_the_record_vs_item_b_mask'] = {'absent': P.common.rel(old_p), 'note': "the earlier version's mask is not on this machine: not compared (record only; chooses nothing)"}
    return (res, mask)

def city_step(g, attrs, mask):
    import pyproj
    import shapefile
    r = shapefile.Reader(PLACES + '.shp', encoding='utf-8')
    fields = [f[0] for f in r.fields[1:]]
    need = ('NAME', 'ADM0_A3', 'ADM0NAME', 'POP_MAX', 'LATITUDE', 'LONGITUDE', 'ADM1NAME')
    up = {f.upper(): f for f in fields}
    miss = [k for k in need if k not in up]
    if miss:
        raise ValueError('populated places lacks %s' % miss)
    cand = {c: [] for c in CITIES}
    for i, sr in enumerate(r.iterShapeRecords()):
        rec = sr.record.as_dict()
        nm = rec[up['NAME']]
        if nm not in cand:
            continue
        if not (rec[up['ADM0_A3']] == 'USA' or rec[up['ADM0NAME']] == 'United States of America'):
            continue
        pt = sr.shape.points[0]
        cand[nm].append({'record': i, 'name': nm, 'adm1': rec[up['ADM1NAME']], 'pop_max': rec[up['POP_MAX']], 'lon': float(pt[0]), 'lat': float(pt[1]), 'attr_lon': float(rec[up['LONGITUDE']]), 'attr_lat': float(rec[up['LATITUDE']])})
    r.close()
    p4 = '+proj=lcc +lat_1=%r +lat_2=%r +lat_0=%r +lon_0=%r +x_0=0 +y_0=0 +a=6370000 +b=6370000 +units=m +no_defs' % (float(attrs['P_ALP']), float(attrs['P_BET']), float(attrs['YCENT']), float(attrs['P_GAM']))
    proj = pyproj.Proj(p4)
    out = []
    lon_c = g['LON'].astype(np.float64)
    lat_c = g['LAT'].astype(np.float64)
    for c in CITIES:
        lst = cand[c]
        if not lst:
            raise ValueError('no record for %s' % c)
        best = max(lst, key=lambda d: d['pop_max'] or 0)
        x, y = proj(best['lon'], best['lat'])
        col = int(np.floor((x - float(attrs['XORIG'])) / float(attrs['XCELL'])))
        row = int(np.floor((y - float(attrs['YORIG'])) / float(attrs['YCELL'])))
        d2 = (lon_c - best['lon']) ** 2 + (lat_c - best['lat']) ** 2
        nr, nc = np.unravel_index(int(np.argmin(d2)), d2.shape)
        block = [[rr, cc] for rr in range(row - 1, row + 2) for cc in range(col - 1, col + 2) if 0 <= rr < P.NROWS and 0 <= cc < P.NCOLS]
        bm = [[rr, cc] for rr, cc in block if bool(mask[rr, cc])]
        out.append({'city': c, 'paper_name': PAPER_NAMES.get(c, c), 'ne_record': best['record'], 'ne_name': best['name'], 'ne_adm1': best['adm1'], 'pop_max': best['pop_max'], 'records_with_name_in_us': len(lst), 'pop_max_of_others': sorted((d['pop_max'] for d in lst if d is not best)), 'lon': best['lon'], 'lat': best['lat'], 'point_equals_attributes': bool(abs(best['lon'] - best['attr_lon']) < 1e-06 and abs(best['lat'] - best['attr_lat']) < 1e-06), 'attr_lon_minus_point': best['attr_lon'] - best['lon'], 'attr_lat_minus_point': best['attr_lat'] - best['lat'], 'attr_point_cell': [int(np.floor((proj(best['attr_lon'], best['attr_lat'])[1] - float(attrs['YORIG'])) / float(attrs['YCELL']))), int(np.floor((proj(best['attr_lon'], best['attr_lat'])[0] - float(attrs['XORIG'])) / float(attrs['XCELL'])))], 'x_m': float(x), 'y_m': float(y), 'row': row, 'col': col, 'nearest_centre_row_col': [int(nr), int(nc)], 'cell_masked': bool(mask[row, col]), 'block_3x3': block, 'block_masked': bm, 'n_block_masked': len(bm)})
    return {'proj4': p4, 'cities': out}

def main():
    S.common.set_temp()
    P.ensure_dirs()
    t0 = P.now_local()
    inputs = S.inventory_inputs()
    g = G1.read_gridcro(inputs)
    res = {'start': t0, 'step1': G1.step1(inputs, g)}
    res['step1']['epa_grid_gridcro2d_mismatch'] = epa_check(g['attrs'], EPA_CROSS)
    res['gridcro2d_units'] = {n: g[n + '_units'] for n in ('LAT', 'LON', 'HT', 'LWMASK', 'PURB')}
    meta = mcip_meta(inputs)
    res['ad81_2_mcip_members'] = meta
    res['ad81_1_wind'] = wind_attrs(inputs)
    res['ad81_3_expectation'] = expectation(meta)
    bad = {n: r['epa_grid_mismatch'] for n, r in meta.items() if r['epa_grid_mismatch']}
    bad.update({k: v for k, v in (('UWIND 201501', res['ad81_1_wind']['UWIND']['epa_grid_mismatch']), ('VWIND 201501', res['ad81_1_wind']['VWIND']['epa_grid_mismatch']), ('T1 201501', res['ad81_1_wind']['T1_201501_epa_grid_mismatch']), ('GRIDCRO2D', res['step1']['epa_grid_gridcro2d_mismatch'])) if v})
    res['ad55_mismatch'] = bad
    if bad or not res['step1']['pass']:
        P.write_json(os.path.join(P.WORK_P2, 'grid_mask.json'), res)
        raise SystemExit('STOP for the user: grid check %s' % bad)
    res['step2'], mask = mask_step(g)
    cities = city_step(g, g['attrs'], mask)
    res['cities'] = cities
    doc = {'run': 'emul', 'stage': 'the second pass, the city cells', 'made': P.now_local(), 'design': "the study design's city rule (POP_MAX); Natural Earth populated places", 'source': P.common.rel(PLACES + '.shp'), 'rule': "names matched in the US (ADM0_A3 USA or ADM0NAME United States of America); where names repeat, the largest POP_MAX; the cell holding the record's point (Lambert conformal, sphere 6370000 m, GRIDCRO2D's attributes); the series is the mean over the masked cells of the 3 x 3 block centred on it", 'order': 'PAPER Fig. 5: New York City, Los Angeles; Chicago, Houston; Phoenix, Philadelphia; San Antonio, San Diego; Dallas, Austin', 'mask': res['step2']['mask_file'], 'row_col': '0-based [row, col] of the 299 x 459 arrays (row 0 the south edge, as the arrays are stored)', 'proj4': cities['proj4'], 'cities': cities['cities']}
    res['city_cells_file'] = P.write_json(os.path.join(P.CFG_P2, 'CITY_CELLS.json'), doc)
    res['end'] = P.now_local()
    P.write_json(os.path.join(P.WORK_P2, 'grid_mask.json'), res)
    print('step1 pass %s; mismatches %s; expectation met %s; mask cells %d; cities %d' % (res['step1']['pass'], bad or 'none', res['ad81_3_expectation']['as_expected'], res['step2']['cells_mask'], len(cities['cities'])))
if __name__ == '__main__':
    P.run_main(main)
