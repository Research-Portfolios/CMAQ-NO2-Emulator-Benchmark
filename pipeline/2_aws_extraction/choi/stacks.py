"""Sums point and marine emission stacks into grid cells: takes each stack's daily mean of NO, NO2 and HONO over the 24 hourly records and adds it to its cell of the 299 x 459 grid by the row and column in the stack groups file, counting the stacks outside the grid."""
import numpy as np
SPECIES = ('NO', 'NO2', 'HONO')
GRID = (299, 459)

def open_dataset(src):
    import netCDF4
    if isinstance(src, netCDF4.Dataset):
        return (src, False)
    if isinstance(src, (bytes, bytearray, memoryview)):
        return (netCDF4.Dataset('inmemory.nc', mode='r', memory=bytes(src)), True)
    return (netCDF4.Dataset(str(src), mode='r'), True)

def _var4(ds, name):
    v = ds.variables[name]
    if len(v.dimensions) != 4:
        raise ValueError('%s has dimensions %s, not (TSTEP, LAY, ROW, COL)' % (name, v.dimensions))
    return v

def hours_check(ds, n_hours=24):
    if 'TFLAG' not in ds.variables:
        return {'ok': False, 'reason': 'TFLAG missing', 'n_records': None}
    tf = np.asarray(ds.variables['TFLAG'][:])
    n_rec = int(tf.shape[0])
    out = {'n_records': n_rec}
    if n_rec < n_hours:
        out.update(ok=False, reason='%d records, fewer than %d' % (n_rec, n_hours))
        return out
    d = tf[:n_hours, 0, 0].astype(np.int64)
    t = tf[:n_hours, 0, 1].astype(np.int64)
    want = np.arange(n_hours, dtype=np.int64) * 10000
    same_var = bool((tf[:n_hours, :, :] == tf[:n_hours, :1, :]).all())
    ok = bool((d == d[0]).all() and (t == want).all() and same_var)
    out.update(ok=ok, date=int(d[0]), first_time=int(t[0]), last_time=int(t[-1]), reason=None if ok else 'records 0 to %d are not hours 00 to %02d of one date (dates %s, times %s)' % (n_hours - 1, n_hours - 1, sorted(set(d.tolist()))[:3], t.tolist()[:3]))
    if n_rec > n_hours:
        out['next_record'] = [int(tf[n_hours, 0, 0]), int(tf[n_hours, 0, 1])]
    return out

def stack_daily_means(ds, species=SPECIES, n_hours=24, return_hourly=False):
    means, units, hourly, missing = ({}, {}, {}, [])
    n_nf, n_neg, n_stk = ({}, {}, None)
    for sp in species:
        if sp not in ds.variables:
            missing.append(sp)
            continue
        v = _var4(ds, sp)
        tstep, lay, row, col = v.shape
        if lay != 1 or col != 1:
            raise ValueError('%s has LAY %d and COL %d (expected 1 and 1)' % (sp, lay, col))
        if tstep < n_hours:
            raise ValueError('%s has %d records' % (sp, tstep))
        a = np.asarray(v[0:n_hours, 0, :, 0], dtype=np.float64)
        m = a.mean(axis=0)
        means[sp] = m
        units[sp] = str(getattr(v, 'units', '')).strip()
        n_nf[sp] = int((~np.isfinite(m)).sum())
        n_neg[sp] = int((m < 0).sum())
        if return_hourly:
            hourly[sp] = a
        n_stk = row if n_stk is None else n_stk
        if row != n_stk:
            raise ValueError('species with different stack counts')
    out = {'means': means, 'units': units, 'missing': missing, 'n_nonfinite': n_nf, 'n_negative': n_neg, 'n_stacks': n_stk}
    if return_hourly:
        out['hourly'] = hourly
    return out

def _whole(a, name):
    a = np.asarray(a)
    if a.dtype.kind in 'iu':
        return (a.astype(np.int64), None)
    if not np.all(np.isfinite(a)) or not np.all(np.mod(a, 1) == 0):
        return (None, '%s holds values that are not whole numbers' % name)
    return (a.astype(np.int64), None)

def groups_row_col(gds):
    out = {'reason': None}
    for name in ('ROW', 'COL'):
        if name not in gds.variables:
            out['reason'] = '%s variable missing from the groups file' % name
            return out
    r = np.asarray(_var4(gds, 'ROW')[0, 0, :, 0])
    c = np.asarray(_var4(gds, 'COL')[0, 0, :, 0])
    out['row'], e1 = _whole(r, 'ROW')
    out['col'], e2 = _whole(c, 'COL')
    out['reason'] = e1 or e2
    out['n_stacks'] = int(r.size)
    for name in ('LATITUDE', 'LONGITUDE'):
        if name in gds.variables:
            out[name.lower()] = np.asarray(_var4(gds, name)[0, 0, :, 0], dtype=np.float64)
    return out

def sum_to_cells(values, row, col, nrows=GRID[0], ncols=GRID[1], base=1):
    v = np.asarray(values, dtype=np.float64)
    r = np.asarray(row, dtype=np.int64) - int(base)
    c = np.asarray(col, dtype=np.int64) - int(base)
    if not v.shape == r.shape == c.shape:
        raise ValueError('values, row and col differ in length')
    inside = (r >= 0) & (r < nrows) & (c >= 0) & (c < ncols)
    grid = np.zeros((nrows, ncols), dtype=np.float64)
    np.add.at(grid, (r[inside], c[inside]), v[inside])
    flat = r[inside] * ncols + c[inside]
    u, cnt = np.unique(flat, return_counts=True)
    info = {'n_stacks': int(v.size), 'n_summed': int(inside.sum()), 'n_outside': int((~inside).sum()), 'n_cells_with_stacks': int(u.size), 'n_cells_with_two_or_more': int((cnt >= 2).sum()), 'max_stacks_in_a_cell': int(cnt.max()) if cnt.size else 0}
    return (grid, info)

def stack_to_cell(stack_src, groups_src, species=SPECIES, nrows=GRID[0], ncols=GRID[1], base=1, n_hours=24, strict_species=True, return_hourly=False):
    res = {'ok': False, 'reason': None, 'grids': {}, 'daily_means': {}, 'units': {}, 'base': int(base), 'species': list(species)}
    ds, c1 = open_dataset(stack_src)
    gds, c2 = open_dataset(groups_src)
    try:
        hc = hours_check(ds, n_hours)
        res['hours'] = hc
        g = groups_row_col(gds)
        n_g = g.get('n_stacks')
        res['n_stacks_groups'] = n_g
        first = [sp for sp in species if sp in ds.variables]
        n_f = int(_var4(ds, first[0]).shape[2]) if first else None
        res['n_stacks_file'] = n_f
        if g['reason']:
            res['reason'] = g['reason']
            return res
        if n_f is None or n_g is None or n_f != n_g:
            res['reason'] = 'stack count mismatch: emission file %s, stack groups file %s; the sector is dropped' % (n_f, n_g)
            return res
        if not hc['ok']:
            res['reason'] = 'hours: %s' % hc['reason']
            return res
        dm = stack_daily_means(ds, species, n_hours, return_hourly)
        res['units'] = dm['units']
        res['missing_species'] = dm['missing']
        res['n_nonfinite'] = dm['n_nonfinite']
        res['n_negative'] = dm['n_negative']
        if dm['missing'] and strict_species:
            res['reason'] = 'species missing from the file: %s' % ', '.join(dm['missing'])
            return res
        info = None
        for sp, m in dm['means'].items():
            grid, info = sum_to_cells(m, g['row'], g['col'], nrows, ncols, base)
            res['grids'][sp] = grid
            res['daily_means'][sp] = m
        if return_hourly:
            res['hourly'] = dm['hourly']
        res.update(info or {})
        res['date'] = hc.get('date')
        res['ok'] = True
        return res
    finally:
        if c1:
            ds.close()
        if c2:
            gds.close()
