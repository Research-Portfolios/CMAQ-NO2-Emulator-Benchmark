"""Computes one tar member's daily means with the same reducers as the Amazon Web Services stage: gridded emissions as the mean of the 24 hourly records at layer 1, and point and marine stacks summed into cells, optionally with each stack's NO + NO2 + HONO sum, read either inside the archive or whole."""
import gc
import io
import os
import numpy as np
import c7a_common as C
import c7a_tar as T
import ncclassic
tario, stacks, s3io = C.use_vendor()

class Opened:

    def __init__(self, fobj, ext):
        self.fobj, self.ext = (fobj, ext)

    def __enter__(self):
        self.was = gc.isenabled()
        gc.disable()
        try:
            if self.ext == 'ncf':
                self.f = ncclassic.ClassicFile(self.fobj)
            else:
                import h5py
                self.f = h5py.File(self.fobj, 'r')
        except Exception:
            if self.was:
                gc.enable()
            raise
        return self.f

    def __exit__(self, *a):
        try:
            self.f.close()
        finally:
            if self.was:
                gc.enable()
            gc.collect()

def reduce_merged(f):
    res = s3io.grid_means(f, C.MERGED_SPECIES, C.N_HOURS, C.LAYER)
    res['attrs'] = s3io.file_attrs(f)
    return res

def reduce_stack(f):
    ds = s3io.H5DS(f)
    hc = stacks.hours_check(ds, C.N_HOURS)
    out = {'means': {}, 'units': {}, 'missing': [], 'n_nonfinite': {}, 'n_negative': {}, 'n_stacks': None, 'hours': hc}
    for sp in C.PT_SPECIES:
        if sp not in f:
            out['missing'].append(sp)
            continue
        dm = stacks.stack_daily_means(ds, [sp], C.N_HOURS)
        out['means'][sp] = dm['means'][sp]
        out['units'][sp] = dm['units'][sp]
        out['n_nonfinite'][sp] = dm['n_nonfinite'][sp]
        out['n_negative'][sp] = dm['n_negative'][sp]
        if out['n_stacks'] is None:
            out['n_stacks'] = dm['n_stacks']
        elif out['n_stacks'] != dm['n_stacks']:
            raise ValueError('species with different stack counts')
        del dm
    return out

def read_selective(fh, m, ext, kind):
    win = T.LocalWindow(fh, m['data_offset'], m['size'])
    with Opened(win, ext) as f:
        res = reduce_merged(f) if kind == 'merged' else reduce_stack(f)
    dig, nb, nseg = win.digest()
    res.update(member_sha256=dig, member_bytes_read=nb, segments=nseg, member_sha256_kind='sha256 of the bytes read, offset order', reader='selective, local window')
    return res

def read_whole(fh, m, ext, kind):
    data, sha = T.read_member(fh, m)
    if ext == 'nc4':
        sp = C.MERGED_SPECIES if kind == 'merged' else C.PT_SPECIES
        res, data = s3io.whole_read(None, None, m, None, sp, 'grid' if kind == 'merged' else 'stack', C.N_HOURS, C.LAYER, data=data, zero_copy=True)
        if kind == 'merged':
            res['attrs'] = res.get('attrs') or {}
    else:
        with Opened(io.BytesIO(bytes(data)), 'ncf') as f:
            res = reduce_merged(f) if kind == 'merged' else reduce_stack(f)
    del data
    res.update(member_sha256=sha, member_bytes_read=int(m['size']), member_sha256_kind='sha256 of the whole member', reader='whole, in memory')
    return res
NON_COORD = '_nc4_non_coord_'

class NonCoord:

    def __init__(self, f):
        self.f = f

    def __contains__(self, name):
        return NON_COORD + name in self.f or name in self.f

    def __getitem__(self, name):
        if NON_COORD + name in self.f:
            return self.f[NON_COORD + name]
        return self.f[name]

def read_groups(fh, m, ext):
    win = T.LocalWindow(fh, m['data_offset'], m['size'])
    with Opened(win, ext) as f:
        g = stacks.groups_row_col(s3io.H5DS(NonCoord(f)))
    dig, nb, nseg = win.digest()
    g.update(member_sha256=dig, member_bytes_read=nb)
    return g

def lcc_rowcol(lat, lon, a):
    import pyproj
    p = pyproj.Proj('+proj=lcc +lat_1=%r +lat_2=%r +lat_0=%r +lon_0=%r +x_0=0 +y_0=0 +a=%r +b=%r +units=m +no_defs' % (a['P_ALP'], a['P_BET'], a['YCENT'], a['P_GAM'], C.R_EARTH, C.R_EARTH))
    x, y = p(lon, lat)
    c0 = np.floor((x - a['XORIG']) / a['XCELL']).astype(np.int64)
    r0 = np.floor((y - a['YORIG']) / a['YCELL']).astype(np.int64)
    return (r0, c0, x, y)

def rowcol_check(g, a):
    r0, c0, x, y = lcc_rowcol(g['latitude'], g['longitude'], a)
    nr, nc = (int(a['NROWS']), int(a['NCOLS']))
    inside = (r0 >= 0) & (r0 < nr) & (c0 >= 0) & (c0 < nc)
    out = {'n_stacks': int(r0.size), 'n_outside_by_latlon': int((~inside).sum())}
    for base in (1, 0):
        ok = (g['row'] == r0 + base) & (g['col'] == c0 + base)
        out['agree_base%d' % base] = int((ok & inside).sum())
    base = 1 if out['agree_base1'] >= out['agree_base0'] else 0
    ok = (g['row'] == r0 + base) & (g['col'] == c0 + base)
    out['base'] = base
    out['agree'] = int((ok & inside).sum())
    out['disagree'] = int((~ok & inside).sum())
    out['file_rowcol_outside_grid'] = int(((g['row'] - base < 0) | (g['row'] - base >= nr) | (g['col'] - base < 0) | (g['col'] - base >= nc)).sum())
    bad = ~ok & inside
    if bad.any():
        dr = np.abs(g['row'][bad] - (r0[bad] + base))
        dc = np.abs(g['col'][bad] - (c0[bad] + base))
        out['disagree_max_cells'] = int(max(dr.max(), dc.max()))
        out['disagree_within_1_cell'] = int(((dr <= 1) & (dc <= 1)).sum())
    return out

def cell_sums(res, g, base, p1):
    grids, info = ({}, None)
    for sp in C.PT_SPECIES:
        grid, info = stacks.sum_to_cells(res['means'][sp], g['row'], g['col'], C.GRID[0], C.GRID[1], base=base)
        grids[sp] = grid.astype(np.float32)
    if p1:
        tot = res['means']['NO'] + res['means']['NO2'] + res['means']['HONO']
        grid, _ = stacks.sum_to_cells(tot, g['row'], g['col'], C.GRID[0], C.GRID[1], base=base)
        grids['P1_NOX'] = grid.astype(np.float32)
    return (grids, info)

def write_npz(path, arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    part = path + '.partial'
    with open(part, 'wb') as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(part, path)
    sha, n = C.sha256_file(path)
    return (n, sha)
