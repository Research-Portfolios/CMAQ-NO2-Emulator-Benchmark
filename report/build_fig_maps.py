"""Draws the maps fig_F2, fig_F7 and fig_A_F3: the 2016 mean NO2 of CMAQ and the residual U-Net with both U-Nets' differences from CMAQ, the residual U-Net's seasonal differences, and its yearly differences, with state lines from the NHGIS 2010 state file when it is present."""
import json
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_fig_common as C
import build_mask_stream as MS
import build_style as S
import build_port_fig_A_F5 as PF
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import shapefile
import shapely
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch, Polygon, Rectangle
from matplotlib.path import Path
from matplotlib.transforms import offset_copy
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LW = os.path.join(ROOT, 'results', 'grid', 'LWMASK.npy')
LW_SHA = 'b7718b1a9faba28a235a7d6a53400f6fc4ec3407972b048915f14ee49c55a176'
SHP = os.environ.get('NHGIS_STATE_2010', os.path.join(ROOT, 'build', 'nhgis', 'US_state_2010'))
CITY = os.path.join(ROOT, 'results', 'grid', 'CITY_CELLS.json')
CACHE = os.path.join(ROOT, 'build', '_map_lines.npz')
NR, NC, CELL = (299, 459, 12000.0)
XORIG, YORIG = (-2556000.0, -1728000.0)
LCC = '+proj=lcc +lat_1=33.0 +lat_2=45.0 +lat_0=40.0 +lon_0=-97.0 +x_0=0 +y_0=0 +a=6370000 +b=6370000 +units=m +no_defs'
NON_CONUS = {'Alaska', 'Hawaii', 'Puerto Rico'}
STEPS = [0.1, 0.2, 0.25, 0.5, 1.0, 2.0]
TOL = 4000.0
MIN_AREA = 50000000.0
SEASONS = ['winter', 'spring', 'summer', 'fall']
YEARS = [2015, 2016, 2017, 2018, 2019]
SEASON_FORM = 'code'
C.setup()
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['hatch.linewidth'] = 0.35
with open(os.path.join(ROOT, 'results', 'grid', 'testdays_cv.json'), encoding='utf-8') as f:
    DAYS = {int(y): int(v['n']) for y, v in json.load(f)['folds'].items()}
MASK, PLACE = MS.get_mask()
FLAT = np.flatnonzero(MASK.ravel())
assert FLAT.size == MS.CELLS
with open(CITY, encoding='utf-8') as f:
    CC = json.load(f)
assert CC['proj4'] == LCC, 'grid projection differs'
assert MS.SHA in json.dumps(CC['mask']), 'CITY_CELLS names another mask'
assert 'row 0 the south edge' in CC['row_col']

def stem(c):
    return ''.join((ch for ch in c if ch.isalnum()))
f4m16 = np.load(C.sealed('F4_mean_fold2016_M.npy')).astype(np.float64)
ANCHORS = []
for c in CC['cities']:
    s = np.load(C.sealed('F5_city_fold2016_M_%s.npy' % stem(c['city'])))
    g = np.array([r * NC + k for r, k in c['block_masked']])
    ix = np.searchsorted(FLAT, g)
    assert np.all(FLAT[ix] == g), 'a city cell is not in the mask'
    res = []
    for row in (0, 1):
        a = float(f4m16[row, ix].mean())
        b = float(np.mean(s[row]))
        res.append(abs(a - b) / max(1.0, abs(b)))
    assert max(res) < 1e-06, 'placement fails at %s' % c['city']
    rr = int(np.floor((c['y_m'] - YORIG) / CELL))
    kk = int(np.floor((c['x_m'] - XORIG) / CELL))
    assert [rr, kk] == [c['row'], c['col']], 'grid origin differs'
    ANCHORS.append({'city': c['city'], 'cells': int(len(g)), 'rel_residual_cmaq': res[0], 'rel_residual_model': res[1]})
LWM = np.load(LW)
assert C.S.sha256_file(LW) == LW_SHA and LWM.shape == (NR, NC)
assert np.all(LWM.astype(bool)[MASK]), 'a masked cell is water in LWMASK'
OTHER = LWM.astype(bool) & ~MASK
t1 = pyproj.Transformer.from_crs(pyproj.CRS.from_proj4(LCC), pyproj.CRS.from_proj4('+proj=longlat +a=6370000 +b=6370000 +no_defs'), always_xy=True)
t2 = pyproj.Transformer.from_crs('EPSG:4269', 'EPSG:5070', always_xy=True)
XE, YE = np.meshgrid(XORIG + np.arange(NC + 1) * CELL, YORIG + np.arange(NR + 1) * CELL)
lon, lat = t1.transform(XE, YE)
AX, AY = t2.transform(lon, lat)

def lines_from(geom):
    out = []
    for part in shapely.get_parts(geom):
        if part.geom_type == 'LineString':
            out.append(np.asarray(part.coords))
        elif part.geom_type == 'MultiLineString':
            out.extend((np.asarray(p.coords) for p in part.geoms))
    xs, ys = ([], [])
    for a in out:
        xs.extend(a[:, 0].tolist() + [np.nan])
        ys.extend(a[:, 1].tolist() + [np.nan])
    return np.array([xs, ys])

def state_lines():
    if not os.path.exists(SHP + '.shp'):
        raise SystemExit('The maps need the 2010 state boundary file of IPUMS NHGIS (US_state_2010, free from nhgis.org); set NHGIS_STATE_2010 to its path without the .shp suffix, or place it at build/nhgis/US_state_2010.shp (see README.md).')
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        if str(z['shp_sha']) == C.S.sha256_file(SHP + '.shp') and 'tol' in z.files and (float(z['tol']) == TOL):
            return (z['inner'], z['outer'])
    with open(SHP + '.prj') as f:
        alb = pyproj.CRS.from_wkt(f.read())
    tt = pyproj.Transformer.from_crs(alb, 'EPSG:5070', always_xy=True)
    r = shapefile.Reader(SHP)
    fields = [f[0] for f in r.fields[1:]]
    nf = [f for f in fields if f.upper() in ('NAME10', 'NAME')][0]
    polys = []
    for sr in r.iterShapeRecords():
        if sr.record[nf] in NON_CONUS:
            continue
        g = shapely.make_valid(shapely.geometry.shape(sr.shape.__geo_interface__))
        g = shapely.transform(g, lambda xy: np.column_stack(tt.transform(xy[:, 0], xy[:, 1])))
        keep = []
        for p in shapely.get_parts(g):
            if p.geom_type != 'Polygon' or p.area < MIN_AREA:
                continue
            holes = [r for r in p.interiors if shapely.Polygon(r).area >= MIN_AREA]
            keep.append(shapely.Polygon(p.exterior, holes))
        polys.append(shapely.MultiPolygon(keep) if len(keep) > 1 else keep[0])
    assert len(polys) == 49, 'expected the 48 contiguous states and DC'
    simp = shapely.coverage_simplify(np.array(polys, dtype=object), TOL)
    simp = [shapely.make_valid(g) for g in simp]
    union = shapely.union_all(simp)
    outer = shapely.line_merge(shapely.boundary(union))
    allb = shapely.union_all([shapely.boundary(g) for g in simp])
    inner = shapely.line_merge(shapely.difference(allb, shapely.buffer(outer, 300.0)))
    inner_xy, outer_xy = (lines_from(inner), lines_from(outer))
    np.savez(CACHE, inner=inner_xy, outer=outer_xy, shp_sha=np.array(C.S.sha256_file(SHP + '.shp')), tol=np.array(TOL))
    return (inner_xy, outer_xy)
INNER, OUTER = state_lines()
rr, kk = np.nonzero(MASK)
cx = np.concatenate([AX[rr, kk], AX[rr + 1, kk + 1], AX[rr, kk + 1], AX[rr + 1, kk]])
cy = np.concatenate([AY[rr, kk], AY[rr + 1, kk + 1], AY[rr, kk + 1], AY[rr + 1, kk]])
PAD = 40000.0
EXT = [cx.min() - PAD, cx.max() + PAD, cy.min() - PAD, cy.max() + PAD]
ASPECT = (EXT[3] - EXT[2]) / (EXT[1] - EXT[0])

def grid(values):
    z = np.full(NR * NC, np.nan)
    z[FLAT] = values
    return np.ma.masked_invalid(z.reshape(NR, NC))

def step_for(diffs):
    p99 = float(np.percentile(np.abs(np.concatenate(diffs)), 99))
    for s in STEPS:
        if 3 * s >= p99:
            return (s, p99)
    return (STEPS[-1], p99)

def k4(s):
    b = [-1000000000.0, -3 * s, -2 * s, -s, 0.0, s, 2 * s, 3 * s, 1000000000.0]
    return (ListedColormap(S.K4_NEG + S.K4_POS), b)
LEV_B = [-1000000000.0, 1.0, 2.0, 4.0, 8.0, 16.0, 1000000000.0]
LEV_CMAP = ListedColormap(S.LEVEL_6)
DOT_EVERY = 8
DOT_GREY = '#000000'
DOT_S = 0.22
DOT_S_INSET = 2.0

def dot_cells(values, sign):
    sel = values > 0 if sign > 0 else values < 0
    r_, k_ = np.divmod(FLAT, NC)
    h = DOT_EVERY // 2
    lat = (r_ % DOT_EVERY == 0) & (k_ % DOT_EVERY == 0) | (r_ % DOT_EVERY == h) & (k_ % DOT_EVERY == h)
    return sel & lat
GXC = 0.25 * (AX[:-1, :-1] + AX[1:, :-1] + AX[:-1, 1:] + AX[1:, 1:])
GYC = 0.25 * (AY[:-1, :-1] + AY[1:, :-1] + AY[:-1, 1:] + AY[1:, 1:])

def _us_polygon():
    xs, ys = OUTER
    br = list(np.flatnonzero(np.isnan(xs))) + [len(xs)]
    segs, st = ([], 0)
    for c_ in br:
        if c_ - st >= 2:
            segs.append(shapely.LineString(np.column_stack([xs[st:c_], ys[st:c_]])))
        st = c_ + 1
    return shapely.union_all(shapely.get_parts(shapely.polygonize(segs)))
US_POLY = _us_polygon()
_ro, _ko = np.nonzero(OTHER)
_in = shapely.contains_xy(US_POLY, GXC[_ro, _ko], GYC[_ro, _ko])
NOTEV_R, NOTEV_K = (_ro[_in], _ko[_in])

def _cells_path(rr_, kk_):
    if len(rr_) == 0:
        return None
    q = np.stack([np.column_stack([AX[rr_, kk_], AY[rr_, kk_]]), np.column_stack([AX[rr_, kk_ + 1], AY[rr_, kk_ + 1]]), np.column_stack([AX[rr_ + 1, kk_ + 1], AY[rr_ + 1, kk_ + 1]]), np.column_stack([AX[rr_ + 1, kk_], AY[rr_ + 1, kk_]]), np.column_stack([AX[rr_, kk_], AY[rr_, kk_]])], axis=1)
    u = shapely.union_all(shapely.polygons(q))
    verts, codes = ([], [])
    for p in shapely.get_parts(u):
        if p.geom_type != 'Polygon':
            continue
        for ring in [p.exterior] + list(p.interiors):
            a = np.asarray(ring.coords)
            verts.extend(a.tolist())
            codes.extend([Path.MOVETO] + [Path.LINETO] * (len(a) - 2) + [Path.CLOSEPOLY])
    return Path(verts, codes)
NOTEV_PATH = _cells_path(NOTEV_R, NOTEV_K)
HATCH = '//////'
HATCH_INK = '#9A9A9A'
MAPKEY_GAP_PT = S.CLEAR_RATIO * S.KEY_COLSPACE_EM * S.KEY_FS

class SwatchSet(object):

    def __init__(self, colours, frame=True):
        self.colours = list(colours)
        self.frame = frame

class DotSwatch(object):

    def __init__(self, fill):
        self.fill = fill

class ShareSwatch(object):
    pass

def _handler_base():
    from matplotlib.legend_handler import HandlerBase
    return HandlerBase

class HandlerSwatches(_handler_base()):

    def create_artists(self, legend, orig, xd, yd, w, h, fs, trans):
        out = [Rectangle((-xd, -yd), w, h, facecolor='none', edgecolor='none', lw=0, transform=trans)]
        n = len(orig.colours)
        for i, c in enumerate(orig.colours):
            out.append(Rectangle((-xd + i * w / n, -yd), w / n, h, facecolor=c, edgecolor='none', lw=0, transform=trans))
        if orig.frame:
            out.append(Rectangle((-xd, -yd), w, h, facecolor='none', edgecolor=S.FRAME_INK, lw=0.4, transform=trans))
        return out

class HandlerDots(_handler_base()):

    def create_artists(self, legend, orig, xd, yd, w, h, fs, trans):
        from matplotlib.lines import Line2D
        r_ = Rectangle((-xd, -yd), w, h, facecolor=orig.fill, edgecolor=S.INK2, lw=0.4, transform=trans)
        xs = [-xd + f * w for f in (0.17, 0.5, 0.83, 0.33, 0.67)]
        ys = [-yd + f * h for f in (0.28, 0.28, 0.28, 0.72, 0.72)]
        d = Line2D(xs, ys, ls='none', marker='o', ms=0.9, mew=0, color=DOT_GREY, transform=trans)
        return [r_, d]

class HandlerShare(_handler_base()):

    def create_artists(self, legend, orig, xd, yd, w, h, fs, trans):
        return [Rectangle((-xd, -yd), w, h, facecolor='none', edgecolor='none', lw=0, transform=trans), Rectangle((-xd, -yd), w / 2, h, facecolor=BAR_BELOW, lw=0, transform=trans), Rectangle((-xd + w / 2, -yd), w / 2, h, facecolor=BAR_ABOVE, lw=0, transform=trans), Rectangle((-xd, -yd), w, h, facecolor='none', edgecolor=S.FRAME_INK, lw=S.FRAME_LW, transform=trans)]
KEY_HANDLERS = {SwatchSet: HandlerSwatches(), DotSwatch: HandlerDots(), ShareSwatch: HandlerShare()}

def draw_map(ax, values, cmap, bounds, dot_sign=None, ext=None, dot_s=DOT_S):
    e = EXT if ext is None else ext
    ax.set_xlim(e[0], e[1])
    ax.set_ylim(e[2], e[3])
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    C.frame(ax)
    n = cmap.N
    cls = np.digitize(values, bounds[1:-1])
    z = np.full(NR * NC, np.nan)
    z[FLAT] = cls
    z = z.reshape(NR, NC)
    z[OTHER] = n
    full = ListedColormap(list(cmap.colors) + [S.LAND_OTHER])
    ax.pcolormesh(AX, AY, np.ma.masked_invalid(z), cmap=full, norm=BoundaryNorm(np.arange(n + 2) - 0.5, n + 1), shading='flat', rasterized=True, linewidth=0, edgecolors='none', antialiased=False, zorder=2)
    if NOTEV_PATH is not None:
        ax.add_patch(PathPatch(NOTEV_PATH, facecolor='none', edgecolor=HATCH_INK, lw=0, hatch=HATCH, hatchcolor=HATCH_INK, zorder=2.5))
    n_dot = 0
    if dot_sign is not None:
        pick = dot_cells(values, dot_sign)
        n_dot = int(pick.sum())
        r_, k_ = np.divmod(FLAT, NC)
        ax.scatter(GXC[r_, k_][pick], GYC[r_, k_][pick], s=dot_s, marker='o', color=DOT_GREY, linewidths=0, zorder=3, rasterized=False)
    ax.plot(INNER[0], INNER[1], color='#FFFFFF', lw=0.4, zorder=4, solid_joinstyle='round', solid_capstyle='round')
    ax.plot(OUTER[0], OUTER[1], color=S.MUTED, lw=0.3, zorder=5, solid_joinstyle='round', solid_capstyle='round')
    return n_dot
FIG_W = 6.5
M = 0.08
GUT = 0.12
TAGH = 0.17
MW = (FIG_W - 2 * M - GUT) / 2
KB_H = 0.13

def axin(fig, x, y, w, h, **kw):
    W, H = fig.get_size_inches()
    return fig.add_axes([x / W, y / H, w / W, h / H], **kw)

def num(x):
    s = '%g' % x
    return s.replace('-', S.MINUS)

def end_num(x):
    return ('%.1f' % x).replace('-', S.MINUS)

def signed(x, nd=3):
    return ('%+.*f' % (nd, x)).replace('-', S.MINUS)

def diff_key(fig, x0, y0, w, s, lo, hi, arms, arms_now=True):
    ax = axin(fig, x0, y0, w, KB_H)
    for i, col in enumerate(S.K4_NEG + S.K4_POS):
        ax.add_patch(Rectangle((i, 0), 1, 1, facecolor=col, edgecolor='#FFFFFF', lw=0.4))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks(range(9))
    ax.set_xticklabels([end_num(lo), num(-3 * s), num(-2 * s), num(-s), '0', num(s), num(2 * s), num(3 * s), end_num(hi)])
    ax.tick_params(axis='x', direction='out', length=S.TICK_LEN_PT, width=S.TICK_W_PT, pad=S.TICK_PAD_PT, labelsize=S.FS_TICK, colors=S.INK2)
    C.frame(ax)
    ax._g1_arms = arms
    if arms_now:
        place_arms(fig, ax)
    return ax

def place_arms(fig, ax):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fr = ax.get_window_extent(r)
    lab_y0 = min((t.get_window_extent(r).y0 for t in ax.get_xticklabels())) / fig.dpi
    ya = lab_y0 - S.BAR_TITLE_GAP_PT / 72.0
    out = []
    for xx, word, ha in ((fr.x0 / fig.dpi, ax._g1_arms[0], 'left'), (fr.x1 / fig.dpi, ax._g1_arms[1], 'right')):
        out.append(ax.text(xx, ya, word, ha=ha, va='top', fontsize=S.FS_TICK, color=S.INK, transform=fig.dpi_scale_trans, clip_on=False))
    return out

def _text_hw(fig, txt, fs):
    r = fig.canvas.get_renderer()
    t = fig.text(0, 0, txt, fontsize=fs)
    b = t.get_window_extent(r)
    t.remove()
    return (b.width / fig.dpi, b.height / fig.dpi)

def bar_key_height(fig):
    _, th = _text_hw(fig, 'Hg0', S.KEY_FS)
    _, lh = _text_hw(fig, 'Hg0', S.FS_TICK)
    return th + S.BAR_TITLE_GAP_PT / 72.0 + KB_H + (S.TICK_LEN_PT + S.TICK_PAD_PT) / 72.0 + lh + S.BAR_TITLE_GAP_PT / 72.0 + lh

def entries_height(fig, n):
    _, th = _text_hw(fig, 'Hg0', S.KEY_FS)
    return n * th + (n - 1) * S.KEY_ROWSPACE_EM * S.KEY_FS / 72.0

def draw_sample(ax, kind, x, yc, w, h, colour=None, sign=1):
    tr = ax.figure.dpi_scale_trans
    kw = dict(transform=tr, clip_on=False)
    if kind == 'bar':
        ax.add_patch(Rectangle((x, yc - h / 2), w / 2, h, facecolor=BAR_BELOW, lw=0, **kw))
        ax.add_patch(Rectangle((x + w / 2, yc - h / 2), w / 2, h, facecolor=BAR_ABOVE, lw=0, **kw))
        ax.add_patch(Rectangle((x, yc - h / 2), w, h, facecolor='none', edgecolor=S.FRAME_INK, lw=S.FRAME_LW, **kw))
    elif kind == 'dots':
        fill = S.K4_POS[0] if sign > 0 else S.K4_NEG[-1]
        ax.add_patch(Rectangle((x, yc - h / 2), w, h, facecolor=fill, edgecolor=S.INK2, lw=0.4, **kw))
        ax.plot([x + f * w for f in (0.17, 0.5, 0.83, 0.33, 0.67)], [yc - h / 2 + f * h for f in (0.28, 0.28, 0.28, 0.72, 0.72)], ls='none', marker='o', ms=0.9, mew=0, color=DOT_GREY, **kw)
    elif kind == 'hatch':
        ax.add_patch(Rectangle((x, yc - h / 2), w, h, facecolor='#FFFFFF', edgecolor=S.INK2, lw=0.4, hatch=HATCH, hatchcolor=HATCH_INK, **kw))
    elif kind in ('disc', 'hollow'):
        mc = S.MODEL_COLOUR['M']
        ax.add_patch(Rectangle((x, yc - h / 2), w, h, facecolor='none', edgecolor='none', lw=0, **kw))
        ax.plot([x + w - 2.1 / 72.0], [yc], marker=S.MODEL_MARKER['M'], ms=4.2, mfc='white' if kind == 'hollow' else mc, mec=mc, mew=0.9, ls='none', **kw)
    elif kind == 'ref':
        ax.add_patch(Rectangle((x, yc - h / 2), w, h, facecolor='none', edgecolor='none', lw=0, **kw))
        xr = x + w - 2.1 / 72.0
        ax.plot([xr, xr], [yc - h / 2, yc + h / 2], color=S.REF_INK, lw=0.6, solid_capstyle='butt', **kw)

def bar_entries_key(fig, top_in, centre_in, bar_w, s, lo, hi, arms, title, entries):
    PT = 1.0 / 72.0
    _, th = _text_hw(fig, 'Hg0', S.KEY_FS)
    tth = _text_hw(fig, title, S.KEY_FS)[1]
    y0 = top_in - tth - S.BAR_TITLE_GAP_PT * PT - KB_H
    bax = diff_key(fig, 0.0, y0, bar_w, s, lo, hi, arms, arms_now=False)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fr = bax.get_window_extent(r)
    labs = [t.get_window_extent(r) for t in bax.get_xticklabels()]
    left = (fr.x0 - min((b.x0 for b in labs))) / fig.dpi
    right = (max((b.x1 for b in labs)) - fr.x1) / fig.dpi
    sw = S.KEY_HANDLELENGTH_EM * S.KEY_FS * PT
    sh = 0.7 * S.KEY_FS * PT
    tp = S.KEY_TEXTPAD_EM * S.KEY_FS * PT
    eg = S.KEY_COLSPACE_EM * S.KEY_FS * PT
    lw_ = max((_text_hw(fig, e[1], S.KEY_FS)[0] for e in entries))
    ew = sw + tp + lw_
    total = left + bar_w + right + eg + ew
    x_start = centre_in - total / 2.0
    bx0 = x_start + left
    W, H = fig.get_size_inches()
    bax.set_position([bx0 / W, y0 / H, bar_w / W, KB_H / H])
    bax.set_gid('key')
    place_arms(fig, bax)
    fig.canvas.draw()
    tls = [(t.get_text(), t.get_window_extent(r)) for t in bax.get_xticklabels()]
    nd = max((len(x.split('.')[1]) for x, _ in tls if '.' in x), default=0)
    for x, b in tls:
        if nd and '.' not in x:
            x = x + '.' + '0' * nd
        bax.text(0.5 * (b.x0 + b.x1) / fig.dpi, b.y1 / fig.dpi, x, ha='center', va='top', fontsize=S.FS_TICK, color=S.INK2, transform=fig.dpi_scale_trans, clip_on=False)
    bax.set_xticklabels([''] * len(tls))
    bk = C.bar_key(fig, bax, title=title)
    fig.canvas.draw()
    tt = bk['title'].get_window_extent(r)
    arm_bot = min((t.get_window_extent(r).y0 for t in bax.texts)) / fig.dpi
    ktop = tt.y1 / fig.dpi
    yc = 0.5 * (ktop + arm_bot)
    n = len(entries)
    pitch = th + S.KEY_ROWSPACE_EM * S.KEY_FS * PT
    xe = bx0 + bar_w + right + eg
    ents = []
    for i, (kind, label, colour, sign) in enumerate(entries):
        yi = yc + (n - 1) / 2.0 * pitch - i * pitch
        draw_sample(bax, kind, xe, yi, sw, sh, sign=sign)
        ents.append(bax.text(xe + sw + tp, yi, label, ha='left', va='center', fontsize=S.KEY_FS, color=colour or S.INK, transform=fig.dpi_scale_trans, clip_on=False))
    fig.canvas.draw()
    eb = [t.get_window_extent(r) for t in ents]
    bot = min([arm_bot] + [b.y0 / fig.dpi for b in eb])
    x1 = max((b.x1 for b in eb)) / fig.dpi
    rec = {'form': 'bar key at the left, entry column at the right, centred on each other vertically', 'block_left_in': round(x_start, 4), 'block_right_in': round(x1, 4), 'block_centre_offset_pt': round((0.5 * (x_start + x1) - centre_in) * 72.0, 2), 'bar_key_height_pt': round((ktop - arm_bot) * 72.0, 2), 'entry_column_height_pt': round((max((b.y1 for b in eb)) - min((b.y0 for b in eb))) * 72.0 / fig.dpi, 2), 'entry_gap_pt': round(eg * 72.0, 2), 'mark_gap_pt': round(tp * 72.0, 2), 'row_pitch_pt': round(pitch * 72.0, 2), 'top_in': round(ktop, 4), 'bottom_in': round(bot, 4)}
    return (bax, rec)

def key_title(fig, x0, y, title, sub=None):
    W, H = fig.get_size_inches()
    if sub:
        fig.text(x0 / W, y / H, sub, ha='left', va='baseline', fontsize=S.FS_TICK, color=S.INK)
        y += 0.14
    fig.text(x0 / W, y / H, title, ha='left', va='baseline', fontsize=S.FS_LABEL, color=S.INK)
ENTRY_SW, ENTRY_SH, ENTRY_GAP = (0.22, 0.11, 0.07)

def entry(fig, x, y, kind, label, sign=1, colour=None, **kw):
    W, H = fig.get_size_inches()
    a = axin(fig, x, y - ENTRY_SH / 2, ENTRY_SW, ENTRY_SH)
    a.set_xlim(0, 1)
    a.set_ylim(0, 1)
    a.set_axis_off()
    if kind == 'line':
        a.plot([0.0, 1.0], [0.5, 0.5], color=S.MODEL_COLOUR[kw['m']], lw=1.1, ls=S.MODEL_LS[kw['m']], solid_capstyle='butt')
    elif kind == 'disc':
        mc = S.MODEL_COLOUR['M']
        a.plot([0.5], [0.5], marker=S.MODEL_MARKER['M'], ms=4.2, mfc='white' if kw.get('hollow') else mc, mec=mc, mew=0.9, ls='none', clip_on=False)
    elif kind == 'dots':
        fill = S.K4_POS[0] if sign > 0 else S.K4_NEG[-1]
        a.add_patch(Rectangle((0, 0), 1, 1, facecolor=fill, edgecolor=S.INK2, lw=0.4))
        a.scatter([0.17, 0.5, 0.83, 0.33, 0.67], [0.28, 0.28, 0.28, 0.72, 0.72], s=0.6, color=DOT_GREY, linewidths=0)
    elif kind == 'hatch':
        a.add_patch(Rectangle((0, 0), 1, 1, facecolor='#FFFFFF', edgecolor=S.INK2, lw=0.4, hatch=HATCH, hatchcolor=HATCH_INK))
    elif kind == 'bar':
        a.add_patch(Rectangle((0, 0), 0.5, 1, facecolor=BAR_BELOW, lw=0))
        a.add_patch(Rectangle((0.5, 0), 0.5, 1, facecolor=BAR_ABOVE, lw=0))
        a.add_patch(Rectangle((0, 0), 1, 1, facecolor='none', edgecolor=S.FRAME_INK, lw=S.FRAME_LW))
    fig.text((x + ENTRY_SW + ENTRY_GAP) / W, y / H, label, ha='left', va='center', fontsize=S.FS_TICK, color=colour or S.INK)

def name_above(ax, text, gap_pt=3.0, size=None):
    tr = offset_copy(ax.transAxes, fig=ax.figure, x=0.0, y=gap_pt, units='points')
    return ax.text(0.0, 1.0, text, transform=tr, ha='left', va='bottom', fontsize=size or S.FS_LABEL, color=S.INK)
_LAND = MASK | LWM.astype(bool)
_LXC, _LYC = (GXC[_LAND], GYC[_LAND])
_LINES = np.concatenate([INNER, OUTER], axis=1)
_LINES = _LINES[:, np.isfinite(_LINES[0])]

def scale_bar(ax, box, km=200.0):
    w, h = (box[1] - box[0], box[3] - box[2])
    L = km * 1000.0
    pad = 0.75 * CELL
    rec = None
    for fx, fy in ((0.6, 0.07), (0.05, 0.07), (0.6, 0.8), (0.05, 0.8), (0.33, 0.07)):
        x0, y0 = (box[0] + fx * w, box[2] + fy * h)
        x1 = x0 + L
        f = (x0 - 0.03 * w - pad, x1 + 0.03 * w + pad, y0 - 0.04 * h - pad, y0 + 0.17 * h + pad)
        nc = int(np.sum((_LXC > f[0]) & (_LXC < f[1]) & (_LYC > f[2]) & (_LYC < f[3])))
        nl = int(np.sum((_LINES[0] > f[0]) & (_LINES[0] < f[1]) & (_LINES[1] > f[2]) & (_LINES[1] < f[3])))
        rec = {'fx': fx, 'fy': fy, 'land_cells_in_field': nc, 'line_vertices_in_field': nl}
        if nc == 0 and nl == 0:
            break
    x0, y0 = (box[0] + rec['fx'] * w, box[2] + rec['fy'] * h)
    x1 = x0 + L
    ax.plot([x0, x1], [y0, y0], color=S.INK, lw=1.0, solid_capstyle='butt', zorder=7)
    for xe in (x0, x1):
        ax.plot([xe, xe], [y0, y0 + 0.025 * h], color=S.INK, lw=0.6, zorder=7)
    ax.text((x0 + x1) / 2.0, y0 + 0.04 * h, '200 km', ha='center', va='bottom', fontsize=S.FS_TICK, color=S.INK, zorder=7)
    return rec

def inset_box():
    nx, ny = t2.transform(-74.006, 40.7128)
    px, py = t2.transform(-75.1652, 39.9526)
    mx, my = ((nx + px) / 2.0, (ny + py) / 2.0)
    return [mx - 320000.0, mx + 320000.0, my - 240000.0, my + 240000.0]
INSET_CITIES = [('New York', -74.006, 40.7128), ('Philadelphia', -75.1652, 39.9526), ('Baltimore', -76.6122, 39.2904), ('Washington', -77.0369, 38.9072)]

def city_names(ax, box, bar_rec):
    import matplotlib.patheffects as pe
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    w, h = (box[1] - box[0], box[3] - box[2])
    pad = 0.75 * CELL
    bx0, by0 = (box[0] + bar_rec['fx'] * w, box[2] + bar_rec['fy'] * h)
    taken = [(bx0 - 0.03 * w - pad, bx0 + 200000.0 + 0.03 * w + pad, by0 - 0.04 * h - pad, by0 + 0.17 * h + pad)]
    pad = 0.4 * CELL
    pts = []
    for name, lo_, la_ in INSET_CITIES:
        x, y = t2.transform(lo_, la_)
        if box[0] < x < box[1] and box[2] < y < box[3]:
            pts.append((name, lo_, la_, float(x), float(y)))
    for _, _, _, x, y in pts:
        taken.append((x - 1.2 * CELL, x + 1.2 * CELL, y - 1.2 * CELL, y + 1.2 * CELL))
    cand = [(box[0] + fx * w, box[2] + fy * h) for fx in np.linspace(0.04, 0.96, 47) for fy in np.linspace(0.04, 0.96, 47)]
    rec = []
    prev_v0 = None
    pts.sort(key=lambda p_: -p_[4])
    for name, lo_, la_, x, y in pts:
        circ, = ax.plot([x], [y], ls='none', marker='o', ms=3.2, mfc='none', mec=S.INK, mew=0.6, zorder=8)
        t = ax.text(x, y, name, ha='left', va='center', fontsize=S.FS_TICK, color=S.INK, zorder=9)
        t.set_path_effects([pe.withStroke(linewidth=2.0, foreground='#FFFFFF')])
        best = None
        for cx, cy in sorted(cand, key=lambda c: (c[0] - x) ** 2 + (c[1] - y) ** 2):
            t.set_position((cx, cy))
            e = t.get_window_extent(r)
            (u0, v0), (u1, v1) = inv.transform([(e.x0, e.y0), (e.x1, e.y1)])
            f = (u0 - pad, u1 + pad, v0 - pad, v1 + pad)
            if u0 < x + 1.5 * CELL or (prev_v0 is not None and v1 > prev_v0 - 0.5 * CELL):
                continue
            if f[0] < box[0] or f[1] > box[1] or f[2] < box[2] or (f[3] > box[3]):
                continue
            if any((not (f[1] < g[0] or f[0] > g[1] or f[3] < g[2] or (f[2] > g[3])) for g in taken)):
                continue
            nc = int(np.sum((_LXC > f[0]) & (_LXC < f[1]) & (_LYC > f[2]) & (_LYC < f[3])))
            nl = int(np.sum((_LINES[0] > f[0]) & (_LINES[0] < f[1]) & (_LINES[1] > f[2]) & (_LINES[1] < f[3])))
            if nc == 0 and nl == 0:
                best = (cx, cy, (u0, u1, v0, v1))
                break
        if best is None:
            circ.remove()
            t.remove()
            rec.append({'name': name, 'lon': lo_, 'lat': la_, 'placed': False})
            continue
        cx, cy, (u0, u1, v0, v1) = best
        t.set_position((cx, cy))
        taken.append((u0 - pad, u1 + pad, v0 - pad, v1 + pad))
        prev_v0 = v0
        qx, qy = (u0 - 0.3 * CELL, (v0 + v1) / 2.0)
        dist = float(np.hypot(qx - x, qy - y))
        lead = None
        if dist > 2.0 * CELL:
            k = 1.0 - 1.0 * CELL / dist
            ax.plot([x + (qx - x) * (1.0 * CELL / dist), x + (qx - x) * k], [y + (qy - y) * (1.0 * CELL / dist), y + (qy - y) * k], color=S.INK, lw=0.4, zorder=8, solid_capstyle='butt')
            lead = round(dist / 1000.0, 1)
        rec.append({'name': name, 'lon': lo_, 'lat': la_, 'placed': True, 'label_xy_m': [round(cx, 1), round(cy, 1)], 'leader_km': lead})
    return rec
BAR_BELOW = S.GREYS6[2]
BAR_ABOVE = S.GREYS6[0]

def layout_record(fig, extra=None):
    W, H = fig.get_size_inches()
    r = {'figure_in': [float(W), float(H)], 'map_frame': 'neat line', 'hatch': {'pattern': HATCH, 'colour': HATCH_INK, 'cells_us_land_not_evaluated': int(len(NOTEV_R)), 'cells_row_col': [[int(a_), int(b_)] for a_, b_ in zip(NOTEV_R, NOTEV_K)]}, 'dots': {'lattice': 'rows and columns both multiples of %d, or both %d more than a multiple' % (DOT_EVERY, DOT_EVERY // 2), 'colour': DOT_GREY, 'size_pt2': DOT_S, 'size_pt2_inset': DOT_S_INSET, 'sign': "the rarer sign over the figure's difference maps"}, 'key': "classes equal width, ends at the arrays' min and max"}
    if extra:
        r.update(extra)
    return r

def inches_to_figure(fig):
    from matplotlib.transforms import Affine2D
    W, H = fig.get_size_inches()
    tr = Affine2D().scale(1.0 / W, 1.0 / H) + fig.transFigure
    n = 0
    for a in fig.findobj():
        if getattr(a, '_transform', None) is fig.dpi_scale_trans:
            a.set_transform(tr)
            n += 1
    return n

def finish(fig, stem, plotted, sources, cap, alt, long_desc):
    print('inches_to_figure %s: %d parts' % (stem, inches_to_figure(fig)))
    plotted['fc_check'] = C.fc_check(fig)
    plotted['text_check'] = C.text_check(fig)
    C.write_text(stem, cap, alt, long_desc)
    S.save(fig, stem, plotted, sources)

def tex_s(x):
    return '%g' % x

def notev_count(plotted):
    return len(plotted['layout']['hatch']['cells_row_col'])

def tail_tex(n):
    return 'Hatching marks the %s cells of US land not evaluated, and white marks water and land outside the US; white lines are state boundaries and the gray line the coast and national borders. The maps use an equal-area projection.' % C.thousands(n)

def tail_txt(n):
    return 'Hatching marks the %s cells of US land not evaluated; white lines are state boundaries and a gray line the coast and national borders; water and land outside the US are white.' % C.thousands(n)

def dots_short(sign, where=None):
    return "Dots mark the cells %s CMAQ, the rarer sign over the figure's difference maps." % ('above' if sign > 0 else 'below')

def build_f2():
    src = {m: C.sealed('F4_mean_fold2016_%s.npy' % m) for m in ('M', 'P')}
    a = {m: np.load(p).astype(np.float64) for m, p in src.items()}
    cm = a['M'][0]
    assert np.array_equal(a['P'][0], cm)
    d = {m: a[m][1] - a[m][0] for m in ('M', 'P')}
    for m in ('M', 'P'):
        assert np.allclose(d[m], a[m][2], atol=1e-05)
    s, p99 = step_for([d['P'], d['M']])
    pooled = np.concatenate([d['P'], d['M']])
    npos, nneg = (int((pooled > 0).sum()), int((pooled < 0).sum()))
    sign = 1 if npos < nneg else -1
    within = {m: float(np.mean(np.abs(d[m]) < s)) for m in ('M', 'P')}
    beyond = {m: float(np.mean(np.abs(d[m]) >= 3 * s)) for m in ('M', 'P')}
    means = {'cmaq': float(cm.mean()), 'M': float(a['M'][1].mean()), 'P': float(a['P'][1].mean())}
    lev_share = {k: [float(np.mean((v >= lo) & (v < hi))) for lo, hi in zip(LEV_B[:-1], LEV_B[1:])] for k, v in (('cmaq', cm), ('M', a['M'][1]))}
    kmap, kb = k4(s)
    lo = float(min(d['P'].min(), d['M'].min()))
    hi = float(max(d['P'].max(), d['M'].max()))
    cls_share = {m: np.bincount(np.digitize(d[m], kb[1:-1]), minlength=8) / d[m].size for m in ('M', 'P')}
    PT = 1.0 / 72.0
    CW = 2.26
    GUTC = S.GUTTER_PT * PT
    X0 = (FIG_W - (2 * CW + GUTC)) / 2.0
    XR = X0 + CW + GUTC
    MH = CW * ASPECT
    W = FIG_W
    probe = plt.figure(figsize=(W, 2.0))
    _, th = _text_hw(probe, 'Hg0', S.KEY_FS)
    _, lh = _text_hw(probe, 'Hg0', S.FS_TICK)
    plt.close(probe)
    TITLE_H = S.TITLE_GAP_PT * PT + th
    TICKS_H = (S.TICK_LEN_PT + S.TICK_PAD_PT) * PT + lh
    LK = S.KEY_GAP_PT * PT + th + S.BAR_TITLE_GAP_PT * PT + KB_H + TICKS_H
    HH = 0.4
    HG = 0.06
    BOT = TITLE_H + HH + HG + KB_H + TICKS_H + S.BAR_TITLE_GAP_PT * PT + lh
    GR = S.GUTTER_PT * PT
    H = M + TITLE_H + MH + LK + S.BLOCK_GAP_PT * PT + 2 * (TITLE_H + MH) + 2 * GR + BOT + M + 0.04
    fig = plt.figure(figsize=(W, H))
    y_a = H - M - TITLE_H - MH
    dots = {}
    ax_a = axin(fig, X0, y_a, CW, MH)
    draw_map(ax_a, cm, LEV_CMAP, LEV_B)
    C.tag(ax_a, 'a', 'CMAQ, land mean %.2f ppb' % means['cmaq'])
    ax_b = axin(fig, XR, y_a, CW, MH)
    draw_map(ax_b, a['M'][1], LEV_CMAP, LEV_B)
    C.tag(ax_b, 'b', 'Residual U-Net, land mean %.2f ppb' % means['M'])
    lrect = C.key_rect(fig, [ax_a, ax_b], 2.4, KB_H, side='below')
    lk = C.stepped_bar_key(fig, lrect, S.LEVEL_6, [0, 1, 2, 4, 8, 16], title='2016 mean NO$_2$ (ppb)', open_low=False, open_high=True, ticklabels=['0', '1', '2', '4', '8', '16'])
    box0 = inset_box()
    mx, my = (0.5 * (box0[0] + box0[1]), 0.5 * (box0[2] + box0[3]))
    bh_ = box0[3] - box0[2]
    bw_ = bh_ / ASPECT
    box = [mx - bw_ / 2.0, mx + bw_ / 2.0, box0[2], box0[3]]
    bars = {}
    named = {}
    y = y_a - LK - S.BLOCK_GAP_PT * PT - TITLE_H - MH
    rows_ = []
    for m, t, name in (('P', 'c', 'Reference U-Net minus CMAQ'), ('M', 'd', 'Residual U-Net minus CMAQ')):
        ax = axin(fig, X0, y, CW, MH)
        dots[t] = draw_map(ax, d[m], kmap, kb, sign)
        ax.add_patch(Rectangle((box[0], box[2]), box[1] - box[0], box[3] - box[2], fill=False, edgecolor=S.INK, lw=0.6, zorder=6))
        C.tag(ax, t, name)
        ia = axin(fig, XR, y, CW, MH)
        draw_map(ia, d[m], kmap, kb, sign, ext=box, dot_s=DOT_S_INSET)
        bars[t] = scale_bar(ia, box)
        named[t] = city_names(ia, box, bars[t])
        name_above(ia, 'Northeast corridor, box in (%s)' % t)
        rows_.append((ax, ia))
        y -= MH + TITLE_H + GR
    y_hist = M + 0.04 + BOT - TITLE_H - HH
    bar_y = y_hist - HG - KB_H
    kx0, kw = (X0, CW)
    dbar = diff_key(fig, kx0, bar_y, kw, s, lo, hi, ('U-Net below CMAQ', 'U-Net above CMAQ'), arms_now=False)
    hax = axin(fig, kx0, y_hist, kw, HH)
    top = 20.0 * np.ceil(100.0 * max((v.max() for v in cls_share.values())) / 20.0)
    for m in ('M', 'P'):
        hax.plot(np.repeat(np.arange(9), 2)[1:-1], np.repeat(100.0 * cls_share[m], 2), color=S.MODEL_COLOUR[m], lw=1.1, ls=S.MODEL_LS[m], zorder=2)
    hax.set_xlim(0, 8)
    hax.set_ylim(0, 1.05 * top)
    hax.set_yticks(np.arange(0, top + 1, 20))
    hax.set_xticks([])
    hax.set_ylabel('Land cells (%)', fontsize=S.FS_LABEL)
    C.frame(hax)
    ins_d = rows_[1][1]
    hm = [Line2D([], [], color=S.MODEL_COLOUR[q], lw=1.1, ls=S.MODEL_LS[q]) for q in ('M', 'P')]
    hd = DotSwatch(S.K4_POS[0] if sign > 0 else S.K4_NEG[-1])
    hh = Patch(facecolor='#FFFFFF', edgecolor=S.INK2, lw=0.4, hatch=HATCH, hatchcolor=HATCH_INK)
    legs = C.house_key(fig, [ins_d], None, None, ncol=1, side='below', rows=[[(hm[0], 'Residual U-Net')], [(hm[1], 'Reference U-Net')], [(hd, 'U-Net %s CMAQ' % ('above' if sign > 0 else 'below'))], [(hh, 'Land not evaluated')]], handler_map=KEY_HANDLERS)
    lg = legs[0]
    for t, q in zip(lg.get_texts()[:2], ('M', 'P')):
        t.set_color(S.MODEL_TEXT[q])
    fig.canvas.draw()
    rr_ = fig.canvas.get_renderer()
    hfr = hax.get_window_extent(rr_)
    htb = hax.get_tightbbox(rr_)
    bfr = dbar.get_window_extent(rr_)
    blab = [t.get_window_extent(rr_) for t in dbar.get_xticklabels() if t.get_text()]
    dl = max(hfr.x0 - htb.x0, bfr.x0 - min((b.x0 for b in blab))) / fig.dpi
    dr = max(0.0, max((b.x1 for b in blab)) - bfr.x1) / fig.dpi
    wl = lg.get_window_extent(rr_).width / fig.dpi
    eg = S.KEY_COLSPACE_EM * S.KEY_FS * PT
    gl, gr = (X0, XR + CW)
    kw = min(CW, gr - gl - dl - dr - eg - wl)
    tot = dl + kw + dr + eg + wl
    xs = gl + 0.5 * (gr - gl - tot)
    kx0 = xs + dl
    hax.set_position([kx0 / W, y_hist / H, kw / W, HH / H])
    dbar.set_position([kx0 / W, bar_y / H, kw / W, KB_H / H])
    place_arms(fig, dbar)
    fig.canvas.draw()
    hfr = hax.get_window_extent(rr_)
    ht = fig.text(0.5 * (hfr.x0 + hfr.x1) / fig.dpi / W, (hfr.y1 / fig.dpi + S.TITLE_GAP_PT * PT) / H, 'U-Net minus CMAQ (ppb)', ha='center', va='bottom', fontsize=S.KEY_FS, color=S.INK, gid='bartitle:hist')
    fig.canvas.draw()
    ktop = ht.get_window_extent(rr_).y1
    kbot = min((t.get_window_extent(rr_).y0 for t in dbar.texts))
    lgh = lg.get_window_extent(rr_).height
    lg.set_bbox_to_anchor(((kx0 + kw + dr + eg + 0.5 * wl) / W, (0.5 * (ktop + kbot) + 0.5 * lgh) / fig.dpi / H), transform=fig.transFigure)
    lg._beside_block = "Figure 2's bottom row: the key beside the class shares and the bar, the row centred on the grid (grid.bottom_row)"
    fig.canvas.draw()
    lgb = lg.get_window_extent(rr_)
    htb = hax.get_tightbbox(rr_)
    blab = [t.get_window_extent(rr_) for t in dbar.get_xticklabels() if t.get_text()]
    row_l = min(htb.x0, min((b.x0 for b in blab))) / fig.dpi
    row_r = lgb.x1 / fig.dpi
    bottom_rec = {'form': 'one block centred on the grid: class shares over the bar at the left, the key at the right ', 'grid_left_in': round(gl, 4), 'grid_right_in': round(gr, 4), 'block_left_in': round(row_l, 4), 'block_right_in': round(row_r, 4), 'block_centre_offset_pt': round((0.5 * (row_l + row_r) - 0.5 * (gl + gr)) * 72.0, 2), 'inside_grid': bool(row_l >= gl - 0.5 * PT and row_r <= gr + 0.5 * PT), 'class_share_width_in': round(kw, 4), 'entry_gap_pt': round(eg * 72.0, 2), 'key_vcentre_offset_pt': round((0.5 * (lgb.y0 + lgb.y1) - 0.5 * (ktop + kbot)) * 72.0 / fig.dpi, 2)}
    grid_rec = {'column_width_in': CW, 'columns_left_in': [round(X0, 4), round(XR, 4)], 'map_height_in': round(MH, 4), 'inset_box_shape': 'map aspect (height 480 km kept, width 480 km over the aspect)', 'row_gap_pt': S.GUTTER_PT, 'level_key_gap_pt': S.KEY_GAP_PT, 'level_key_to_row_c_pt': S.BLOCK_GAP_PT, 'bottom_row': bottom_rec, 'level_key': 'horizontal stepped bar under (a), (b), centred'}
    plotted = {'mask_sha256': MS.SHA, 'mask_place': PLACE, 'anchors': ANCHORS, 'extent_m': EXT, 'level_bounds': LEV_B[1:-1], 'level_colours': S.LEVEL_6, 'diff_step_s': s, 'diff_p99_abs': p99, 'diff_bounds': kb[1:-1], 'diff_colours': S.K4_NEG + S.K4_POS, 'pooled_pos': npos, 'pooled_neg': nneg, 'dot_sign': sign, 'dotted_cells': dots, 'share_abs_lt_s': within, 'share_abs_ge_3s': beyond, 'means': means, 'level_class_shares': lev_share, 'layout': layout_record(fig, {'inset_box_m': box, 'scale_bar_fields': bars, 'inset_city_names': named, 'keys': ['level under (a), (b), centred', 'difference with class shares under (c), (d)'], 'grid': grid_rec}), 'values_in_mask_order': {'a_cmaq': np.round(cm, 5), 'b_M': np.round(a['M'][1], 5), 'c_P_minus_cmaq': np.round(d['P'], 5), 'd_M_minus_cmaq': np.round(d['M'], 5)}, 'key_ends_ppb': {'min_c_d': lo, 'max_c_d': hi}, 'nv_key_class_shares': {'bounds': kb[1:-1], 'M': cls_share['M'], 'P': cls_share['P'], 'rule': 'share of the land cells in each key class (lower break <= value < upper break)'}}
    cap = "\\textbf{Modeled mean surface NO$_2$ (ppb) over the %d test days of 2016, CMAQ and two emulators, contiguous US land cells of the 12~km grid, 2016 held out.} (a) CMAQ and (b) the residual U-Net in six blue classes of doubling width (breaks 1 to 16~ppb). (c) The reference and (d) the residual U-Net minus CMAQ in steps of %s~ppb, purple below CMAQ and green above; outer classes run beyond $\\pm$%s~ppb. Insets enlarge the boxed Northeast corridor; open circles mark cities. The lines over the color bar give each class's share of land cells. %s %s" % (DAYS[2016], tex_s(s), tex_s(3 * s), dots_short(sign), tail_tex(notev_count(plotted)))
    alt = 'US maps of 2016 mean NO2 from CMAQ and the residual U-Net, and each U-Net minus CMAQ in purple and green, with insets.'
    cm_t, m_t = ('%.2f' % means['cmaq'], '%.2f' % means['M'])
    land_means_txt = '%s ppb for both' % cm_t if cm_t == m_t else '%s ppb for CMAQ and %s ppb for the residual U-Net' % (cm_t, m_t)
    cn = [c['name'] for c in named['c'] if c['placed']]
    city_list = ', '.join(cn[:-1]) + ' and ' + cn[-1]
    long_desc = 'Top row: two maps of the contiguous United States in an equal-area projection, (a) CMAQ and (b) the residual U-Net, mean surface NO2 over the 2016 test days in six classes of one blue hue from pale (under 1 ppb) to dark (16 ppb or more), each title giving the land mean (%s); a horizontal key under them has breaks at 1, 2, 4, 8 and 16 ppb and a pointed open end. Middle and bottom rows: (c) the reference U-Net minus CMAQ and (d) the residual U-Net minus CMAQ, in four purple classes (below CMAQ) and four green classes (above CMAQ), steps of %s ppb; dots mark the cells where the emulator is %s CMAQ. A box on each map marks the Northeast corridor, enlarged at its right with a 200 km scale bar; in each inset small open circles mark %s, each named beside it or joined to its name by a thin line. Under the maps, one key names the classes, with the lowest and highest differences, %s and %s ppb, at its ends and, above it, the share of land cells in each class as a solid vermilion line (residual U-Net) and a dashed blue line (reference U-Net); entries name the dots and the hatch. %s' % (land_means_txt, tex_s(s), 'above' if sign > 0 else 'below', city_list, '%.1f' % lo, '%.1f' % hi, tail_txt(notev_count(plotted)))
    finish(fig, 'fig_F2', plotted, list(src.values()), cap, alt, long_desc)
    return plotted

def share_bar(fig, x, y, w, neg, pos):
    bx = axin(fig, x, y, w, 0.11)
    bx.barh([0], [neg], color=BAR_BELOW, height=1, lw=0)
    bx.barh([0], [pos], left=[neg], color=BAR_ABOVE, height=1, lw=0)
    bx.set_xlim(0, 1)
    bx.set_ylim(-0.5, 0.5)
    bx.set_yticks([])
    bx.set_xticks([0.5])
    bx.set_xticklabels(['50'])
    bx.set_xlabel('Land cells (%)', fontsize=S.FS_LABEL)
    bx.tick_params(axis='x', direction='out', length=S.TICK_LEN_PT, width=S.TICK_W_PT, pad=S.TICK_PAD_PT, labelsize=S.FS_TICK, colors=S.INK2)
    C.frame(bx)
    for xx, word, dx, ha in ((0.0, '%d%% below' % round(neg * 100), -4, 'right'), (1.0, '%d%% above' % round(pos * 100), 4, 'left')):
        tr = offset_copy(bx.transAxes, fig=fig, x=dx, y=0, units='points')
        bx.text(xx, 0.5, word, transform=tr, ha=ha, va='center', fontsize=S.FS_TICK, color=S.INK)
    return bx

def build_f7():
    src = [C.sealed('D7_maps_fold%d_M_s0.npy' % y) for y in YEARS]
    st = np.stack([np.load(p).astype(np.float64) for p in src])
    assert st.shape == (5, 4, MS.CELLS) and np.all(np.isfinite(st))
    v = -st.mean(axis=0)
    s, p99 = step_for([v[k] for k in range(4)])
    npos, nneg = (int((v > 0).sum()), int((v < 0).sum()))
    sign = 1 if npos < nneg else -1
    kmap, kb = k4(s)
    shares = {SEASONS[k]: {'neg': float(np.mean(v[k] < 0)), 'pos': float(np.mean(v[k] > 0)), 'abs_lt_s': float(np.mean(np.abs(v[k]) < s)), 'mean': float(v[k].mean())} for k in range(4)}
    lo, hi = (float(v.min()), float(v.max()))
    PT = 1.0 / 72.0
    GUT7 = S.GUTTER_PT * PT
    MW7 = (FIG_W - 2 * M - GUT7) / 2.0
    MH = MW7 * ASPECT
    W = FIG_W
    probe = plt.figure(figsize=(W, 2.0))
    _, th = _text_hw(probe, 'Hg0', S.KEY_FS)
    _, lh = _text_hw(probe, 'Hg0', S.FS_TICK)
    KBH = max(bar_key_height(probe), entries_height(probe, 3))
    plt.close(probe)
    TITLE_H = S.TITLE_GAP_PT * PT + th
    BARZ = 0.17 + (S.TICK_LEN_PT + S.TICK_PAD_PT) * PT + lh
    BARZ += S.AXIS_LABELPAD_PT * PT + th
    ROWG = S.GUTTER_PT * PT
    H = M + 2 * (TITLE_H + MH + BARZ) + ROWG + MAPKEY_GAP_PT * PT + KBH + M + 0.03
    fig = plt.figure(figsize=(W, H))
    form, frec = PF.season_form(MW7)
    form = SEASON_FORM
    names = [PF.season_label(q, form) for q in SEASONS]
    dots = {}
    sbars = []
    for k in range(4):
        x = M + k % 2 * (MW7 + GUT7)
        y = H - M - TITLE_H - MH - k // 2 * (TITLE_H + MH + BARZ + ROWG)
        ax = axin(fig, x, y, MW7, MH)
        dots[SEASONS[k]] = draw_map(ax, v[k], kmap, kb, sign)
        C.tag(ax, 'abcd'[k], '%s, land mean %s ppb' % (names[k], signed(shares[SEASONS[k]]['mean']).replace('+', '')))
        sbars.append(share_bar(fig, x + 0.62, y - 0.17, MW7 - 1.24, shares[SEASONS[k]]['neg'], shares[SEASONS[k]]['pos']))
    fig.canvas.draw()
    rr_ = fig.canvas.get_renderer()
    sb_bot = min((b.get_tightbbox(rr_).y0 for b in sbars)) / fig.dpi
    bax, krec = bar_entries_key(fig, sb_bot - MAPKEY_GAP_PT * PT, W / 2.0, MW7 - 0.3, s, lo, hi, ('Residual U-Net below CMAQ', 'Residual U-Net above CMAQ'), 'Residual U-Net minus CMAQ, mean NO$_2$ (ppb)', [('bar', 'Land cells below and above CMAQ', None, 1), ('dots', 'Residual U-Net %s CMAQ' % ('above' if sign > 0 else 'below'), None, sign), ('hatch', 'Land not evaluated', None, 1)])
    krec['gap_share_labels_to_key_pt'] = round((sb_bot - krec['top_in']) * 72.0, 2)
    krec['season_form_measured_here'] = frec
    krec['season_form_used'] = form
    plotted = {'mask_sha256': MS.SHA, 'anchors_tested_in': 'fig_F2', 'diff_step_s': s, 'diff_p99_abs': p99, 'diff_bounds': kb[1:-1], 'diff_colours': S.K4_NEG + S.K4_POS, 'pos': npos, 'neg': nneg, 'dot_sign': sign, 'dotted_cells': dots, 'season_shares': shares, 'layout': layout_record(fig, {'keys': ['difference under the grid'], 'key_block': krec, 'share_bars': 'neg and pos of season_shares; mid mark at 50 percent', 'bar_colours': [BAR_BELOW, BAR_ABOVE]}), 'negated': 'stored D7 maps are CMAQ minus model; drawn values are model minus CMAQ', 'values_in_mask_order': {SEASONS[k]: np.round(v[k], 5) for k in range(4)}, 'key_ends_ppb': {'min_4_seasons': lo, 'max_4_seasons': hi}}
    cap = "\\textbf{Mean difference between the residual U-Net and CMAQ in modeled daily mean surface NO$_2$ (ppb) by season, contiguous US land cells of the 12~km grid, test days of the held-out years 2015 to 2019.} (a) Winter (JFD, each year's January, February and December, without December 2019), (b) spring (MAM), (c) summer (JJA) and (d) fall (SON). Each map averages the five yearly seasonal means. Steps of %s~ppb, purple below CMAQ and green above; outer classes run beyond $\\pm$%s~ppb. Each bar gives the shares of land cells below CMAQ (dark gray) and above (light gray). %s %s" % (tex_s(s), tex_s(3 * s), dots_short(sign), tail_tex(notev_count(plotted)))
    alt = 'Four US maps by season: the residual U-Net minus CMAQ, mean of five held-out years, purple below and green above.'
    season_means_txt = ', '.join(('%.3f' % shares[SEASONS[k]]['mean'] for k in range(3))) + ' and %.3f' % shares[SEASONS[3]]['mean']
    season_bars_txt = ', '.join(('%d and %d' % (round(shares[SEASONS[k]]['neg'] * 100), round(shares[SEASONS[k]]['pos'] * 100)) for k in range(3))) + ', %d and %d' % (round(shares[SEASONS[3]]['neg'] * 100), round(shares[SEASONS[3]]['pos'] * 100))
    long_desc = 'Four maps of the contiguous United States in an equal-area projection, two by two: (a) winter (JFD), (b) spring (MAM), (c) summer (JJA), (d) fall (SON), each titled with its land mean (%s ppb). Each cell shows the residual U-Net minus CMAQ, the mean of the five held-out years, in four purple classes (below CMAQ) and four green classes (above CMAQ), steps of %s ppb; dots mark the cells where the emulator is %s CMAQ. Under each map a bar splits its land cells into those below CMAQ (dark gray, share printed at the left) and above it (light gray, share at the right), with a mark at 50 percent (%s percent). One key block under the maps: at the left a color bar titled Residual U-Net minus CMAQ, mean NO2 (ppb), with the lowest and highest values, %s and %s ppb, at its ends and the two sides named under them; at the right three entries name the bars, the dots and the hatch. %s' % (season_means_txt, tex_s(s), 'above' if sign > 0 else 'below', season_bars_txt, '%.1f' % lo, '%.1f' % hi, tail_txt(notev_count(plotted)))
    finish(fig, 'fig_F7', plotted, src, cap, alt, long_desc)
    return plotted

def year_strip(ax, xs, ref, labels, hollow, side_of, above=False, xlim=None):
    n = len(xs)
    ys = np.arange(n)[::-1].astype(float)
    mc = S.MODEL_COLOUR['M']
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.axvline(ref, color=S.REF_INK, lw=0.6, zorder=1)
    for x, y, lab, h in zip(xs, ys, labels, hollow):
        ax.plot([ref, x], [y, y], color=mc, lw=0.9, zorder=2, solid_capstyle='butt')
        ax.plot([x], [y], marker=S.MODEL_MARKER['M'], ms=4.2, mfc='white' if h else mc, mec=mc, mew=0.9, ls='none', zorder=3)
        if lab:
            sd = side_of(x)
            ax.annotate(lab, (x, y), xytext=(5 * sd, 0), textcoords='offset points', ha='left' if sd > 0 else 'right', va='center', fontsize=S.FS_TICK, color=S.INK)
    ax.set_ylim(-0.6, n - (0.15 if above else 0.4))
    if above:
        fig_ = ax.figure
        fig_.canvas.draw()
        r_ = fig_.canvas.get_renderer()
        fb = ax.get_window_extent(r_)
        k2 = 2.0 * fig_.dpi / 72.0
        ref_px = ax.transData.transform((ref, 0))[0]
        ax._g1_moved = []
        for t_ in [t for t in ax.texts if t.get_text().strip()]:
            b_ = t_.get_window_extent(r_)
            if b_.x0 < fb.x0 or b_.x1 > fb.x1:
                xd = t_.xy[0]
                d_px = ax.transData.transform((xd, 0))[0]
                if xd > ref:
                    a_px, ha = (max(d_px, ref_px + k2), 'left')
                    if a_px + b_.width > fb.x1:
                        a_px, ha = (d_px, 'right')
                else:
                    a_px, ha = (min(d_px, ref_px - k2), 'right')
                    if a_px - b_.width < fb.x0:
                        a_px, ha = (d_px, 'left')
                t_.set_position(((a_px - d_px) * 72.0 / fig_.dpi, 4.0))
                t_.set_ha(ha)
                t_.set_va('bottom')
                ax._g1_moved.append((t_, ref))
    ax.set_yticks(ys)
    C.house_ticks(ax, 'both')
    C.frame(ax)
    return ys

def value_column(ax, labels, inset_pt=2.0):
    n = len(labels)
    ys = np.arange(n)[::-1].astype(float)
    tr = offset_copy(ax.get_yaxis_transform(which='grid'), fig=ax.figure, x=inset_pt, y=0, units='points')
    out = []
    for lab, y in zip(labels, ys):
        out.append(ax.text(1.0, y, lab, transform=tr, ha='left', va='center_baseline', fontsize=S.FS_TICK, color=S.INK, gid='valuecol'))
    ax._value_column = out
    return out

def value_column_measure(ax, rr_):
    dpi = ax.figure.dpi
    rows, rights, gaps = ([], [], [])
    disc_px = []
    for ln in ax.lines:
        xd, yd = (ln.get_xdata(), ln.get_ydata())
        if ln.get_marker() not in (None, '', 'None') and len(xd) == 1 and (ln.get_markersize() > 3):
            disc_px.append(ax.transData.transform((xd[0], yd[0])))
    for t_ in ax._value_column:
        b_ = t_.get_window_extent(rr_)
        yc = ax.transData.transform((0.0, t_.get_position()[1]))[1]
        rows.append(round((0.5 * (b_.y0 + b_.y1) - yc) * 72.0 / dpi, 3))
        rights.append(round(b_.x1 * 72.0 / dpi, 3))
        for px, py in disc_px:
            gx = max(b_.x0 - px, px - b_.x1, 0.0)
            gy = max(b_.y0 - py, py - b_.y1, 0.0)
            gaps.append(float(np.hypot(gx, gy)) * 72.0 / dpi - 2.1)
    return {'centre_offset_pt': rows, 'right_edge_spread_pt': round(max(rights) - min(rights), 3), 'least_gap_to_disc_edge_pt': round(min(gaps), 2) if gaps else None}

def build_af3():
    yrs = [2015, 2017, 2018, 2019]
    src = [C.sealed('F4_mean_fold%d_M.npy' % y) for y in yrs]
    a = [np.load(p).astype(np.float64) for p in src]
    d = [x[1] - x[0] for x in a]
    for x, dd in zip(a, d):
        assert np.allclose(dd, x[2], atol=1e-05)
    s, p99 = step_for(d)
    pooled = np.concatenate(d)
    npos, nneg = (int((pooled > 0).sum()), int((pooled < 0).sum()))
    sign = 1 if npos < nneg else -1
    kmap, kb = k4(s)
    shares = {y: {'abs_lt_s': float(np.mean(np.abs(dd) < s)), 'neg': float(np.mean(dd < 0)), 'mean': float(dd.mean())} for y, dd in zip(yrs, d)}
    src16 = C.sealed('F4_mean_fold2016_M.npy')
    d16 = f4m16[1] - f4m16[0]
    assert np.allclose(d16, f4m16[2], atol=1e-05)
    nv4 = {'share_below': float(np.mean(d16 < 0)), 'land_mean': float(d16.mean())}
    lo = float(min(min((dd.min() for dd in d)), d16.min()))
    hi = float(max(max((dd.max() for dd in d)), d16.max()))
    PT = 1.0 / 72.0
    MWA = 2.62
    GUTA = S.GUTTER_PT * PT
    XA = (FIG_W - (2 * MWA + GUTA)) / 2.0
    MH = MWA * ASPECT
    W = FIG_W
    probe = plt.figure(figsize=(W, 2.0))
    _, th = _text_hw(probe, 'Hg0', S.KEY_FS)
    KBH = max(bar_key_height(probe), entries_height(probe, 6))
    plt.close(probe)
    TITLE_H = S.TITLE_GAP_PT * PT + th
    ROWG = S.GUTTER_PT * PT
    H = M + 3 * (TITLE_H + MH) + 2 * ROWG + MAPKEY_GAP_PT * PT + KBH + M + 0.03
    fig = plt.figure(figsize=(W, H))
    order = [(2015, d[0]), (2016, d16), (2017, d[1]), (2018, d[2]), (2019, d[3])]
    dots = {}
    dots16 = None
    maps_ = []
    for i, (y, dd) in enumerate(order):
        x = XA + i % 2 * (MWA + GUTA)
        yb = H - M - TITLE_H - MH - i // 2 * (TITLE_H + MH + ROWG)
        ax = axin(fig, x, yb, MWA, MH)
        maps_.append(ax)
        n_ = draw_map(ax, dd, kmap, kb, sign)
        if y == 2016:
            dots16 = n_
            C.tag(ax, 'abcde'[i], '2016 held out, %d test days, as Figure 2(d)' % DAYS[2016])
        else:
            dots[str(y)] = n_
            C.tag(ax, 'abcde'[i], '%d held out, %d test days' % (y, DAYS[y]))
    ye = H - M - TITLE_H - MH - 2 * (TITLE_H + MH + ROWG)
    cx0 = XA + MWA + GUTA
    sh = 0.486
    YL = 0.56
    SG = S.GUTTER_PT * PT + S.TICK_LEN_PT * PT
    VC = (2.0 + 24.0) * PT
    SW = (MWA - YL - SG - 2.0 * VC) / 2.0
    allyr = [2015, 2016, 2017, 2018, 2019]
    negs = [100.0 * (nv4['share_below'] if y == 2016 else shares[y]['neg']) for y in allyr]
    mns = [nv4['land_mean'] if y == 2016 else shares[y]['mean'] for y in allyr]
    hol = [y == 2016 for y in allyr]
    fa = axin(fig, cx0 + YL, ye + sh, SW, MH - sh)
    assert all((0.0 <= v_ <= 100.0 for v_ in negs)), negs
    year_strip(fa, negs, 50.0, [''] * len(negs), hol, lambda v_: 1 if v_ >= 50 else -1, above=True, xlim=(0, 100))
    value_column(fa, ['%d%%' % round(n_) for n_ in negs])
    fa.set_xticks([0, 50, 100])
    fa.set_yticklabels([str(y) for y in allyr])
    fa.set_xlabel('Land cells (%)', fontsize=S.FS_LABEL)
    fa.set_ylabel('Held-out year', fontsize=S.FS_LABEL)
    C.tag(fa, 'f', 'Below CMAQ')
    ga = axin(fig, cx0 + YL + SW + VC + SG, ye + sh, SW, MH - sh)
    year_strip(ga, mns, 0.0, [''] * len(mns), hol, lambda v_: 1 if v_ > 0 else -1, above=True, xlim=(-0.43, 0.43))
    value_column(ga, [signed(m_) for m_ in mns])
    ga.set_xticks([-0.2, 0.2])
    ga.set_xticklabels([num(-0.2), '0.2'])
    ga.set_yticklabels([])
    ga.set_xlabel('Difference (ppb)', fontsize=S.FS_LABEL)
    C.tag(ga, 'g', 'Land mean')
    fig.canvas.draw()
    rr_ = fig.canvas.get_renderer()
    eb_ = maps_[4].get_window_extent(rr_).y0
    tb_ = min((a_.xaxis.label.get_window_extent(rr_).y0 for a_ in (fa, ga)))
    dy_ = (tb_ - eb_) / fig.dpi
    for a_ in (fa, ga):
        p_ = a_.get_position()
        a_.set_position([p_.x0, p_.y0 - dy_ / H, p_.width, p_.height + dy_ / H])
    sh -= dy_
    fig.canvas.draw()
    rr_ = fig.canvas.get_renderer()
    tb_ = min((a_.xaxis.label.get_window_extent(rr_).y0 for a_ in (fa, ga)))
    print("strips: sh %.4f in; x titles end %.2f pt from map (e)'s bottom" % (sh, (tb_ - eb_) * 72.0 / fig.dpi))
    assert abs(tb_ - eb_) * 72.0 / fig.dpi < 0.5
    grid_bot = min((a.get_tightbbox(rr_).y0 for a in maps_ + [fa, ga])) / fig.dpi
    bax, krec = bar_entries_key(fig, grid_bot - MAPKEY_GAP_PT * PT, W / 2.0 - 0.85 * PT, 2.85, s, lo, hi, ('Residual U-Net below CMAQ', 'Residual U-Net above CMAQ'), 'Residual U-Net minus CMAQ, mean NO$_2$ (ppb)', [('disc', 'Residual U-Net, each held-out year', S.MODEL_TEXT['M'], 1), ('hollow', '2016, as in Figure 2(d)', S.MODEL_TEXT['M'], 1), ('ref', 'Half the land cells', None, 1), ('ref', 'Zero difference', None, 1), ('dots', 'Residual U-Net %s CMAQ' % ('above' if sign > 0 else 'below'), None, sign), ('hatch', 'Land not evaluated', None, 1)])
    krec['gap_grid_to_key_pt'] = round((grid_bot - krec['top_in']) * 72.0, 2)
    fig.canvas.draw()
    inside = []
    for pa, nm in ((fa, 'f'), (ga, 'g')):
        fb = pa.get_window_extent(rr_)
        for t_ in pa.texts:
            if not t_.get_text().strip() or (t_.get_gid() or '').startswith('tagline:') or t_.get_gid() == 'valuecol':
                continue
            b_ = t_.get_window_extent(rr_)
            inside.append({'panel': nm, 'text': t_.get_text(), 'inside': bool(b_.x0 >= fb.x0 and b_.x1 <= fb.x1 and (b_.y0 >= fb.y0) and (b_.y1 <= fb.y1)), 'clear_left_pt': round((b_.x0 - fb.x0) * 72.0 / fig.dpi, 2), 'clear_right_pt': round((fb.x1 - b_.x1) * 72.0 / fig.dpi, 2)})
    print('strip values inside frames:', all((q['inside'] for q in inside)), ascii([(q['panel'], q['text'], q['clear_left_pt'], q['clear_right_pt']) for q in inside]))
    assert all((q['inside'] for q in inside)), inside
    moved = []
    for pa, nm in ((fa, 'f'), (ga, 'g')):
        labs_ = [t for t in pa.texts if t.get_text().strip() and (not (t.get_gid() or '').startswith('tagline:'))]
        for t_, ref_ in getattr(pa, '_g1_moved', []):
            b_ = t_.get_window_extent(rr_)
            rp = pa.transData.transform((ref_, 0))[0]
            dref = (b_.x0 - rp if b_.x0 >= rp else rp - b_.x1) * 72.0 / fig.dpi
            dn = []
            for o in labs_:
                if o is t_:
                    continue
                ob = o.get_window_extent(rr_)
                gx = max(ob.x0 - b_.x1, b_.x0 - ob.x1)
                gy = max(ob.y0 - b_.y1, b_.y0 - ob.y1)
                dn.append(max(gx, gy) * 72.0 / fig.dpi)
            moved.append({'panel': nm, 'text': t_.get_text(), 'ref_clear_pt': round(dref, 2), 'nearest_value_pt': round(min(dn), 2)})
    print('moved values:', ascii(moved))
    assert all((q['ref_clear_pt'] >= 2.0 - 1e-06 and q['nearest_value_pt'] >= 2.0 - 1e-06 for q in moved)), moved
    krec['strip_values_moved_over_discs'] = moved
    krec['strip_values_inside_frames'] = inside
    krec['value_columns'] = {nm: value_column_measure(pa, rr_) for pa, nm in ((fa, 'f'), (ga, 'g'))}
    print('value columns:', ascii(krec['value_columns']))
    krec['form_choice'] = "beside (bar key left, six entries in one column right): the block fits the width and the two parts are about as tall (bar_key_height_pt, entry_column_height_pt); two rows of two below the bar would add a row band to the figure's height"
    plotted = {'mask_sha256': MS.SHA, 'anchors_tested_in': 'fig_F2', 'diff_step_s': s, 'diff_p99_abs': p99, 'diff_bounds': kb[1:-1], 'diff_colours': S.K4_NEG + S.K4_POS, 'pos': npos, 'neg': nneg, 'dot_sign': sign, 'dotted_cells': dots, 'year_shares': shares, 'layout': layout_record(fig, {'keys': ['difference under the grid'], 'key_block': krec, 'panels': '(a) 2015, (b) 2016, (c) 2017, (d) 2018, (e) 2019, (f) share below, (g) land mean'}), 'values_in_mask_order': {str(y): np.round(dd, 5) for y, dd in zip(yrs, d)}, 'key_ends_ppb': {'min_5_years': lo, 'max_5_years': hi}, '2016_share_below_and_land_mean': nv4, 'b_2016_drawn': {'source': os.path.relpath(src16, ROOT), 'source_sha256': C.S.sha256_file(src16), 'rule': 'row 1 minus row 0, as Figure 2 (d); classes and dot sign of this figure', 'dotted_cells': dots16, 'values_in_mask_order': np.round(d16, 5)}}
    cap = "\\textbf{Mean difference between the residual U-Net and CMAQ in modeled daily mean surface NO$_2$ (ppb) over the test days of each held-out year, contiguous US land cells of the 12~km grid.} (a) 2015 (%d test days), (b) 2016 (%d), (c) 2017 (%d), (d) 2018 (%d) and (e) 2019 (%d), each from that year's seed-0 network; (b) is Figure~\\ref{fig:F2}(d) drawn on this figure's classes. Classes step by %s~ppb from zero, purple where the residual U-Net is below CMAQ and green where it is above; the outer classes run beyond $\\pm$%s~ppb to the extremes of the five maps, printed at the key's ends. (f) The share of each map's land cells below CMAQ and (g) each map's land mean, each disc on a stem from the black line, which marks half the cells in (f) and zero difference in (g); the values stand at the right. The open marks are 2016, the year mapped in (b). %s %s" % (DAYS[2015], DAYS[2016], DAYS[2017], DAYS[2018], DAYS[2019], tex_s(s), tex_s(3 * s), dots_short(sign), tail_tex(notev_count(plotted)))
    alt = 'Five US maps of the residual U-Net minus CMAQ, one per held-out year, and two strips of yearly shares and means.'
    long_desc = "Five maps of the contiguous United States in an equal-area projection, two a row: (a) 2015, %d test days, (b) 2016, %d (as Figure 2(d)), (c) 2017, %d, (d) 2018, %d, (e) 2019, %d, each the held-out year's test-day mean of the residual U-Net minus CMAQ, in four purple classes (below CMAQ) and four green classes (above CMAQ), steps of %s ppb; dots mark the cells where the residual U-Net is %s CMAQ. Beside (e), two strips list the years from 2015 at the top to 2019 at the bottom, y axis titled Held-out year: (f) Below CMAQ, x axis Land cells (%%), the percent of land cells below CMAQ, each year a filled vermilion disc on a line from 50 percent, and (g) Land mean, x axis Difference (ppb), the land mean, each a disc on a line from zero, with the values in a column at the right of each strip (%s percent; %s ppb for 2015 to 2019); 2016 is an open disc. One key block under the maps: at the left a color bar titled Residual U-Net minus CMAQ, mean NO2 (ppb), with the lowest and highest values, %s and %s ppb, at its ends and the two sides named under them; at the right six entries name the discs, the two strips' black lines (Half the land cells, Zero difference), the dots and the hatch. %s" % (DAYS[2015], DAYS[2016], DAYS[2017], DAYS[2018], DAYS[2019], tex_s(s), 'above' if sign > 0 else 'below', ', '.join(('%d' % round(n_) for n_ in negs[:4])) + ' and %d' % round(negs[4]), ', '.join(('%+.3f' % m_ for m_ in mns[:4])) + ' and %+.3f' % mns[4], '%.1f' % lo, '%.1f' % hi, tail_txt(notev_count(plotted)))
    finish(fig, 'fig_A_F3', plotted, src, cap, alt, long_desc)
    return plotted
if __name__ == '__main__':
    which = sys.argv[1:] or ['F2', 'F7', 'AF3']
    out = {}
    if 'F2' in which:
        p = build_f2()
        out['F2'] = p
    if 'F7' in which:
        p = build_f7()
        out['F7'] = p
    if 'AF3' in which:
        p = build_af3()
        out['AF3'] = p
    print('hatch cells (US land not evaluated):', len(NOTEV_R))
    for k, v in out.items():
        tc, fc = (v['text_check'], v['fc_check'])
        print(k, 'size', tc['figure_in'], '| text pass', tc['pass'], 'overlaps', tc['overlaps'][:4], 'edge', tc['within_0.08in_of_edge'][:4], 'min pt', tc['min_size_pt'], '| fc pass', fc['pass'], 'case_pass', fc['case_pass'], 'case', ascii(fc['case'][:6]), {q: ascii(fc[q][:3]) for q in ('open_frames', 'outside', 'overlaps', 'small')})
        print('  near<4pt', ascii(tc['text_blocks_nearer_than_4pt'][:8]))
        if 'scale_bar_fields' in v['layout']:
            print('  scale bars', v['layout']['scale_bar_fields'])
        print('  nv', ascii({q: v[q] for q in v if q.startswith('nv') and q != 'nv_key_class_shares'}))
    print('anchors', [(a['city'], '%.1e' % max(a['rel_residual_cmaq'], a['rel_residual_model'])) for a in ANCHORS])
