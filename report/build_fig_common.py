"""Shared helpers of the figure scripts: the style setup, the reading of evaluation outputs after their sha256 is checked, the text checks, color measures, and the layout of keys, ticks and blocks."""
import itertools
import json
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import build_style as S
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.text import Text
REF_LW = 0.8
REF_DASH = (0, (4.0, 2.0))
SERIES_LW = 0.75
DASH = {'M': (0, (3.0, 1.5)), 'P': (0, (1.2, 1.0))}
MARGIN_IN = 0.08
MIN_PT = 7.0

class SealedError(RuntimeError):
    pass
_DONE = None

def _done():
    global _DONE
    if _DONE is None:
        p = os.path.join(S.CV, 'EVAL_DONE.json')
        with open(p, encoding='utf-8') as f:
            _DONE = json.load(f)['files']
    return _DONE

def sealed(name):
    p = os.path.join(S.CV, name)
    d = _done()
    if name not in d:
        raise SealedError('%s: not listed in EVAL_DONE.json files' % name)
    if not os.path.exists(p):
        raise SealedError('%s: missing from results/evaluation_test_years/' % name)
    h = S.sha256_file(p)
    if h != d[name]['sha256']:
        raise SealedError('%s: sha256 differs from EVAL_DONE.json' % name)
    if os.path.getsize(p) != d[name]['bytes']:
        raise SealedError('%s: bytes differ from EVAL_DONE.json' % name)
    return p

def rows(name):
    with open(sealed(name), encoding='utf-8') as f:
        return json.load(f)['rows']

def setup():
    import matplotlib._mathtext as MT
    MT.SHRINK_FACTOR = 7.0 / 8.0
    S.apply()
    plt.rcParams.update({'mathtext.fontset': 'custom', 'mathtext.rm': S.FAMILY, 'mathtext.sf': S.FAMILY, 'mathtext.it': S.FAMILY + ':italic', 'mathtext.bf': S.FAMILY + ':bold', 'axes.unicode_minus': True})

def _texts(fig):
    out = []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        if t.axes is not None and (not t.axes.get_visible()):
            continue
        out.append(t)
    return out

def text_check(fig):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.get_size_inches()
    k = 72.0 / fig.dpi
    items = []
    for t in _texts(fig):
        bb = t.get_window_extent(r)
        items.append({'text': t.get_text(), 'size_pt': float(t.get_fontsize()), 'box_pt': [bb.x0 * k, bb.y0 * k, bb.x1 * k, bb.y1 * k], 'has_subscript': '$_' in t.get_text(), 'gid': t.get_gid() or ''})
    small = [i for i in items if i['size_pt'] < MIN_PT - 1e-09]
    subs = sorted(set((i['text'] for i in items if i['has_subscript'])))
    over = []
    near = []
    for a, b in itertools.combinations(items, 2):
        ax0, ay0, ax1, ay1 = a['box_pt']
        bx0, by0, bx1, by1 = b['box_pt']
        w = min(ax1, bx1) - max(ax0, bx0)
        h = min(ay1, by1) - max(ay0, by0)
        if w > 0.05 and h > 0.05:
            over.append([a['text'], b['text'], round(w, 2), round(h, 2)])
        elif not (a['gid'] and a['gid'] == b['gid']):
            dist = max(-w, -h, 0.0) if w <= 0 or h <= 0 else 0.0
            if w <= 0 and h <= 0:
                dist = (w * w + h * h) ** 0.5
            if dist < 4.0:
                near.append([a['text'], b['text'], round(dist, 2)])
    m = MARGIN_IN * 72.0
    outside = [i['text'] for i in items if i['box_pt'][0] < m or i['box_pt'][1] < m or i['box_pt'][2] > W * 72.0 - m or (i['box_pt'][3] > H * 72.0 - m)]
    res = {'n_texts': len(items), 'min_size_pt': min((i['size_pt'] for i in items)), 'below_7pt': [i['text'] for i in small], 'mathtext_subscripts_at_0.7': subs, 'overlaps': over, 'within_0.08in_of_edge': outside, 'text_blocks_nearer_than_4pt': near, 'figure_in': [float(W), float(H)], 'pass': not small and (not over) and (not outside)}
    return res

def tag(ax, letter, name=None, gap_pt=None):
    from matplotlib.transforms import offset_copy
    if gap_pt is None:
        gap_pt = S.TITLE_GAP_PT
    fig = ax.figure
    base = offset_copy(ax.transAxes, fig=fig, x=0.0, y=gap_pt, units='points')
    t = ax.text(0.0, 1.0, '(%s)' % letter, transform=base, ha='left', va='bottom', fontsize=8, fontweight='bold', color=S.INK, gid='tagline:%d' % id(ax))
    if name:
        r = fig.canvas.get_renderer()
        w = t.get_window_extent(r).width * 72.0 / fig.dpi
        tr = offset_copy(ax.transAxes, fig=fig, x=w + 2.5, y=gap_pt, units='points')
        ax.text(0.0, 1.0, name, transform=tr, ha='left', va='bottom', fontsize=8, color=S.INK, gid='tagline:%d' % id(ax))
    return t

def block_title(fig, axes, text, axis='y', side='left', gap_pt=None, fontsize=None):
    if gap_pt is None:
        gap_pt = S.AXIS_LABELPAD_PT
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bb = [a.get_window_extent(r) for a in axes]
    x0, x1 = (min((b.x0 for b in bb)), max((b.x1 for b in bb)))
    y0, y1 = (min((b.y0 for b in bb)), max((b.y1 for b in bb)))
    g = gap_pt * fig.dpi / 72.0
    inv = fig.transFigure.inverted()
    fs = fontsize or S.FS_LABEL
    if axis == 'y':
        tl = [t.get_window_extent(r) for a in axes for t in a.get_yticklabels(which='both') if t.get_visible() and t.get_text().strip()]
        if side == 'left':
            edge = min([b.x0 for b in tl if b.x1 <= x0 + 1] + [x0]) - g
            ha = 'right'
        else:
            edge = max([b.x1 for b in tl if b.x0 >= x1 - 1] + [x1]) + g
            ha = 'left'
        fx, fy = inv.transform((edge, 0.5 * (y0 + y1)))
        t = fig.text(fx, fy, text, rotation=90, ha=ha, va='center', fontsize=fs, color=S.INK, gid='axistitle:y:' + side)
        _reg(fig, '_house_axistitles', (t, list(axes), 'y', side))
        return t
    tl = [t.get_window_extent(r) for a in axes for t in a.get_xticklabels(which='both') if t.get_visible() and t.get_text().strip()]
    edge = min([b.y0 for b in tl if b.y1 <= y0 + 1] + [y0]) - g
    fx, fy = inv.transform((0.5 * (x0 + x1), edge))
    t = fig.text(fx, fy, text, ha='center', va='top', fontsize=fs, color=S.INK, gid='axistitle:x:bottom')
    _reg(fig, '_house_axistitles', (t, list(axes), 'x', 'bottom'))
    return t

def _reg(fig, attr, item):
    lst = getattr(fig, attr, None)
    if lst is None:
        lst = []
        setattr(fig, attr, lst)
    lst.append(item)

def gaps_report(fig, pairs):
    fig.canvas.draw()
    return [[n, round(gap_pt(fig, a, b), 2)] for n, a, b in pairs]

def gap_pt(fig, a, b, axis='y'):
    from matplotlib.axes import Axes
    r = fig.canvas.get_renderer()
    k = 72.0 / fig.dpi

    def box(o):
        if isinstance(o, Axes):
            return o.get_tightbbox(r)
        return o.get_window_extent(r)
    A = box(a)
    B = box(b)
    if axis == 'y':
        return (A.y0 - B.y1) * k
    return (B.x0 - A.x1) * k
_D65 = np.array([0.95047, 1.0, 1.08883])
_M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.072175], [0.0193339, 0.119192, 0.9503041]])
CVD = {'protan': np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]]), 'deutan': np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.01182, 0.04294, 0.968881]]), 'tritan': np.array([[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.3039]])}
VIEWS = ('normal', 'greyscale', 'protan', 'deutan', 'tritan')

def hex2rgb(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])

def rgb2hex(c):
    c = np.clip(np.round(np.asarray(c) * 255.0), 0, 255).astype(int)
    return '#%02X%02X%02X' % tuple(c)

def _lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def _delin(c):
    c = np.clip(np.asarray(c, float), 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)

def _lin2lab(lin):
    xyz = np.dot(_M, lin) / _D65
    e, k = (216.0 / 24389.0, 24389.0 / 27.0)
    f = np.where(xyz > e, np.cbrt(xyz), (k * xyz + 16.0) / 116.0)
    return np.array([116.0 * f[1] - 16.0, 500.0 * (f[0] - f[1]), 200.0 * (f[1] - f[2])])

def lch(L, Cc, h):
    r = np.deg2rad(h)
    a, b = (Cc * np.cos(r), Cc * np.sin(r))
    fy = (L + 16.0) / 116.0
    fx, fz = (fy + a / 500.0, fy - b / 200.0)
    e, k = (216.0 / 24389.0, 24389.0 / 27.0)
    xr = fx ** 3 if fx ** 3 > e else (116 * fx - 16) / k
    yr = fy ** 3 if L > k * e else L / k
    zr = fz ** 3 if fz ** 3 > e else (116 * fz - 16) / k
    xyz = np.array([xr, yr, zr]) * _D65
    return rgb2hex(_delin(np.dot(np.linalg.inv(_M), xyz)))

def lab_of(hexc, view='normal'):
    lin = _lin(hex2rgb(hexc))
    if view in CVD:
        lin = np.clip(np.dot(CVD[view], lin), 0.0, 1.0)
    return _lin2lab(lin)

def de2000(l1, l2):
    L1, a1, b1 = l1
    L2, a2, b2 = l2
    C1, C2 = (np.hypot(a1, b1), np.hypot(a2, b2))
    Cb = (C1 + C2) / 2.0
    G = 0.5 * (1 - np.sqrt(Cb ** 7 / (Cb ** 7 + 25.0 ** 7)))
    a1p, a2p = ((1 + G) * a1, (1 + G) * a2)
    C1p, C2p = (np.hypot(a1p, b1), np.hypot(a2p, b2))
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp, dCp, dh = (L2 - L1, C2p - C1p, h2p - h1p)
    if C1p * C2p == 0:
        dh = 0.0
    elif dh > 180:
        dh -= 360
    elif dh < -180:
        dh += 360
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh / 2.0))
    Lbp, Cbp = ((L1 + L2) / 2.0, (C1p + C2p) / 2.0)
    if C1p * C2p == 0:
        hbp = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hbp = (h1p + h2p) / 2.0
    elif h1p + h2p < 360:
        hbp = (h1p + h2p + 360) / 2.0
    else:
        hbp = (h1p + h2p - 360) / 2.0
    T = 1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp)) + 0.32 * np.cos(np.radians(3 * hbp + 6)) - 0.2 * np.cos(np.radians(4 * hbp - 63))
    dth = 30 * np.exp(-((hbp - 275) / 25.0) ** 2)
    Rc = 2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25.0 ** 7))
    SL = 1 + 0.015 * (Lbp - 50) ** 2 / np.sqrt(20 + (Lbp - 50) ** 2)
    SC = 1 + 0.045 * Cbp
    SH = 1 + 0.015 * Cbp * T
    RT = -np.sin(np.radians(2 * dth)) * Rc
    return float(np.sqrt((dLp / SL) ** 2 + (dCp / SC) ** 2 + (dHp / SH) ** 2 + RT * (dCp / SC) * (dHp / SH)))

def ramp_measure(colours, skip_grey_pairs=()):
    out = {}
    for v in VIEWS:
        ds = []
        for i, (c1, c2) in enumerate(zip(colours[:-1], colours[1:])):
            if v == 'greyscale':
                if i in skip_grey_pairs:
                    continue
                ds.append(abs(lab_of(c1)[0] - lab_of(c2)[0]))
            else:
                ds.append(de2000(lab_of(c1, v), lab_of(c2, v)))
        out[v] = round(min(ds), 2) if ds else None
    out['pass'] = all((x is None or x >= 10.0 for x in out.values() if not isinstance(x, bool)))
    return out

def contrast(hexc, bg='#FFFFFF'):

    def lum(h):
        return float(np.dot([0.2126, 0.7152, 0.0722], _lin(hex2rgb(h))))
    l1, l2 = (lum(hexc), lum(bg))
    return (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)

def frame(ax, lw=None, ink=None):
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(S.FRAME_LW if lw is None else lw)
        s.set_edgecolor(S.FRAME_INK if ink is None else ink)
    ax.grid(False)

def house_tag(ax, letter, name=None, gap_pt=None):
    return tag(ax, letter, name, gap_pt)
register_block = S.register_block
layout_check = S.layout_check

def _px(fig, pt):
    return pt * fig.dpi / 72.0

def house_ticks(ax, axis='both'):
    ax.tick_params(axis=axis, which='major', direction='out', length=S.TICK_LEN_PT, width=S.TICK_W_PT, pad=S.TICK_PAD_PT)

def title_gap_over(ax):
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fr = ax.get_window_extent(r)
    top = fr.y1
    parts = [ax.xaxis.label] + list(ax.texts) + [t for t in ax.get_xticklabels(which='both') if t.get_visible()]
    for t in parts:
        g = t.get_gid() or ''
        if g.startswith('tagline:') or not t.get_visible() or (not t.get_text().strip()):
            continue
        b = t.get_window_extent(r)
        if b.y0 >= fr.y1 - 1.0 and min(b.x1, fr.x1) > max(b.x0, fr.x0):
            top = max(top, b.y1)
    return S.TITLE_GAP_PT + (top - fr.y1) * 72.0 / fig.dpi

def _block_box(fig, axes_block, r):
    from matplotlib.transforms import Bbox
    frs = [a.get_window_extent(r) for a in axes_block]
    tight = Bbox.union([a.get_tightbbox(r) for a in axes_block])
    return (min((f.x0 for f in frs)), max((f.x1 for f in frs)), tight)

def house_key(fig, axes_block, handles, labels, ncol=None, side='above', gap_pt=None, rows=None, fontsize=None, title=None, **kw):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fs = S.KEY_FS if fontsize is None else fontsize
    gap = S.KEY_GAP_PT if gap_pt is None else gap_pt
    if rows is None:
        ent = list(zip(handles, labels))
        n = ncol or len(ent)
        rows = [ent[i:i + n] for i in range(0, len(ent), n)]
    rows = [list(rw) for rw in rows if rw]
    segs = []
    for rw in rows:
        if segs and len(segs[-1][0]) == len(rw):
            segs[-1].append(rw)
        else:
            segs.append([rw])
    bx0, bx1, tight = _block_box(fig, axes_block, r)
    cx = 0.5 * (bx0 + bx1)
    inv = fig.transFigure.inverted()
    rowgap = _px(fig, S.KEY_ROWSPACE_EM * fs)
    opts = dict(frameon=False, fontsize=fs, borderpad=0.0, borderaxespad=0.0, handlelength=S.KEY_HANDLELENGTH_EM, handletextpad=S.KEY_TEXTPAD_EM, columnspacing=S.KEY_COLSPACE_EM, labelspacing=S.KEY_ROWSPACE_EM, title_fontsize=fs)
    opts.update(kw)
    legs = []
    order = range(len(segs)) if side == 'below' else range(len(segs) - 1, -1, -1)
    y = tight.y0 - _px(fig, gap) if side == 'below' else tight.y1 + _px(fig, gap)
    for i in order:
        seg = segs[i]
        m = len(seg[0])
        ent = [seg[rr][c] for c in range(m) for rr in range(len(seg))]
        loc = 'upper center' if side == 'below' else 'lower center'
        fx, fy = inv.transform((cx, y))
        lg = fig.legend([e[0] for e in ent], [e[1] for e in ent], loc=loc, bbox_to_anchor=(fx, fy), bbox_transform=fig.transFigure, ncol=m, title=title if i == 0 else None, **opts)
        fig.canvas.draw()
        b = lg.get_window_extent(fig.canvas.get_renderer())
        y = b.y0 - rowgap if side == 'below' else b.y1 + rowgap
        legs.append(lg)
    if side != 'below':
        legs = legs[::-1]
    _reg(fig, '_house_keys', {'legends': legs, 'axes': list(axes_block), 'side': side})
    return legs

def key_rect(fig, axes_block, width_in, height_in, side='below', gap_pt=None, title=True, lower_title=False, ticks_below=True):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    gap = S.KEY_GAP_PT if gap_pt is None else gap_pt
    bx0, bx1, tight = _block_box(fig, axes_block, r)
    W, H = fig.get_size_inches()
    t = fig.text(0, 0, 'Hg0', fontsize=S.KEY_FS)
    th = t.get_window_extent(r).height / fig.dpi
    t.set_fontsize(S.FS_TICK)
    lh = t.get_window_extent(r).height / fig.dpi
    t.remove()
    title_h = th + S.BAR_TITLE_GAP_PT / 72.0 if title else 0.0
    ticks_h = (S.TICK_LEN_PT + S.TICK_PAD_PT) / 72.0 + lh if ticks_below else 0.0
    low_h = th + S.BAR_TITLE_GAP_PT / 72.0 if lower_title else 0.0
    x0 = 0.5 * (bx0 + bx1) / fig.dpi - width_in / 2.0
    if side == 'below':
        top = tight.y0 / fig.dpi - gap / 72.0 - title_h
        y0 = top - height_in
    else:
        y0 = tight.y1 / fig.dpi + gap / 72.0 + ticks_h + low_h
    return [x0 / W, y0 / H, width_in / W, height_in / H]

def bar_key(fig, cax, title=None, lower_title=None, gap_pt=None, fontsize=None, bar=None):
    if isinstance(cax, (list, tuple)):
        cax = fig.add_axes(list(cax))
        frame(cax)
        cax.set_yticks([])
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    g = _px(fig, S.BAR_TITLE_GAP_PT if gap_pt is None else gap_pt)
    fs = S.KEY_FS if fontsize is None else fontsize
    cb = getattr(cax, '_colorbar', None)
    if bar is None and cb is not None:
        bar = cb.outline
    bb = (bar if bar is not None else cax).get_window_extent(r)
    fr = cax.get_window_extent(r)
    top, bot = (max(bb.y1, fr.y1), min(bb.y0, fr.y0))
    for tk in cax.xaxis.get_major_ticks():
        for lab, up in ((tk.label1, False), (tk.label2, True)):
            if lab.get_visible() and lab.get_text().strip():
                lb = lab.get_window_extent(r)
                if up:
                    top = max(top, lb.y1)
                else:
                    bot = min(bot, lb.y0)
    for lab in (cax.xaxis.label,):
        if lab.get_visible() and lab.get_text().strip():
            lb = lab.get_window_extent(r)
            if cax.xaxis.get_label_position() == 'top':
                top = max(top, lb.y1)
            else:
                bot = min(bot, lb.y0)
    inv = fig.transFigure.inverted()
    cx = 0.5 * (bb.x0 + bb.x1)
    out = {'cax': cax, 'title': None, 'lower_title': None}
    if title:
        fx, fy = inv.transform((cx, top + g))
        out['title'] = fig.text(fx, fy, title, ha='center', va='bottom', fontsize=fs, color=S.INK, gid='bartitle')
    if lower_title:
        fx, fy = inv.transform((cx, bot - g))
        out['lower_title'] = fig.text(fx, fy, lower_title, ha='center', va='top', fontsize=fs, color=S.INK, gid='bartitle:lower')
    _reg(fig, '_house_bars', {'cax': cax, 'bar': bar, 'title': out['title'], 'lower_title': out['lower_title']})
    return out

def stepped_bar_key(fig, cax, colours, bounds, title=None, lower_title=None, open_low=True, open_high=True, ticklabels=None, fontsize=None):
    import matplotlib as mpl
    from matplotlib.colors import BoundaryNorm, ListedColormap
    if isinstance(cax, (list, tuple)):
        cax = fig.add_axes(list(cax))
    lo = 1 if open_low else 0
    hi = len(colours) - (1 if open_high else 0)
    inner = list(colours[lo:hi])
    if len(bounds) != len(inner) + 1:
        raise ValueError('stepped_bar_key: %d bounds for %d inner classes' % (len(bounds), len(inner)))
    cmap = ListedColormap(inner).with_extremes(under=colours[0] if open_low else None, over=colours[-1] if open_high else None)
    norm = BoundaryNorm(list(bounds), cmap.N)
    ext = {(True, True): 'both', (True, False): 'min', (False, True): 'max', (False, False): 'neither'}[bool(open_low), bool(open_high)]
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal', extend=ext, ticks=list(bounds), spacing='uniform', extendfrac='auto', drawedges=True)
    cb.outline.set_edgecolor(S.FRAME_INK)
    cb.outline.set_linewidth(S.FRAME_LW)
    cb.dividers.set_color(S.FRAME_INK)
    cb.dividers.set_linewidth(S.FRAME_LW)
    if ticklabels is None:
        ticklabels = [('%g' % v).replace('-', S.MINUS) for v in bounds]
    cb.set_ticks(list(bounds), labels=list(ticklabels))
    cax.xaxis.set_ticks_position('bottom')
    cax.tick_params(axis='x', which='major', direction='out', length=S.TICK_LEN_PT, width=S.TICK_W_PT, pad=S.TICK_PAD_PT, labelsize=S.FS_TICK, colors=S.INK2)
    cax.minorticks_off()
    out = bar_key(fig, cax, title=title, lower_title=lower_title, fontsize=fontsize, bar=cb.outline)
    out['colorbar'] = cb
    return out

def fc_check(fig):
    import re as _re
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    res = {'open_frames': [], 'gridlines': [], 'outside': [], 'overlaps': [], 'small': [], 'case': []}
    fb = fig.bbox
    for ax in fig.axes:
        if not ax.get_visible() or not ax.axison:
            continue
        if any((not s.get_visible() for s in ax.spines.values())):
            if ax.get_gid() not in ('noframe', 'key'):
                res['open_frames'].append(ax.get_label() or str(id(ax)))
        for g in ax.get_xgridlines() + ax.get_ygridlines():
            if g.get_visible():
                res['gridlines'].append(ax.get_label() or str(id(ax)))
                break
    texts = _texts(fig)
    boxes = [t.get_window_extent(r) for t in texts]
    allowed_lc = _re.compile('^(ppb|km|ug|e\\.g\\.|i\\.e\\.|n\\b|x\\b|r\\b|\\$|[0-9\\-−+.%(]|[a-z]\\)|[a-z]$)')
    names = ('CMAQ', 'U-Net', 'NO', 'RMSE', 'MAE', 'US', 'EQUATES', 'Northeast', 'January', 'February', 'December', 'ISO', 'Holm', 'R', 'C-')
    for t, b in zip(texts, boxes):
        s = t.get_text().strip()
        if b.x0 < fb.x0 - 0.5 or b.y0 < fb.y0 - 0.5 or b.x1 > fb.x1 + 0.5 or (b.y1 > fb.y1 + 0.5):
            res['outside'].append(s)
        if t.get_fontsize() < MIN_PT - 1e-09:
            res['small'].append(s)
        if s and s[0].isalpha() and s[0].islower() and (not allowed_lc.match(s)):
            res['case'].append(s)
        words = s.split()
        for w in words[1:]:
            if len(w) > 3 and w[0].isupper() and w[1:].islower() and (not any((w.startswith(n) for n in names))) and (w.rstrip(',.;:)') not in CITY_WORDS):
                res['case'].append(s)
                break
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, b = (boxes[i], boxes[j])
            w = min(a.x1, b.x1) - max(a.x0, b.x0)
            h = min(a.y1, b.y1) - max(a.y0, b.y0)
            if w > 0.5 and h > 0.5:
                res['overlaps'].append([texts[i].get_text(), texts[j].get_text()])
    res['pass'] = not any((res[k] for k in ('open_frames', 'gridlines', 'outside', 'overlaps', 'small')))
    res['case_pass'] = not res['case']
    return res
CITY_WORDS = set('New York Los Angeles Chicago Houston Phoenix Philadelphia San Antonio Diego Dallas Austin Northeast Albers Equal Area Conic'.split())

def write_text(stem, caption, alt, long_desc):
    if len(alt) > 125:
        raise ValueError('%s: alt text %d characters' % (stem, len(alt)))
    for s in (caption, alt, long_desc):
        if any((ord(c) > 126 for c in s)):
            raise ValueError('%s: non-ASCII text' % stem)
    os.makedirs(S.FIG, exist_ok=True)
    with open(os.path.join(S.FIG, stem + '_caption.tex'), 'w', encoding='ascii', newline='\n') as f:
        f.write(caption.strip() + '\n')
    with open(os.path.join(S.FIG, stem + '_alt.txt'), 'w', encoding='ascii', newline='\n') as f:
        f.write(alt.strip() + '\n\n' + long_desc.strip() + '\n')

def fmt(x, nd=2):
    s = ('%.' + str(nd) + 'f') % x
    if s.startswith('-'):
        s = '$-$' + s[1:]
    return s

def fmt_plain(x, nd=2):
    s = ('%.' + str(nd) + 'f') % x
    return s.replace('-', 'minus ') if s.startswith('-') else s

def thousands(n):
    return '{:,}'.format(int(n))

def thousands_tex(n):
    return '{:,}'.format(int(n)).replace(',', '{,}')

def months_ticks(year):
    import datetime as DT
    d0 = DT.date(year, 1, 1)
    pos = [(DT.date(year, m, 15) - d0).days + 1 for m in range(1, 13)]
    return (pos, list('JFMAMJJASOND'))

def month_starts(year):
    import datetime as DT
    d0 = DT.date(year, 1, 1)
    return [(DT.date(year, m, 1) - d0).days + 1 for m in range(1, 13)]

def test_days(year):
    with open(os.path.join(S.ROOT, 'results', 'grid', 'testdays_cv.json'), encoding='ascii') as f:
        d = json.load(f)['folds'][str(year)]
    return (np.array(d['indices'], dtype=int) + 1, d['dates'])

def with_gaps(doy, *series):
    doy = np.asarray(doy)
    br = np.where(np.diff(doy) > 1)[0]
    x = np.insert(doy.astype(float), br + 1, np.nan)
    out = [np.insert(np.asarray(s, dtype=float), br + 1, np.nan) for s in series]
    return (x, out)
