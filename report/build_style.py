"""The shared style of every figure: one sans-serif family at 7 and 8 pt, the colors, the panel letters, the text and layout checks, and the saving of a vector PDF, a 300 dpi PNG and the plotted values beside each figure."""
import hashlib
import json
import os
import sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(ROOT, 'build')
FIG = os.path.join(RUN, 'figures')
SEALED = os.path.join(ROOT, 'results')
CV = os.path.join(SEALED, 'evaluation_test_years')
LB = os.path.join(SEALED, 'evaluation_held_back')
sys.path.insert(0, os.path.join(ROOT, 'report'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import figure_style as FS
FAMILY = FS.resolve_family()
INK = '#1A1A1A'
INK2 = '#4D4D4D'
MUTED = '#767676'
GRID = '#DCDCDC'
CMAQ = '#1A1A1A'
MODEL_COLOUR = {'M': '#D55E00', 'P': '#0072B2', 'GBT': '#4F4F4F', 'PIX-R': '#848484', 'MET-REG': '#686868', 'GBT-RIDGE': '#C6C6C6', 'CLIM-EM': '#868686', 'CLIM-WD': '#A8A8A8', 'CLIM-C': '#D7D7D7'}
MODEL_MARKER = {'M': 'o', 'P': 's', 'GBT': '^', 'PIX-R': 'v', 'MET-REG': 'D', 'GBT-RIDGE': '<', 'CLIM-EM': 'P', 'CLIM-WD': 'X', 'CLIM-C': 'h'}
MODEL_ORDER = ['M', 'P', 'PIX-R', 'GBT', 'GBT-RIDGE', 'MET-REG', 'CLIM-WD', 'CLIM-EM', 'CLIM-C']
MODEL_LS = {'M': '-', 'P': (0, (4.0, 1.6))}
MODEL_TEXT = {'M': '#B84F00', 'P': '#0072B2'}
CMAQ_LEVEL = '#BDBDBD'
REF_INK = '#1A1A1A'
FRAME_INK = '#4D4D4D'
FRAME_LW = 0.5
FS_LABEL = 8
FS_TICK = 7
FS_TAG = 8
TITLE_GAP_PT = 3.0
AXIS_LABELPAD_PT = 5.0
KEY_FS = FS_LABEL
KEY_HANDLELENGTH_EM = 2.0
KEY_TEXTPAD_EM = 0.5
KEY_COLSPACE_EM = 1.6
KEY_ROWSPACE_EM = 0.5
KEY_GAP_PT = 8.0
BAR_TITLE_GAP_PT = 3.0
GUTTER_PT = 10.0
BLOCK_GAP_PT = 20.0
TICK_LEN_PT = 2.5
TICK_W_PT = 0.6
TICK_PAD_PT = 3.5
CLEAR_RATIO = 2.0
CENTRE_TOL_PT = 1.0
GAP_TOL_PT = 0.6
GUTTER_TOL_PT = 1.0
MINUS = '−'
PLAIN = {'M': 'Residual U-Net', 'P': 'Reference U-Net', 'GBT': 'Gradient-boosted trees', 'GBT-RIDGE': 'Linear model on tree features', 'MET-REG': 'Weather regression', 'CLIM-WD': 'Weekday climatology', 'CLIM-EM': 'Emission-scaled climatology', 'CLIM-C': 'Cell mean', 'PIX-R': 'Per-cell network', 'A1': 'Reference U-Net, squared-error loss', 'A2': 'Residual blocks, lags, log emissions', 'A3': 'As A2, climatology-residual target', 'A4': 'As A2, main training length', 'M-C7': 'Residual U-Net, climatological emissions'}
LEVEL_6 = ['#A8D2E5', '#73A2B5', '#598192', '#45616C', '#2B3E46', '#0F191D']
LEVEL_BREAKS = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0]
K4_NEG = ['#4E4291', '#7362C9', '#9784DF', '#CEBDFE']
K4_POS = ['#A1D39C', '#55A253', '#1F832A', '#045B15']
K4_ANCHOR = '#F2F2F2'
GREYS6 = ['#C7C7C7', '#9A9A9A', '#777777', '#565656', '#303030', '#000000']
LAND_OTHER = '#FFFFFF'
WATER = '#FFFFFF'
CITY_ORDER = ['NewYork', 'LosAngeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'SanAntonio', 'SanDiego', 'Dallas', 'Austin']
CITY_NAME = {'NewYork': 'New York', 'LosAngeles': 'Los Angeles', 'Chicago': 'Chicago', 'Houston': 'Houston', 'Phoenix': 'Phoenix', 'Philadelphia': 'Philadelphia', 'SanAntonio': 'San Antonio', 'SanDiego': 'San Diego', 'Dallas': 'Dallas', 'Austin': 'Austin'}
YEARS = [2015, 2016, 2017, 2018, 2019]

def apply():
    try:
        import matplotlib._mathtext as _MT
        _MT.SHRINK_FACTOR = 7.0 / 8.0
    except Exception:
        pass
    plt.rcParams.update({'font.family': FAMILY, 'mathtext.fontset': 'custom', 'mathtext.rm': FAMILY, 'mathtext.sf': FAMILY, 'mathtext.it': FAMILY + ':italic', 'mathtext.bf': FAMILY + ':bold', 'axes.unicode_minus': True, 'axes.labelpad': AXIS_LABELPAD_PT, 'axes.titlepad': TITLE_GAP_PT, 'legend.fontsize': KEY_FS, 'legend.title_fontsize': KEY_FS, 'legend.handlelength': KEY_HANDLELENGTH_EM, 'legend.handletextpad': KEY_TEXTPAD_EM, 'legend.columnspacing': KEY_COLSPACE_EM, 'legend.labelspacing': KEY_ROWSPACE_EM, 'xtick.major.pad': TICK_PAD_PT, 'ytick.major.pad': TICK_PAD_PT, 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.edgecolor': FRAME_INK, 'axes.linewidth': FRAME_LW, 'axes.grid': False, 'xtick.color': INK2, 'ytick.color': INK2, 'xtick.major.width': TICK_W_PT, 'ytick.major.width': TICK_W_PT, 'xtick.major.size': TICK_LEN_PT, 'ytick.major.size': TICK_LEN_PT, 'axes.labelcolor': INK, 'text.color': INK, 'axes.spines.top': True, 'axes.spines.right': True, 'xtick.direction': 'out', 'ytick.direction': 'out', 'lines.solid_capstyle': 'butt', 'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'figure.dpi': 100, 'savefig.dpi': 600, 'path.simplify': False})

def panel_letter(ax, letter, x=-0.02, y=1.02):
    ax.text(x, y, '(%s)' % letter, transform=ax.transAxes, ha='right', va='bottom', fontsize=8, fontweight='bold', color=INK)

def load_json(folder, name):
    with open(os.path.join(folder, name), encoding='utf-8') as f:
        return json.load(f)

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if not np.isfinite(v) else v
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, float) and (not np.isfinite(o)):
        return None
    return o
PNG_DPI = 600
EDGE_TOL_PX = 0.5

def _shown_ticklabel_ids(fig):
    shown, hidden = ([], set())
    axes = []
    for a in fig.axes:
        axes.append(a)
        axes.extend(getattr(a, 'child_axes', []))
    for a in axes:
        if not a.get_visible() or not a.axison:
            for t in a.findobj(lambda o: hasattr(o, 'get_text')):
                hidden.add(id(t))
            continue
        for name, axis in (('x', a.xaxis), ('y', a.yaxis)):
            lo, hi = sorted(axis.get_view_interval())
            eps = (hi - lo) * 1e-09 + 1e-12
            for kind, ticks in (('major', axis.get_major_ticks()), ('minor', axis.get_minor_ticks())):
                for tk in ticks:
                    loc = tk.get_loc()
                    inview = loc is not None and lo - eps <= loc <= hi + eps
                    for side, lab in ((1, tk.label1), (2, tk.label2)):
                        if not inview or not tk.get_visible():
                            hidden.add(id(lab))
                        elif lab.get_visible() and lab.get_text().strip():
                            shown.append((a, name, kind + str(side), lab))
    return (shown, hidden)

def text_canvas_check(fig, stem=None):
    from matplotlib.text import Text
    res = {'stem': stem, 'test': 'every text, key and axes frame inside the canvas; tick labels on one line, neighbours not overlapping', 'tolerance_px': EDGE_TOL_PX}
    try:
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        fb = fig.bbox
        k = 1.0 / fig.dpi
        res['figure_in'] = [float(v) for v in fig.get_size_inches()]

        def out_of(b):
            return b.x0 < fb.x0 - EDGE_TOL_PX or b.y0 < fb.y0 - EDGE_TOL_PX or b.x1 > fb.x1 + EDGE_TOL_PX or (b.y1 > fb.y1 + EDGE_TOL_PX)

        def box_in(b):
            return [round(b.x0 * k, 4), round(b.y0 * k, 4), round(b.x1 * k, 4), round(b.y1 * k, 4)]
        shown, hidden = _shown_ticklabel_ids(fig)
        texts = []
        for t in fig.findobj(Text):
            if id(t) in hidden or not t.get_visible():
                continue
            if not t.get_text().strip():
                continue
            if t.axes is not None and (not t.axes.get_visible()):
                continue
            texts.append(t)
        outside = []
        for t in texts:
            b = t.get_window_extent(r)
            if out_of(b):
                outside.append({'text': t.get_text(), 'box_in': box_in(b)})
        legends = list(fig.legends)
        for a in fig.axes:
            if a.get_legend() is not None:
                legends.append(a.get_legend())
        leg_out = [box_in(lg.get_window_extent(r)) for lg in legends if lg.get_visible() and out_of(lg.get_window_extent(r))]
        ax_out = []
        for a in fig.axes:
            if a.get_visible() and a.axison and out_of(a.get_window_extent(r)):
                ax_out.append({'axes': a.get_label() or a.get_gid() or '', 'box_in': box_in(a.get_window_extent(r))})
        multi, overl = ([], [])
        groups = {}
        for a, name, kind, lab in shown:
            s = lab.get_text()
            if '\n' in s or lab.get_wrap():
                multi.append({'axis': name, 'text': s})
            groups.setdefault((id(a), name, kind), []).append(lab)
        for (aid, name, kind), labs in groups.items():
            bb = [(lab, lab.get_window_extent(r)) for lab in labs]
            key = (lambda p: p[1].x0) if name == 'x' else lambda p: p[1].y0
            bb.sort(key=key)
            for (la, a1), (lb, b1) in zip(bb, bb[1:]):
                w = min(a1.x1, b1.x1) - max(a1.x0, b1.x0)
                h = min(a1.y1, b1.y1) - max(a1.y0, b1.y0)
                if w > 0.5 and h > 0.5:
                    overl.append([la.get_text(), lb.get_text()])
        res.update({'n_texts': len(texts), 'n_ticklabels': len(shown), 'texts_outside_canvas': outside, 'legends_outside_canvas': leg_out, 'axes_outside_canvas': ax_out, 'multiline_ticklabels': multi, 'ticklabel_overlaps': overl})
        res['pass'] = not (outside or leg_out or ax_out or multi or overl)
    except Exception as e:
        res['error'] = '%s: %s' % (type(e).__name__, e)
        res['pass'] = False
    if stem:
        with open(os.path.join(FIG, stem + '_textcheck.json'), 'w', encoding='ascii', newline='\n') as f:
            json.dump(res, f, indent=1)
            f.write('\n')
    return res

def register_block(fig, name, axes, keys=None):
    blocks = getattr(fig, '_house_blocks', None)
    if blocks is None:
        blocks = []
        fig._house_blocks = blocks
    b = {'name': name, 'axes': list(axes), 'keys': list(keys or [])}
    blocks.append(b)
    return b

def _is_bar_axes(fig, a):
    if getattr(a, '_colorbar', None) is not None:
        return True
    return any((b.get('cax') is a for b in getattr(fig, '_house_bars', [])))

def _key_content_box(a, r):
    from matplotlib.transforms import Bbox
    bbs = []
    for o in list(a.texts) + list(a.lines) + list(a.patches) + list(a.collections):
        if not o.get_visible():
            continue
        if hasattr(o, 'get_text') and (not o.get_text().strip()):
            continue
        try:
            b = o.get_window_extent(r)
        except Exception:
            continue
        if b.width > 0 or b.height > 0:
            bbs.append(b)
    return Bbox.union(bbs) if bbs else a.get_window_extent(r)

def layout_check(fig, blocks=None):
    from matplotlib.legend import Legend
    from matplotlib.text import Text
    res = {'test': 'C3 layout, points', 'house': {'TITLE_GAP_PT': TITLE_GAP_PT, 'AXIS_LABELPAD_PT': AXIS_LABELPAD_PT, 'KEY_GAP_PT': KEY_GAP_PT, 'BAR_TITLE_GAP_PT': BAR_TITLE_GAP_PT, 'GUTTER_PT': GUTTER_PT, 'BLOCK_GAP_PT': BLOCK_GAP_PT, 'TICK_LEN_PT': TICK_LEN_PT, 'TICK_W_PT': TICK_W_PT, 'CLEAR_RATIO': CLEAR_RATIO}}
    fails = []
    try:
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        k = 72.0 / fig.dpi

        def rd(v):
            return None if v is None else round(float(v), 2)

        def xover(a, b):
            return min(a.x1, b.x1) - max(a.x0, b.x0) > 0.5

        def yover(a, b):
            return min(a.y1, b.y1) - max(a.y0, b.y0) > 0.5
        shown, hidden = _shown_ticklabel_ids(fig)
        shown_ids = set((id(t[3]) for t in shown))
        names = {}
        for t in fig.findobj(Text):
            g = t.get_gid() or ''
            if g.startswith('tagline:') and t.axes is not None and t.get_text().strip():
                names.setdefault(id(t.axes), []).append(t.get_text())

        def aname(a):
            if id(a) in names:
                return ' '.join(names[id(a)])[:40]
            return a.get_label() or 'axes%d' % fig.axes.index(a) if a in fig.axes else 'axes'
        elems = []
        for t in fig.findobj(Text):
            if id(t) in hidden or not t.get_visible() or (not t.get_text().strip()):
                continue
            if t.axes is not None and (not t.axes.get_visible()):
                continue
            try:
                elems.append((t, t.get_window_extent(r), t.axes))
            except Exception:
                continue
        legends = list(fig.legends)
        for a in fig.axes:
            if a.get_legend() is not None:
                legends.append(a.get_legend())
        for lg in legends:
            if lg.get_visible():
                elems.append((lg, lg.get_window_extent(r), None))
        vis_axes = [a for a in fig.axes if a.get_visible()]
        for a in vis_axes:
            if a.axison and any((s.get_visible() for s in a.spines.values())):
                elems.append((a, a.get_window_extent(r), a))
        tags = {}
        for t in fig.findobj(Text):
            g = t.get_gid() or ''
            if g.startswith('tagline:') and t.get_visible() and t.get_text().strip() and (t.axes is not None):
                tags.setdefault(g, []).append(t)
        ptitles = []
        for g, ts in tags.items():
            from matplotlib.transforms import Bbox
            ax = ts[0].axes
            tb = Bbox.union([t.get_window_extent(r) for t in ts])
            fr = ax.get_window_extent(r)
            top = fr.y1
            for o, b, own in elems:
                if own is not ax or o is ax or o in ts:
                    continue
                if b.y0 >= fr.y1 - 1.0 and b.y1 <= tb.y0 + 0.5 and xover(b, fr):
                    top = max(top, b.y1)
            gap = (tb.y0 - top) * k
            above = None
            for o, b, own in elems:
                if o in ts or o is ax:
                    continue
                if b.y0 >= tb.y1 - 0.5 and xover(b, tb):
                    d = (b.y0 - tb.y1) * k
                    above = d if above is None else min(above, d)
            rec = {'title': ' '.join((t.get_text() for t in ts))[:60], 'gap_pt': rd(gap), 'gap_to_frame_pt': rd((tb.y0 - fr.y1) * k), 'nearest_above_pt': rd(above)}
            ptitles.append(rec)
            if abs(gap - TITLE_GAP_PT) > GAP_TOL_PT:
                fails.append('title gap %s %.1f' % (rec['title'][:12], gap))
            if above is not None and above < CLEAR_RATIO * max(gap, TITLE_GAP_PT):
                fails.append('title nearer above %s %.1f' % (rec['title'][:12], above))
        res['panel_titles'] = ptitles
        atitles = []

        def far_side(lab, lb, direction):
            best, what = (None, '')
            for o, b, own in elems:
                if o is lab:
                    continue
                if direction == 'down' and b.y1 <= lb.y0 + 0.5 and xover(b, lb):
                    d = (lb.y0 - b.y1) * k
                elif direction == 'up' and b.y0 >= lb.y1 - 0.5 and xover(b, lb):
                    d = (b.y0 - lb.y1) * k
                elif direction == 'left' and b.x1 <= lb.x0 + 0.5 and yover(b, lb):
                    d = (lb.x0 - b.x1) * k
                elif direction == 'right' and b.x0 >= lb.x1 - 0.5 and yover(b, lb):
                    d = (b.x0 - lb.x1) * k
                else:
                    continue
                if best is None or d < best:
                    best = d
                    what = o.get_text()[:20] if hasattr(o, 'get_text') else type(o).__name__
            return (best, what)

        def axis_title(lab, axes_list, axis, side):
            lb = lab.get_window_extent(r)
            tl = [b for a, nm, kind, t in shown for b in [t.get_window_extent(r)] if a in axes_list and nm == axis and kind.endswith('1' if side in ('bottom', 'left') else '2')]
            frs = [a.get_window_extent(r) for a in axes_list]
            if axis == 'x':
                if side == 'bottom':
                    edge = min([b.y0 for b in tl] + [f.y0 for f in frs])
                    gap = (edge - lb.y1) * k
                    other = far_side(lab, lb, 'down')
                else:
                    edge = max([b.y1 for b in tl] + [f.y1 for f in frs])
                    gap = (lb.y0 - edge) * k
                    other = far_side(lab, lb, 'up')
            elif side == 'left':
                edge = min([b.x0 for b in tl] + [f.x0 for f in frs])
                gap = (edge - lb.x1) * k
                other = far_side(lab, lb, 'left')
            else:
                edge = max([b.x1 for b in tl] + [f.x1 for f in frs])
                gap = (lb.x0 - edge) * k
                other = far_side(lab, lb, 'right')
            rec = {'title': lab.get_text()[:60], 'axis': axis, 'side': side, 'gap_pt': rd(gap), 'far_side_pt': rd(other[0]), 'far_side_text': other[1]}
            atitles.append(rec)
            if abs(gap - AXIS_LABELPAD_PT) > GAP_TOL_PT:
                fails.append('axis title gap %s %.1f' % (rec['title'][:12], gap))
            if other[0] is not None and other[0] < CLEAR_RATIO * max(gap, AXIS_LABELPAD_PT):
                fails.append('axis title nearer %s than own ticks: %s %.1f' % (other[1][:12], rec['title'][:12], other[0]))
        for a in vis_axes:
            if not a.axison or _is_bar_axes(fig, a):
                continue
            for axis, ax_ in (('x', a.xaxis), ('y', a.yaxis)):
                lab = ax_.label
                if lab.get_visible() and lab.get_text().strip():
                    axis_title(lab, [a], axis, ax_.get_label_position())
        reg = getattr(fig, '_house_axistitles', [])
        for t, axes_list, axis, side in reg:
            if t.get_visible() and t.get_text().strip():
                axis_title(t, axes_list, axis, side)
        known = set((id(x[0]) for x in reg))
        frames = [(a, a.get_window_extent(r)) for a in vis_axes if a.axison and a.get_gid() != 'key' and (not _is_bar_axes(fig, a))]
        for t in list(fig.texts):
            g = t.get_gid() or ''
            if id(t) in known or not t.get_visible() or (not t.get_text().strip()) or g.startswith(('tagline:', 'bartitle', 'axistitle')):
                continue
            tb = t.get_window_extent(r)
            rot = abs(t.get_rotation() % 180.0 - 90.0) < 1.0
            if rot:
                cands = [(a, f) for a, f in frames if yover(f, tb) and f.x0 >= tb.x1 - 0.5]
                if not cands:
                    continue
                x0 = min((f.x0 for a, f in cands))
                col = [a for a, f in cands if f.x0 <= x0 + 2.0]
                axis, side = ('y', 'left')
                edge = min([lab.get_window_extent(r).x0 for a, nm, kd, lab in shown if a in col and nm == 'y' and kd.endswith('1')] + [x0])
                near = (edge - tb.x1) * k
            else:
                cands = [(a, f) for a, f in frames if xover(f, tb) and f.y0 >= tb.y1 - 0.5]
                if not cands:
                    continue
                y0 = min((f.y0 for a, f in cands))
                col = [a for a, f in cands if f.y0 <= y0 + 2.0]
                axis, side = ('x', 'bottom')
                edge = min([lab.get_window_extent(r).y0 for a, nm, kd, lab in shown if a in col and nm == 'x' and kd.endswith('1')] + [y0])
                near = (edge - tb.y1) * k
            if 0.0 <= near <= 30.0:
                n0 = len(atitles)
                axis_title(t, col, axis, side)
                if len(atitles) > n0:
                    atitles[-1]['found'] = 'fig.text by heuristic; use block_title'
        res['axis_titles'] = atitles
        blocks = blocks if blocks is not None else getattr(fig, '_house_blocks', None)
        hk = getattr(fig, '_house_keys', [])
        if not blocks:
            panels = [a for a in vis_axes if a.axison and a.get_gid() not in ('key',) and (not _is_bar_axes(fig, a))]
            blocks_ = [{'name': 'all', 'axes': panels, 'keys': []}]
        else:
            blocks_ = blocks
        key_of = {}
        for h in hk:
            for lg in h['legends']:
                key_of[id(lg)] = h['axes']
        for b in blocks_:
            for kk in b['keys']:
                key_of.setdefault(id(kk), b['axes'])
        default_axes = blocks_[0]['axes'] if len(blocks_) == 1 else None
        keys = []
        cand = list(fig.legends) + [a for a in vis_axes if a.get_gid() == 'key']
        for b in blocks_:
            for kk in b['keys']:
                if kk not in cand:
                    cand.append(kk)
        for a in vis_axes:
            lg = a.get_legend()
            if lg is not None and lg.get_visible():
                keys.append({'key': 'in panel %s' % aname(a), 'inside_panel': True})
        for kk in cand:
            axes_list = key_of.get(id(kk), default_axes)
            if isinstance(kk, Legend):
                if not kk.get_visible():
                    continue
                kb = kk.get_window_extent(r)
                label = 'legend ' + ' / '.join((t.get_text() for t in kk.get_texts()))[:50]
            else:
                kb = _key_content_box(kk, r)
                label = 'key axes %s' % ' / '.join((t.get_text() for t in kk.texts))[:50]
            rec = {'key': label}
            beside = getattr(kk, '_beside_block', None)
            if beside:
                rec.update({'side': 'beside, one block with its part', 'centre_offset_pt': None, 'note': beside})
            elif axes_list:
                from matplotlib.transforms import Bbox
                frs = [a.get_window_extent(r) for a in axes_list]
                bx0 = min((f.x0 for f in frs))
                bx1 = max((f.x1 for f in frs))
                off = (0.5 * (kb.x0 + kb.x1) - 0.5 * (bx0 + bx1)) * k
                tight = Bbox.union([a.get_tightbbox(r) for a in axes_list])
                if kb.y0 >= tight.y1 - 0.5:
                    g = (kb.y0 - tight.y1) * k
                    rec['side'] = 'above'
                elif kb.y1 <= tight.y0 + 0.5:
                    g = (tight.y0 - kb.y1) * k
                    rec['side'] = 'below'
                else:
                    g = None
                    rec['side'] = 'beside or over'
                rec.update({'centre_offset_pt': rd(off), 'gap_to_block_pt': rd(g)})
                if abs(off) > CENTRE_TOL_PT:
                    fails.append('key off centre %.1f %s' % (off, label[:24]))
                if g is not None and abs(g - KEY_GAP_PT) > GAP_TOL_PT:
                    rec['note'] = 'gap to block not KEY_GAP_PT'
            else:
                rec['centre_offset_pt'] = None
                rec['note'] = 'no block known (register the blocks)'
            if isinstance(kk, Legend):
                hs = getattr(kk, 'legend_handles', None) or getattr(kk, 'legendHandles', [])
                pairs = []
                for h, t in zip(hs, kk.get_texts()):
                    if h is None or not t.get_text().strip():
                        continue
                    try:
                        hb = h.get_window_extent(r)
                    except Exception:
                        continue
                    if hb.width <= 0:
                        continue
                    pairs.append((hb, t.get_window_extent(r)))
                mg = [(tb.x0 - hb.x1) * k for hb, tb in pairs]
                rows = {}
                for hb, tb in pairs:
                    rows.setdefault(round(0.5 * (tb.y0 + tb.y1) * k / 2.0), []).append((hb, tb))
                eg = []
                for row in rows.values():
                    row.sort(key=lambda p: p[0].x0)
                    eg.extend(((b2[0].x0 - b1[1].x1) * k for b1, b2 in zip(row, row[1:])))
                rec.update({'mark_gaps_pt': [rd(v) for v in mg], 'entry_gaps_pt': [rd(v) for v in eg], 'n_rows': len(rows)})
                if mg and max(mg) - min(mg) > GAP_TOL_PT:
                    fails.append('key mark gaps uneven %s' % label[:24])
                if mg and eg and (min(eg) < CLEAR_RATIO * max(mg)):
                    fails.append('key entry gap under %gx mark gap %s' % (CLEAR_RATIO, label[:24]))
                if len(rows) == 1 and eg and (max(eg) - min(eg) > GAP_TOL_PT):
                    fails.append('key entry gaps uneven %s' % label[:24])
            keys.append(rec)
        res['keys'] = keys
        bars = []
        for bk in getattr(fig, '_house_bars', []):
            bar = bk.get('bar') or bk['cax']
            bb = bar.get_window_extent(r)
            cxb = 0.5 * (bb.x0 + bb.x1)
            for role in ('title', 'lower_title'):
                t = bk.get(role)
                if t is None:
                    continue
                tb = t.get_window_extent(r)
                off = (0.5 * (tb.x0 + tb.x1) - cxb) * k
                rec = {'bar_title': t.get_text()[:50], 'role': role, 'centre_offset_pt': rd(off)}
                bars.append(rec)
                if abs(off) > CENTRE_TOL_PT:
                    fails.append('bar title off centre %.1f %s' % (off, rec['bar_title'][:20]))
        res['bars'] = bars
        nomark, offhouse = ([], [])
        for a in vis_axes:
            if not a.axison:
                continue
            framed = any((s.get_visible() for s in a.spines.values()))
            for nm, axis in (('x', a.xaxis), ('y', a.yaxis)):
                lo, hi = sorted(axis.get_view_interval())
                eps = (hi - lo) * 1e-09 + 1e-12
                for side in (1, 2):
                    n_lab = n_mark = 0
                    bad = set()
                    for tk in axis.get_major_ticks():
                        loc = tk.get_loc()
                        if loc is None or not lo - eps <= loc <= hi + eps:
                            continue
                        lab = tk.label1 if side == 1 else tk.label2
                        line = tk.tick1line if side == 1 else tk.tick2line
                        if id(lab) not in shown_ids:
                            continue
                        n_lab += 1
                        if line.get_visible() and line.get_markersize() > 0:
                            n_mark += 1
                            if abs(line.get_markersize() - TICK_LEN_PT) > 0.05:
                                bad.add('len %.2f' % line.get_markersize())
                            if abs(line.get_markeredgewidth() - TICK_W_PT) > 0.05:
                                bad.add('width %.2f' % line.get_markeredgewidth())
                            if getattr(tk, '_tickdir', 'out') != 'out':
                                bad.add('dir %s' % tk._tickdir)
                    where = '%s %s%d' % (aname(a), nm, side)
                    if n_lab and n_mark < n_lab:
                        nomark.append({'axes': where, 'labels': n_lab, 'marks': n_mark, 'framed': framed})
                        if framed:
                            fails.append('labels without ticks %s' % where)
                    if bad:
                        offhouse.append({'axes': where, 'found': sorted(bad), 'framed': framed})
                        if framed:
                            fails.append('ticks off house %s %s' % (where, ','.join(sorted(bad))))
        res['ticks_without_marks'] = nomark
        res['ticks_off_house'] = offhouse
        if blocks:
            from matplotlib.transforms import Bbox
            gut, bgaps, unions = ([], [], [])
            for b in blocks:
                axs = [a for a in b['axes'] if a.get_visible()]
                tb = {id(a): a.get_tightbbox(r) for a in axs}
                fr = {id(a): a.get_window_extent(r) for a in axs}
                for a in axs:
                    fa = fr[id(a)]
                    right = [c for c in axs if c is not a and yover(fr[id(c)], fa) and (fr[id(c)].x0 > fa.x1)]
                    if right:
                        c = min(right, key=lambda c: fr[id(c)].x0)
                        gut.append({'block': b['name'], 'dir': 'across', 'pair': [aname(a), aname(c)], 'gap_pt': rd((tb[id(c)].x0 - tb[id(a)].x1) * k)})
                    below = [c for c in axs if c is not a and xover(fr[id(c)], fa) and (fr[id(c)].y1 < fa.y0)]
                    if below:
                        c = max(below, key=lambda c: fr[id(c)].y1)
                        gut.append({'block': b['name'], 'dir': 'down', 'pair': [aname(a), aname(c)], 'gap_pt': rd((tb[id(a)].y0 - tb[id(c)].y1) * k)})
                if axs:
                    unions.append((b['name'], Bbox.union([tb[id(a)] for a in axs])))
            for g in gut:
                if abs(g['gap_pt'] - GUTTER_PT) > GUTTER_TOL_PT:
                    fails.append('gutter %s %s %.1f' % (g['block'][:12], g['dir'], g['gap_pt']))
            for (n1, u1), (n2, u2) in zip(unions, unions[1:]):
                if xover(u1, u2) and u2.y1 <= u1.y0 + 0.5:
                    g = (u1.y0 - u2.y1) * k
                    bgaps.append({'upper': n1, 'lower': n2, 'gap_pt': rd(g)})
                    if abs(g - BLOCK_GAP_PT) > GUTTER_TOL_PT:
                        fails.append('block gap %s/%s %.1f' % (n1[:10], n2[:10], g))
            res['gutters'] = gut
            res['block_gaps'] = bgaps
            res['blocks'] = [b['name'] for b in blocks]
        else:
            res['blocks'] = None
            res['note'] = 'no blocks registered: keys centred on the union of every framed panel; gutters not tested'
    except Exception as e:
        fails.append('ERROR %s: %s' % (type(e).__name__, e))
    res['fails'] = fails
    res['pass'] = not fails
    return res

def unify_tick_decimals(fig):
    import re
    changes = []
    try:
        fig.canvas.draw()
    except Exception as e:
        return [{'error': repr(e)}]
    num = re.compile('^-?\\d+(\\.\\d+)?$')
    axes = []
    for a in fig.axes:
        axes.append(a)
        axes.extend(getattr(a, 'child_axes', []))
    for a in axes:
        if not a.get_visible() or not a.axison:
            continue
        for name, axis in (('x', a.xaxis), ('y', a.yaxis)):
            if getattr(axis, 'keep_tick_text', False):
                continue
            lo, hi = sorted(axis.get_view_interval())
            eps = (hi - lo) * 1e-09 + 1e-12
            ticks = [tk for tk in axis.get_major_ticks() if tk.get_loc() is not None and tk.get_visible() and (lo - eps <= tk.get_loc() <= hi + eps)]
            labs = [tk.label1.get_text() if tk.label1.get_visible() else tk.label2.get_text() for tk in ticks]
            shown = [(tk, t) for tk, t in zip(ticks, labs) if t.strip()]
            if len(shown) < 2:
                continue
            plain = [t.replace(MINUS, '-').replace('−', '-').replace(',', '') for _, t in shown]
            if not all((num.match(p) for p in plain)):
                continue
            if axis.get_scale() == 'log':
                vals = [abs(float(p)) for p in plain if float(p) != 0]
                if vals and max(vals) / min(vals) >= 10.0:
                    continue
            dec = [len(p.split('.')[1]) if '.' in p else 0 for p in plain]
            nd = max(dec)
            if nd == min(dec):
                continue
            locs = [tk.get_loc() for tk, _ in shown]
            new = []
            for p in plain:
                s = '%.*f' % (nd, float(p))
                if float(s) == 0.0:
                    s = s.lstrip('-')
                new.append(s.replace('-', MINUS))
            old = [t for _, t in shown]
            for (tk, _), s in zip(shown, new):
                tk.label1.set_text(s)
                tk.label2.set_text(s)
            fmt = dict(zip(locs, new))
            from matplotlib.ticker import FixedFormatter, FixedLocator
            allloc = list(axis.get_majorticklocs())
            axis.set_major_locator(FixedLocator(allloc))
            axis.set_major_formatter(FixedFormatter([fmt.get(v, '') for v in allloc]))
            fig.canvas.draw()
            rr = fig.canvas.get_renderer()
            bb = [tk.label1.get_window_extent(rr) if tk.label1.get_visible() else tk.label2.get_window_extent(rr) for tk, _ in shown]
            if name == 'x':
                bb.sort(key=lambda b: b.x0)
                clash = any((b1.x0 < b0.x1 for b0, b1 in zip(bb, bb[1:])))
            else:
                bb.sort(key=lambda b: b.y0)
                clash = False
            if clash:
                fmt = dict(zip(locs, old))
                axis.set_major_formatter(FixedFormatter([fmt.get(v, '') for v in allloc]))
                for (tk, _), s in zip(shown, old):
                    tk.label1.set_text(s)
                    tk.label2.set_text(s)
                changes.append({'axis': name, 'old': old, 'new': old, 'kept': 'longer labels would overlap'})
                continue
            changes.append({'axis': name, 'old': old, 'new': new})
    return changes
CLEAR_PT = 1.0

def _seg_hits(xy, box):
    import numpy as _np
    if len(xy) < 2:
        return _np.zeros(0, bool)
    a, b = (xy[:-1], xy[1:])
    ok = _np.all(_np.isfinite(a), 1) & _np.all(_np.isfinite(b), 1)
    dx, dy = (b[:, 0] - a[:, 0], b[:, 1] - a[:, 1])
    u0 = _np.zeros(len(a))
    u1 = _np.ones(len(a))
    rej = ~ok
    with _np.errstate(divide='ignore', invalid='ignore'):
        for p, q in ((-dx, a[:, 0] - box[0]), (dx, box[2] - a[:, 0]), (-dy, a[:, 1] - box[1]), (dy, box[3] - a[:, 1])):
            rej |= (p == 0) & (q < 0)
            t = q / p
            neg = p < 0
            pos = p > 0
            u0 = _np.where(neg, _np.maximum(u0, t), u0)
            u1 = _np.where(pos, _np.minimum(u1, t), u1)
    return ~rej & (u0 <= u1)

def clear_check(fig, stem=None):
    import numpy as _np
    from matplotlib.text import Text
    from matplotlib.lines import Line2D
    from matplotlib.collections import PathCollection, LineCollection
    res = {'stem': stem, 'test': 'every text clear of every frame, line and mark by CLEAR_PT', 'clear_pt': CLEAR_PT}
    hits = []
    try:
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        px = fig.dpi / 72.0
        g = CLEAR_PT * px
        shown, hidden = _shown_ticklabel_ids(fig)
        texts = []
        for t in fig.findobj(Text):
            if id(t) in hidden or not t.get_visible():
                continue
            if not t.get_text().strip():
                continue
            if t.axes is not None and (not t.axes.get_visible()):
                continue
            texts.append(t)
        axes = []
        for a in fig.axes:
            axes.append(a)
            axes.extend(getattr(a, 'child_axes', []))
        obst = []
        marks = []
        for a in axes:
            if not a.get_visible():
                continue
            nm = a.get_label() or a.get_gid() or ''
            if a.axison:
                for side, sp in a.spines.items():
                    if not sp.get_visible() or sp.get_linewidth() <= 0:
                        continue
                    try:
                        pth = sp.get_path()
                        xy = sp.get_transform().transform(pth.vertices)
                    except Exception:
                        continue
                    obst.append(('frame:' + side, nm, xy, None))
            for ln in a.get_lines():
                if not ln.get_visible() or ln.get_alpha() == 0:
                    continue
                cb = None
                if ln.get_clip_on():
                    c = ln.get_clip_box() or a.bbox
                    cb = (c.x0, c.y0, c.x1, c.y1)
                xy = ln.get_transform().transform(_np.asarray(ln.get_xydata(), float))
                ls = ln.get_linestyle()
                if ls not in ('None', '', ' ', 'none') and ln.get_linewidth() > 0:
                    obst.append(('line', nm, xy, cb))
                mk = ln.get_marker()
                if mk not in (None, 'None', '', ' ', 'none') and ln.get_markersize() > 0:
                    h = 0.5 * ln.get_markersize() * px
                    for x, y in xy:
                        if _np.isfinite(x) and _np.isfinite(y) and (cb is None or (cb[0] <= x <= cb[2] and cb[1] <= y <= cb[3])):
                            marks.append(('mark', nm, x, y, h))
            for col in a.collections:
                if not col.get_visible():
                    continue
                if isinstance(col, PathCollection) and len(col.get_offsets()):
                    off = col.get_offset_transform().transform(_np.asarray(col.get_offsets(), float))
                    sz = col.get_sizes()
                    cb = None
                    if col.get_clip_on():
                        c = col.get_clip_box() or a.bbox
                        cb = (c.x0, c.y0, c.x1, c.y1)
                    for i, (x, y) in enumerate(off):
                        if cb is not None and (not (cb[0] <= x <= cb[2] and cb[1] <= y <= cb[3])):
                            continue
                        s_ = sz[i % len(sz)] if len(sz) else 20.0
                        h = 0.5 * _np.sqrt(s_) * px
                        marks.append(('scatter', nm, x, y, h))
                elif isinstance(col, LineCollection):
                    tr = col.get_transform()
                    cb = None
                    if col.get_clip_on():
                        c = col.get_clip_box() or a.bbox
                        cb = (c.x0, c.y0, c.x1, c.y1)
                    for seg in col.get_segments():
                        if len(seg) >= 2:
                            obst.append(('linecollection', nm, tr.transform(_np.asarray(seg, float)), cb))
        for t in texts:
            b = Text.get_window_extent(t, r)
            box = (b.x0 - g, b.y0 - g, b.x1 + g, b.y1 + g)
            tag = t.get_text()[:40]
            for kind, nm, xy, cb in obst:
                bx = box
                if cb is not None:
                    bx = (max(box[0], cb[0]), max(box[1], cb[1]), min(box[2], cb[2]), min(box[3], cb[3]))
                    if bx[0] > bx[2] or bx[1] > bx[3]:
                        continue
                if kind in ('line', 'linecollection') and len(xy) <= 3:
                    lead = 3.0 * px
                    near = False
                    for x, y in (xy[0], xy[-1]):
                        if b.x0 - lead <= x <= b.x1 + lead and b.y0 - lead <= y <= b.y1 + lead:
                            near = True
                    if near:
                        continue
                sh = _seg_hits(xy, bx)
                if sh.any():
                    i0 = int(_np.argmax(sh))
                    hits.append({'text': tag, 'obstacle': kind, 'axes': nm, 'segments': int(sh.sum()), 'n_points': int(len(xy)), 'first_hit_px': [round(float(xy[i0][0]), 1), round(float(xy[i0][1]), 1), round(float(xy[i0 + 1][0]), 1), round(float(xy[i0 + 1][1]), 1)], 'text_box_px': [round(float(v), 1) for v in (b.x0, b.y0, b.x1, b.y1)]})
            for kind, nm, x, y, h in marks:
                if x + h >= box[0] and x - h <= box[2] and (y + h >= box[1]) and (y - h <= box[3]):
                    hits.append({'text': tag, 'obstacle': kind, 'axes': nm, 'at_px': [round(float(x), 1), round(float(y), 1)]})
        panels = [a for a in axes if a.get_visible() and a.axison and (a.get_gid() != 'key') and (not _is_bar_axes(fig, a)) and any((sp.get_visible() for sp in a.spines.values()))]
        past = []
        if panels:
            fx0 = min((a.get_window_extent(r).x0 for a in panels))
            fx1 = max((a.get_window_extent(r).x1 for a in panels))
            tol = 0.5 * px
            items = [(t, Text.get_window_extent(t, r)) for t in texts]
            lgs = list(fig.legends) + [a.get_legend() for a in axes if a.get_legend() is not None]
            items += [(lg, lg.get_window_extent(r)) for lg in lgs if lg.get_visible()]
            owner = {id(lab): a_ for a_, _n, _k, lab in shown}
            for a_ in axes:
                for o_ in (a_.xaxis.label, a_.yaxis.label, a_.title, a_._left_title, a_._right_title):
                    owner[id(o_)] = a_
            for o, b in items:
                side = 'left' if b.x0 < fx0 - tol else 'right' if b.x1 > fx1 + tol else None
                if side is None:
                    continue
                own = owner.get(id(o)) or getattr(o, 'axes', None)
                at_edge = False
                if own is not None and own in panels:
                    fe = own.get_window_extent(r)
                    at_edge = abs(fe.x0 - fx0) <= tol if side == 'left' else abs(fe.x1 - fx1) <= tol
                lab = o.get_text()[:40] if isinstance(o, Text) else 'legend ' + ' / '.join((x.get_text() for x in o.get_texts()))[:40]
                past.append({'text': lab, 'side': side, 'by_pt': round(float(fx0 - b.x0 if side == 'left' else b.x1 - fx1) / px, 1), 'owner_frame_at_edge': bool(at_edge)})
        res.update({'n_texts': len(texts), 'n_lines': len(obst), 'n_marks': len(marks), 'hits': hits, 'past_grid_edge': past, 'pass': not hits})
    except Exception as e:
        res['error'] = '%s: %s' % (type(e).__name__, e)
        res['pass'] = False
    if stem:
        with open(os.path.join(FIG, stem + '_clearcheck.json'), 'w', encoding='ascii', newline='\n') as f:
            json.dump(res, f, indent=1)
            f.write('\n')
    return res
TRIM_PAD_PT = 1.0

def trim_box(fig, meta_pdf=None, dpi=PNG_DPI, pad_pt=TRIM_PAD_PT):
    import ctypes
    import io
    import math
    import pypdfium2 as pdfium
    import pypdfium2.raw as pdfc
    from matplotlib.transforms import Bbox
    buf = io.BytesIO()
    fig.savefig(buf, format='pdf', metadata=meta_pdf)
    doc = pdfium.PdfDocument(buf.getvalue())
    try:
        page = doc[0]
        pw, ph = page.get_size()
        s = dpi / 72.0
        w, h = (int(math.ceil(pw * s)), int(math.ceil(ph * s)))
        bm = pdfium.PdfBitmap.new_native(w, h, pdfc.FPDFBitmap_BGR)
        bm.fill_rect((255, 255, 255, 255), 0, 0, w, h)
        pdfc.FPDF_RenderPageBitmapWithMatrix(bm.raw, page.raw, ctypes.byref(pdfc.FS_MATRIX(s, 0, 0, s, 0, 0)), ctypes.byref(pdfc.FS_RECTF(0, 0, w, h)), 0)
        img = np.array(bm.to_numpy())[:h, :w, :3]
        page.close()
    finally:
        doc.close()
    ink = (img < 255).any(axis=2)
    rows = np.flatnonzero(ink.any(axis=1))
    cols = np.flatnonzero(ink.any(axis=0))
    if rows.size == 0:
        raise SystemExit('trim: the figure draws no ink')
    p = int(round(pad_pt * dpi / 72.0))
    left, right = (int(cols[0]) - p, int(cols[-1]) + 1 + p)
    top, bottom = (int(rows[0]) - p, int(rows[-1]) + 1 + p)
    fw, fh = (float(v) for v in fig.get_size_inches())
    y0 = math.floor(fh * dpi - bottom + 1e-06)
    box = Bbox.from_extents(left / dpi, y0 / dpi, right / dpi, fh - top / dpi)
    rec = {'dpi': dpi, 'pad_px': p, 'pad_pt': p * 72.0 / dpi, 'box_px_from_top_left': [left, top, right, bottom], 'size_pt': [(right - left) * 72.0 / dpi, (fh - top / dpi - y0 / dpi) * 72.0], 'beyond_canvas': bool(left < 0 or top < 0 or y0 < 0 or (right > w) or (bottom > h))}
    return (box, rec)

def save(fig, stem, plotted, sources):
    os.makedirs(FIG, exist_ok=True)
    meta_pdf = {'Title': stem, 'Author': '', 'Subject': '', 'Keywords': '', 'Creator': '', 'Producer': ''}
    pdf = os.path.join(FIG, stem + '.pdf')
    png = os.path.join(FIG, stem + '.png')
    tdc = unify_tick_decimals(fig)
    print('tickdecimals %s changed %d axes' % (stem, len(tdc)))
    if isinstance(plotted, dict):
        plotted['tick_decimals_layout'] = tdc
    box, trim = trim_box(fig, meta_pdf)
    print('trim %s pad %.2f pt size %.2f x %.2f pt%s' % (stem, trim['pad_pt'], trim['size_pt'][0], trim['size_pt'][1], ' BEYOND CANVAS' if trim['beyond_canvas'] else ''))
    fig.savefig(pdf, format='pdf', metadata=meta_pdf, bbox_inches=box)
    fig.savefig(png, format='png', dpi=PNG_DPI, metadata={'Software': ''}, bbox_inches=box)
    tc = text_canvas_check(fig, stem)
    print('textcheck %s pass %s outside %d multiline %d overlaps %d%s' % (stem, tc.get('pass'), len(tc.get('texts_outside_canvas', [])), len(tc.get('multiline_ticklabels', [])), len(tc.get('ticklabel_overlaps', [])), ' ERROR ' + tc['error'] if 'error' in tc else ''))
    cc = clear_check(fig, stem)
    print('clearcheck %s pass %s hits %d past_grid %d%s' % (stem, cc.get('pass'), len(cc.get('hits', [])), len(cc.get('past_grid_edge', [])), ' ERROR ' + cc['error'] if 'error' in cc else ''))
    lc = layout_check(fig, getattr(fig, '_house_blocks', None))
    fl = lc.get('fails', [])
    print('c3check %s pass %s fails %d%s' % (stem, lc.get('pass'), len(fl), (': ' + '; '.join(fl))[:900] if fl else ''))
    if isinstance(plotted, dict):
        lay = plotted.get('layout')
        if lay is None:
            plotted['layout'] = {'c3_check': lc}
        elif isinstance(lay, dict):
            lay['c3_check'] = lc
        else:
            plotted['c3_layout_check'] = lc
    rec = {'stem': stem, 'sources': {os.path.relpath(s, ROOT): sha256_file(s) for s in sources}, 'plotted': _clean(plotted)}
    with open(os.path.join(FIG, stem + '_plotted.json'), 'w', encoding='ascii', newline='\n') as f:
        json.dump(rec, f, indent=1)
        f.write('\n')
    plt.close(fig)
    return (pdf, png)
if __name__ == '__main__':
    print('family:', FAMILY)
