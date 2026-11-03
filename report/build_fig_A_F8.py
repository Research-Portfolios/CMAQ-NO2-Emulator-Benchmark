"""Draws figure fig_A_F8: the exploratory RMSE differences between pairs of models by held-out year, each with its 95 percent interval, checking each row's name and reading against the stored ones."""
import json
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_style as S
import build_c2_sealed as SEAL
import build_c2_common as C2
import build_fig_common as C
from matplotlib.patches import Rectangle
STEM = 'fig_A_F8'
NUMBERS = os.path.join(S.ROOT, 'results', 'report', 'numbers.json')
DEF_KEY = {'ladder P -> A1': ['RMSE(P) - RMSE(A1)'], 'ladder A1 -> A2': ['RMSE(A1) - RMSE(A2)'], 'ladder A2 -> A3': ['RMSE(A2) - RMSE(A3)'], 'ladder A3 -> M': ['RMSE(A3) - RMSE(M)'], 'M-C7 stand-in': ['RMSE(M-C7) - RMSE(M)'], 'A3OFF C input': ['RMSE(A3OFF) - RMSE(A3)', 'C channel'], 'PIX-R context': ['RMSE(PIX-R) - RMSE(M)', 'spatial context'], 'CLIM-EM reference': ['RMSE(CLIM-EM) - RMSE(M)'], 'A4 on 2019': ['RMSE(A4) - RMSE(M)', 'direct target'], 'A3BL unseen place': ['RMSE(A3BL) - RMSE(A3)', "block's cells"], '2 x 2 interaction': ['[RMSE(A4) - RMSE(M)] - [RMSE(A2) - RMSE(A3)]'], 'ensemble P ens5 - M ens5': ['RMSE(P ens5) - RMSE(M ens5)'], 'ensemble M s0 - M ens5': ['RMSE(M s0) - RMSE(M ens5)']}
GROUPS = [('Steps from the reference to the residual U-Net', ['ladder P -> A1', 'ladder A1 -> A2', 'ladder A2 -> A3', 'ladder A3 -> M']), ('Inputs', ['M-C7 stand-in', 'A3OFF C input']), ('Spatial context and reference', ['PIX-R context', 'CLIM-EM reference']), ('Target and training length', ['A4 on 2019', '2 x 2 interaction']), ('Block left out of the loss', ['A3BL unseen place']), ('Seeds and ensembles', ['ensemble P ens5 - M ens5', 'ensemble M s0 - M ens5'])]
NAMES = {'ladder P -> A1': ('Reference U-Net minus step 1', 'Step 1: squared-error loss', 'RMSE(P) - RMSE(A1)', 'Reference U-Net', 'Step 1'), 'ladder A1 -> A2': ('Step 1 minus step 2', 'Step 2: residual blocks, added inputs, log scale', 'RMSE(A1) - RMSE(A2)', 'Step 1', 'Step 2'), 'ladder A2 -> A3': ('Step 2 minus step 3', 'Step 3: residual target and climatology input', 'RMSE(A2) - RMSE(A3)', 'Step 2', 'Step 3'), 'ladder A3 -> M': ('Step 3 minus residual U-Net', 'Step 4: training lengthened to {M} steps', 'RMSE(A3) - RMSE(M)', 'Step 3', 'Residual U-Net'), 'M-C7 stand-in': ('Climatological minus own-date emissions', 'Residual U-Net with each emission input', 'RMSE(M-C7) - RMSE(M)', 'Climatological', 'Own-date'), 'A3OFF C input': ('Without minus with the climatology input', 'Step-3 network', 'RMSE(A3OFF) - RMSE(A3)', 'Without input', 'With input'), 'PIX-R context': ('Per-cell network minus residual U-Net', 'No spatial context; {P} against {M} steps', 'RMSE(PIX-R) - RMSE(M)', 'Per-cell network', 'Residual U-Net'), 'CLIM-EM reference': ('Emission-scaled climatology minus residual U-Net', 'A simpler reference', 'RMSE(CLIM-EM) - RMSE(M)', 'Climatology', 'Residual U-Net'), 'A4 on 2019': ('Step 2, 16,000 steps minus residual U-Net', 'Target change at {M} steps', 'RMSE(A4) - RMSE(M)', 'Step 2, 16,000 steps', 'Residual U-Net'), '2 x 2 interaction': ('Target change at {M} minus at {P} steps', 'Negative: smaller gain at {M} steps', '[RMSE(A4) - RMSE(M)] - [RMSE(A2) - RMSE(A3)]', None, None), 'A3BL unseen place': ('Block left out minus block in the loss', "Step-3 network, on the block's cells", 'RMSE(A3BL) - RMSE(A3)', 'Block left out', 'Block in the loss'), 'ensemble P ens5 - M ens5': ('Reference minus residual U-Net', 'Five-seed means; {P} against {M} steps', 'RMSE(P ens5) - RMSE(M ens5)', 'Reference U-Net', 'Residual U-Net'), 'ensemble M s0 - M ens5': ('One seed minus the five-seed mean', 'Residual U-Net', 'RMSE(M s0) - RMSE(M ens5)', 'One seed', 'Five-seed mean')}
B_LO, B_HI = (-0.06, 0.12)
A_LO, A_HI = (-0.25, 0.7)
BRACKET = ['ladder P -> A1', 'ladder A1 -> A2', 'ladder A2 -> A3']
BAND = '#F0F0F0'
BAR = '#707070'
GUIDE = '#EBEBEB'
CLS_FILL = {'y': '#4F4F4F', 'x': '#A8A8A8', 'within': '#E3E3E3', 'edge': '#FFFFFF', 'zero': '#FFFFFF'}
CLS_HATCH = {'y': None, 'x': None, 'within': None, 'edge': '//////', 'zero': 'xxxxxx'}
CLS_NAME = {'y': 'Y closer to CMAQ', 'x': 'X closer to CMAQ', 'within': 'Within the margin', 'edge': 'Across a margin edge', 'zero': 'Below zero'}
SW_PT, SW_GAP_PT = (6.0, 2.0)

def in_range_b(r):
    return r['lo'] >= B_LO and r['hi'] <= B_HI

def klass(r):
    lo, hi, m = (r['lo'], r['hi'], r['margin'])
    if m == 0.0:
        if hi < 0:
            return 'zero'
        raise SystemExit('a zero-compared row not below zero: reword')
    if lo > m or hi < -m:
        return 'beyond'
    if lo >= -m and hi <= m:
        return 'within'
    return 'edge'

def stored_class(reading):
    if reading.startswith('negative interaction'):
        return 'zero'
    if reading.startswith('the interval excludes differences larger'):
        return 'within'
    if reading.startswith('inconclusive'):
        return 'edge'
    if reading.endswith('closer to CMAQ'):
        return 'beyond'
    raise SystemExit('unknown stored reading %r' % reading)

def fill_class(r):
    cl = klass(r)
    if cl != 'beyond':
        return cl
    return 'y' if r['lo'] > r['margin'] else 'x'

def reading_text(c, r):
    cl = klass(r)
    if cl == 'within':
        return 'Within the margin'
    if cl == 'edge':
        return 'Across a margin edge'
    if cl == 'zero':
        return 'Below zero'
    x_name, y_name = (NAMES[c][3], NAMES[c][4])
    return '%s closer' % (y_name if r['lo'] > r['margin'] else x_name)

def main():
    src, = SEAL.verify(S.CV, ['exploratory.json'])
    d = S.load_json(S.CV, 'exploratory.json')
    rows = d['rows']
    if len(rows) != int(d['count_of_intervals_read']):
        raise SystemExit('row count differs from count_of_intervals_read')
    with open(NUMBERS, encoding='utf-8') as f:
        nb = json.load(f)
    st = {'P': nb['rec.steps.p']['text'], 'M': nb['rec.steps.m']['text']}
    order = []
    for r in rows:
        if r['contrast'] not in order:
            order.append(r['contrast'])
    listed = [c for _, cs in GROUPS for c in cs]
    if sorted(listed) != sorted(order):
        raise SystemExit('group list differs from the stored contrasts')
    for r in rows:
        for k in DEF_KEY[r['contrast']] + [NAMES[r['contrast']][2]]:
            if k not in r['definition']:
                raise SystemExit('name of %r contradicted by its definition %r' % (r['contrast'], r['definition']))
        if klass(r) != stored_class(r['reading']):
            raise SystemExit('class differs from stored reading: %r' % r)
    names = {c: (v[0].format(**st), v[1].format(**st)) for c, v in NAMES.items()}
    groups = []
    for c in order:
        groups.append((c, sorted([r for r in rows if r['contrast'] == c], key=lambda r: int(r['fold']))))
    by_c = dict(groups)
    levels = sorted({r['level'] for r in rows})
    nres = sorted({int(r['n_resamples']) for r in rows})
    margins = sorted({float(r['margin']) for r in rows})
    if len(levels) != 1 or len(nres) != 1:
        raise SystemExit('mixed levels or resample counts: reword')
    zero_margin = [c for c, rs in groups if all((r['margin'] == 0.0 for r in rs))]
    m_pos = [m for m in margins if m > 0]
    if len(m_pos) != 1 or abs(m_pos[0] - float(d['margin'])) > 0:
        raise SystemExit('margins not as expected')
    mg = m_pos[0]
    block, _ = C2.code_const('BLOCK', 'family.py')
    lv = C2.pct(levels[0])
    HEAD, SUB, CGAP, GGAP = (1.35, 1.0, 0.35, 0.4)
    y = 0.0
    heads, subs, labs, pts = ([], [], [], [])
    for gi, (g, cs) in enumerate(GROUPS):
        if gi:
            y += GGAP
        heads.append((g, y))
        y += HEAD
        for ci, c in enumerate(cs):
            if ci:
                y += CGAP
            if c == BRACKET[0]:
                subs.append(('Steps 1 to 3 at %s training steps' % st['P'], y))
                y += SUB
            rs = by_c[c]
            labs.append((c, y))
            for k, r in enumerate(rs):
                pts.append((y + k, c, r))
            y += max(len(rs), 2)
    PITCH = 0.1225
    _pu = PITCH * 72.0
    D_HEAD, D_CGAP, D_GGAP = (10.8 / _pu, 0.0, 12.4 / _pu - 1.0)
    dy = {}
    yd = 0.0
    dheads = []
    for gi, (g, cs) in enumerate(GROUPS):
        if gi:
            yd += D_GGAP
        dheads.append((g, yd))
        yd += D_HEAD
        for ci, c in enumerate(cs):
            if ci:
                yd += D_CGAP
            for k, r in enumerate(by_c[c]):
                dy[c, r['fold']] = yd + k
            yd += len(by_c[c])
    n_draw = yd
    W = 6.5
    x_lab = 0.2
    x_name = 0.3
    a_l, a_r = (2.96, 4.86)
    b_l, b_r = (5.095, 6.04)
    ah = PITCH * (n_draw + 0.9)
    y_ax0 = 0.5

    def yin(u):
        return -u
    C.setup()
    fig = S.plt.figure(figsize=(W, 9.0))
    axes = {}
    for key, l, r_, xl in (('a', a_l, a_r, (A_LO, A_HI)), ('b', b_l, b_r, (B_LO, B_HI))):
        ax = fig.add_axes([l / W, y_ax0 / 9.0, (r_ - l) / W, ah / 9.0])
        ax.set_xlim(*xl)
        ax.set_ylim(-(n_draw - 1 + 0.45), 1.25)
        ax.axvspan(-mg, mg, color=BAND, lw=0, zorder=0)
        for s_ in (-mg, mg):
            ax.axvline(s_, color=S.MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
        ax.axvline(0.0, color=S.REF_INK, lw=0.6, zorder=1)
        C.frame(ax)
        axes[key] = ax
    ax_a, ax_b = (axes['a'], axes['b'])
    ta = [-0.2, 0.0, 0.2, 0.4, 0.6]
    ax_a.set_xticks(ta)
    ax_a.set_xticklabels([C2.tick(v, 1) for v in ta])
    tb = [-0.05, 0.0, 0.05]
    ax_b.set_xticks(tb)
    ax_b.set_xticklabels(['−0.05', '0', '0.05'])
    import matplotlib.transforms as mtrans
    in_b, drawn = ([], [])
    ylist, yearlab, readlab = ([], [], [])
    for yu, c, r in pts:
        yy = -dy[c, r['fold']]
        for key, ax in axes.items():
            ax.axhline(yy, color=GUIDE, lw=0.4, zorder=0.5)
            if key == 'b' and (not in_range_b(r)):
                side = B_HI if r['point'] > B_HI else B_LO
                ax.plot([side - 0.006 if side > 0 else side + 0.006], [yy], ls='none', marker='>' if side > 0 else '<', ms=3.6, mfc=S.MUTED, mec='none', zorder=3)
                continue
            ax.plot([r['lo'], r['hi']], [yy, yy], color=BAR, lw=2.6, solid_capstyle='butt', zorder=2)
            ax.plot([r['point'], r['point']], [yy - 0.3, yy + 0.3], color=S.INK, lw=0.6, solid_capstyle='butt', zorder=3)
            if key == 'b':
                in_b.append((c, r['fold']))
        _bw = (b_r - b_l) * 72.0
        _x0 = 1.0 + (S.TICK_LEN_PT + SW_GAP_PT) / _bw
        _fc = fill_class(r)
        ax_b.add_patch(Rectangle((_x0, yy - 0.32), SW_PT / _bw, 0.64, transform=mtrans.blended_transform_factory(ax_b.transAxes, ax_b.transData), facecolor=CLS_FILL[_fc], edgecolor=S.INK2, lw=0.5, hatch=CLS_HATCH[_fc], clip_on=False, zorder=3))
        rt = reading_text(c, r)
        ylist.append(yy)
        yearlab.append(str(r['fold']))
        readlab.append(rt)
        drawn.append({'contrast': c, 'fold': r['fold'], 'y_units': yu, 'layout_y_drawn': yy, 'row_name': names[c][0], 'description': names[c][1], 'subtraction': NAMES[c][2], 'nv_class': klass(r), 'reading_drawn': rt, 'fill_class': fill_class(r), 'in_panel_b': in_range_b(r)})
    if len(in_b) != 12:
        raise SystemExit('panel (b) should hold 12 rows, has %d' % len(in_b))
    ax_a.set_yticks(ylist)
    ax_a.set_yticklabels(yearlab)
    for t in ax_a.get_yticklabels():
        t.set_color(S.INK2)
    ax_b.set_yticks(ylist)
    ax_b.yaxis.set_ticks_position('right')
    ax_b.set_yticklabels([])
    for ax in (ax_a, ax_b):
        C.house_ticks(ax, 'both')
    import matplotlib.transforms as mtrans
    trn = mtrans.blended_transform_factory(fig.transFigure, ax_a.transData)
    for g, yu in dheads:
        ax_a.text(x_lab / W, -yu, g, transform=trn, ha='left', va='center', fontsize=S.FS_LABEL, color=S.INK, clip_on=False, gid='rowgroup')
    for c, yu in labs:
        y0 = -dy[c, by_c[c][0]['fold']]
        ax_a.text(x_name / W, y0, names[c][0], transform=trn, ha='left', va='center', fontsize=S.FS_TICK, color=S.INK, clip_on=False, gid='rowname')
    C.tag(ax_a, 'a', 'Full range')
    C.tag(ax_b, 'b', 'Magnified')
    yb = 0.45
    ax_a.plot([B_LO, B_LO, B_HI, B_HI], [yb, yb + 0.3, yb + 0.3, yb], color=S.INK, lw=0.6, solid_capstyle='butt', zorder=4)
    xt = C.block_title(fig, [ax_a, ax_b], 'Difference in RMSE against CMAQ (ppb)', axis='x')
    fig.canvas.draw()
    rr = fig.canvas.get_renderer()
    _pa = ax_a.get_position()
    _yc = _pa.y0 + 0.5 * _pa.height
    fig.text(0.06 / W, _yc, 'Comparison (X minus Y) and held-out year', ha='left', va='center', rotation=90, fontsize=S.FS_LABEL, color=S.INK, gid='colhead')
    _xr = b_r + (S.TICK_LEN_PT + SW_GAP_PT + SW_PT + 3.0) / 72.0
    fig.text(_xr / W, _yc, 'Reading of the interval', ha='left', va='center', rotation=90, fontsize=S.FS_LABEL, color=S.INK, gid='colhead')
    S.register_block(fig, 'panels', [ax_a, ax_b])
    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerBase

    class _HBand(HandlerBase):

        def create_artists(self, legend, orig, xd, yd, w, h, fs, tr):
            rect = Rectangle((xd, yd), w, h, fc=BAND, ec='none', transform=tr)
            l1 = Line2D([xd, xd], [yd, yd + h], color=S.MUTED, lw=0.6, ls=(0, (3, 2)), transform=tr)
            l2 = Line2D([xd + w, xd + w], [yd, yd + h], color=S.MUTED, lw=0.6, ls=(0, (3, 2)), transform=tr)
            return [rect, l1, l2]

    class _HInt(HandlerBase):

        def create_artists(self, legend, orig, xd, yd, w, h, fs, tr):
            ym = yd + h / 2.0
            b = Line2D([xd, xd + w], [ym, ym], color=BAR, lw=2.6, solid_capstyle='butt', transform=tr)
            t = Line2D([xd + w / 2.0] * 2, [yd, yd + h], color=S.INK, lw=0.6, transform=tr)
            return [b, t]

    class _HZero(HandlerBase):

        def create_artists(self, legend, orig, xd, yd, w, h, fs, tr):
            return [Line2D([xd + w / 2.0] * 2, [yd, yd + h], color=S.REF_INK, lw=0.6, transform=tr)]

    class _HBracket(HandlerBase):

        def create_artists(self, legend, orig, xd, yd, w, h, fs, tr):
            return [Line2D([xd, xd, xd + w, xd + w], [yd + 0.2 * h, yd + 0.8 * h, yd + 0.8 * h, yd + 0.2 * h], color=S.INK, lw=0.6, transform=tr)]

    class _HArrow(HandlerBase):

        def create_artists(self, legend, orig, xd, yd, w, h, fs, tr):
            return [Line2D([xd + w - 1.8], [yd + h / 2.0], ls='none', marker='>', ms=3.6, mfc=S.MUTED, mec='none', transform=tr)]

    class _K:
        pass
    ks = [_K() for _ in range(5)]
    hmap = {ks[0]: _HInt(), ks[1]: _HBand(), ks[2]: _HZero(), ks[3]: _HBracket(), ks[4]: _HArrow()}
    from matplotlib.patches import Patch
    kc = {q: Patch(facecolor=CLS_FILL[q], edgecolor=S.INK2, lw=0.5, hatch=CLS_HATCH[q]) for q in CLS_FILL}
    used = sorted({fill_class(r) for _, _, r in pts}, key=['y', 'x', 'zero', 'within', 'edge'].index)
    krow = [(kc[q], CLS_NAME[q]) for q in used]
    key = [(ks[0], 'Point and %s%% interval' % lv), (ks[1], 'Margin, −%s to +%s ppb' % (C2.tick(mg, 2), C2.tick(mg, 2))), (ks[2], 'Zero'), (ks[3], 'Range of panel (b)'), (ks[4], 'Row beyond the range of (b)')]
    legs = C.house_key(fig, [ax_a, ax_b], None, None, rows=[key[:3], key[3:], krow[:3], krow[3:]], side='above', handler_map=hmap)
    fig.canvas.draw()
    rr = fig.canvas.get_renderer()
    top_in = max((lg.get_window_extent(rr).y1 for lg in legs)) / fig.dpi
    bot_in = xt.get_window_extent(rr).y0 / fig.dpi
    H = top_in - bot_in + 0.12
    dh = bot_in - 0.06
    for o in list(fig.axes):
        p = o.get_position()
        o.set_position([p.x0, (p.y0 * 9.0 - dh) / H, p.width, p.height * 9.0 / H])
    for o in list(fig.texts):
        x_, y_ = o.get_position()
        o.set_position((x_, (y_ * 9.0 - dh) / H))
    for lg in fig.legends:
        bb = lg.get_bbox_to_anchor()
        x0 = bb.x0 / fig.bbox.width
        y0 = (bb.y0 / fig.dpi - dh) / H
        lg.set_bbox_to_anchor((x0, y0), transform=fig.transFigure)
    fig.set_size_inches(W, H)
    fig.canvas.draw()
    problems = []
    fig.canvas.draw()
    rr = fig.canvas.get_renderer()
    yl_x0 = min((t.get_window_extent(rr).x0 for t in ax_a.get_yticklabels())) / fig.dpi
    for t in ax_a.texts:
        if (t.get_gid() or '') in ('rowname', 'rowgroup'):
            bb = t.get_window_extent(rr)
            if bb.x1 / fig.dpi > yl_x0 - 0.06:
                problems.append('label runs into years: %r' % t.get_text())
    plotted = {'rows_top_to_bottom': [{'contrast': c, 'plain': names[c][0], 'fold': r['fold'], 'point': r['point'], 'lo': r['lo'], 'hi': r['hi'], 'level': r['level'], 'margin': r['margin'], 'half_width': r['half_width'], 'reading': r['reading'], 'definition': r['definition'], 'n_resamples': r['n_resamples'], 'finite_resamples': r['finite_resamples'], 'y': -[p[0] for p in pts if p[1] == c and p[2] is r][0], 'layout_y_drawn': -dy[c, r['fold']]} for c, rs in groups for r in rs], 'plain_mapping': {c: names[c][0] for c in order}, 'definition_keys': DEF_KEY, 'x_limits': [A_LO, A_HI], 'x_ticks': ta, 'x_limits_b': [B_LO, B_HI], 'x_ticks_b': tb, 'margin': mg, 'zero_margin_contrasts': zero_margin, 'count_of_intervals_read': d['count_of_intervals_read'], 'kind': d['kind'], 'covers': d['covers'], 'block_days': block, 'source_keys': 'CV exploratory.json rows (contrast, fold) -> point, lo, hi, level, margin, reading, definition; top-level kind, margin, covers', 'groups': [[g, cs] for g, cs in GROUPS], 'drawn_rows_top_to_bottom': drawn, 'nv_class_rule': 'family.py: beyond when lo > m or hi < -m (closer: the model after minus when lo > m, else the one before); within when -m <= lo and hi <= m; else across an edge; zero-margin rows below zero when hi < 0', 'label_problems': problems, 'steps_text': st, 'figure_in': [W, H]}
    plotted['fc_check'] = C.fc_check(fig)
    plotted['text_check'] = C.text_check(fig)
    fcc, tc = (plotted['fc_check'], plotted['text_check'])
    print('fc_check pass', fcc['pass'], 'case_pass', fcc['case_pass'], ascii(fcc['case']), ascii(fcc['overlaps']), ascii(fcc['outside']), fcc['small'], fcc['open_frames'])
    print('text_check pass', tc['pass'], ascii(tc['overlaps']), ascii(tc['within_0.08in_of_edge']), 'fig', tc['figure_in'])
    print('problems', ascii(problems))
    S.save(fig, STEM, plotted, [src])
    if zero_margin != ['2 x 2 interaction']:
        raise SystemExit('the caption names the interaction as the only row read at zero: reword')
    yrs = sorted({int(r['fold']) for r in rows})
    cap = "\\textbf{Exploratory differences in RMSE against CMAQ (ppb) between pairs of models of daily mean surface NO$_2$, contiguous US land cells of the 12 km grid, by held-out year (%d to %d).} Each row is one model's RMSE minus another's, so a positive value means the model after ``minus'' was closer to CMAQ; the interaction is the gain from the target change at %s steps minus its gain at %s. Networks are single trainings (seed 0) except the five-seed means; the last row compares one training with that mean. Steps 1 to 3 are defined in Appendix G and were trained for %s steps; from step 3 to the residual U-Net, training lengthened to %s steps. The emission row uses the residual U-Net with each emission input, and the climatology-input row the step-3 network. The per-cell network has no spatial context (%s against %s steps), and the emission-scaled climatology is a simpler reference. Step 2 trained for %s steps isolates the target change at that length, and the interaction is negative where that gain is smaller at %s steps. The block row uses the step-3 network on the block's cells; the five-seed means compare %s against %s steps, and the one-seed row is the residual U-Net's. Thin gray lines lead the eye along each row. Panel (a) shows all %d rows and (b) the %d whose intervals lie between $-$0.06 and +0.12 ppb. Ticks mark points and gray bars unadjusted %s percent intervals from %s paired resamples of %d-day blocks of test days, covering test-day sampling only. The square at the right of each row is filled by its interval's reading against the $\\pm$%s ppb margin or, for the interaction, zero (key)." % (yrs[0], yrs[-1], st['M'], st['P'], st['P'], st['M'], st['P'], st['M'], st['M'], st['M'], st['P'], st['M'], int(d['count_of_intervals_read']), len(in_b), lv, '{:,}'.format(nres[0]), block, C2.num_plain(mg, 2))
    alt = 'Two-panel point-and-interval chart of %d exploratory RMSE differences between pairs of models, read against a margin.' % len(rows)
    parts = ['Thin light gray guide lines run across both panels at every row. A key above the panels: point and %s percent interval (gray bar crossed by a black tick); margin, minus %s to plus %s ppb (shaded band with dashed edges); zero (solid line); range of panel (b) (a bracket); row beyond the range of (b) (a small gray triangle); then the fills of the reading squares: ' + ', '.join(('%s (%s)' % (CLS_NAME[q], {'y': 'dark gray', 'x': 'mid gray', 'within': 'light gray', 'edge': 'white with diagonal hatching', 'zero': 'white with cross hatching'}[q]) for q in used)) + '.', 'Left, under the title Comparison (X minus Y) and held-out year, rotated along the left edge: one row name on one line per comparison, the subtraction X minus Y, under six group headings; steps 1 to 3 were trained for %s steps. Then the held-out year of each row, a tick at each. At the right of panel (b), a tick and a filled square at each row, under the title Reading of the interval, rotated along the right edge.' % st['P'], "Panel (a), full range, ticks minus 0.2, 0.0, 0.2, 0.4, 0.6; panel (b), magnified, ticks minus 0.05, 0.00, 0.05, showing only the 12 rows whose intervals lie between minus 0.06 and plus 0.12 ppb, with a small gray triangle at its edge for every other row, pointing toward it. A bracket above the top row of panel (a) spans minus 0.06 to plus 0.12 ppb, the range shown in panel (b). One axis title under both panels: difference in RMSE against CMAQ (ppb). Each square's fill gives the row's reading: the closer model, X or Y, where the interval lies beyond the margin, within the margin, across a margin edge, or below zero for the interaction; the rows below name the closer model.", 'Rows from the top: group; pair; year: point (interval), reading.']
    gname = {c: g for g, cs in GROUPS for c in cs}
    for dr, (yu, c, r) in zip(drawn, pts):
        parts.append('%s; %s, %s: %s (%s to %s), %s.' % (gname[c], names[c][0], r['fold'], C2.num_plain(r['point']), C2.num_plain(r['lo']), C2.num_plain(r['hi']), dr['reading_drawn'].lower().replace('u-net', 'U-Net')))
    C.write_text(STEM, cap, alt, '\n'.join(parts))
    print('words', len(cap.split()), 'H', round(H, 3))
if __name__ == '__main__':
    main()
