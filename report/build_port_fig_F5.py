"""The drawing part of figure fig_F5: the skill scores of nine models, the seeds of the two U-Nets, RMSE by held-out year, and mean error against unbiased RMSE (the square root of RMSE squared minus mean error squared) for four models."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox, blended_transform_factory
import build_style as S
import build_fig_common as FC
YEARS = S.YEARS
ORDER = S.MODEL_ORDER
NAME = {'M': 'Residual U-Net', 'P': 'Reference U-Net', 'GBT': 'Gradient-boosted trees', 'PIX-R': 'Per-cell network', 'MET-REG': 'Weather regression', 'GBT-RIDGE': 'Linear model on tree features', 'CLIM-EM': 'Emission-scaled climatology', 'CLIM-WD': 'Weekday climatology', 'CLIM-C': 'Cell mean'}
D4 = ['M', 'P', 'GBT', 'CLIM-WD']
RING_INK = '#C8C8C8'
LEAD_INK = '#9A9A9A'
LEAD_LW = 0.4
BOX_LW = 0.5
LABEL_DIST_PT = 7.0
GROUP_PPB = 3.0 / 72.0 * (1.3 - 0.4) / 3.72
ZOOM = (0.468, 0.625, -0.036, 0.05)
WB = 0.78
MARGIN = 0.085
D_STRETCH = 1.5

def ink(m):
    return S.MODEL_TEXT.get(m, S.INK)

def tint(hx, a):
    c = FC.hex2rgb(hx) * a + (1 - a)
    return '#%02X%02X%02X' % tuple((int(round(v * 255)) for v in c))

def mk(m, big=False):
    return 4.6 if m in ('M', 'P') else 4.0

def _ext(fig, ax, r):
    fr = ax.get_window_extent(r)
    tb = ax.get_tightbbox(r)
    d = fig.dpi
    return ((fr.x0 - tb.x0) / d, (tb.x1 - fr.x1) / d, (tb.y1 - fr.y1) / d, (fr.y0 - tb.y0) / d)

def _seg_d(q0, q1, p):
    v = q1 - q0
    t_ = np.clip(np.dot(p - q0, v) / max(np.dot(v, v), 1e-09), 0, 1)
    return float(np.hypot(*q0 + t_ * v - p))

def _crosses(a0, a1, b0, b1):

    def orient(p, q, r):
        return np.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))
    return orient(a0, a1, b0) != orient(a0, a1, b1) and orient(b0, b1, a0) != orient(b0, b1, a1)

def _seg_pts(q0, q1, n=40):
    return [q0 + f * (q1 - q0) for f in np.linspace(0.0, 1.0, n)]

def _seg_hits_box(q0, q1, bb, pad=0.0):
    e = bb.padded(pad)
    for s in _seg_pts(q0, q1, 60)[1:-1]:
        if e.x0 < s[0] < e.x1 and e.y0 < s[1] < e.y1:
            return True
    return False

def _box_pt_d(bb, p):
    return float(np.hypot(np.clip(p[0], bb.x0, bb.x1) - p[0], np.clip(p[1], bb.y0, bb.y1) - p[1]))

def draw(plotted, pts):
    SS = plotted['panels']['SS']
    RM = plotted['panels']['RMSE']
    MEAN = plotted['key_order']['mean_skill_score']
    W = 6.5
    mb = {m: [float(pts[m, y]['MB']) for y in YEARS] for m in D4}
    rm = {m: [float(pts[m, y]['RMSE']) for y in YEARS] for m in D4}
    for m in D4:
        if rm[m] != [float(v) for v in RM['models'][m]['y']]:
            raise SystemExit('STOP: the RMSE of (d) differs from (c)')
    ub = {m: [float(np.sqrt(r * r - b * b)) for r, b in zip(rm[m], mb[m])] for m in D4}
    ha_, hc = (1.22, 1.06)
    X0, X1 = (0.4, 1.3)
    allub = [v for m in D4 for v in ub[m]]
    allmb = [v for m in D4 for v in mb[m]]
    if min(allub) < X0 + 0.02 or max(allub) > X1 - 0.02:
        raise SystemExit("STOP: (d)'s unbiased RMSE outside its axis")
    Y0 = np.floor((min(allmb) - 0.02) / 0.05) * 0.05
    Y1 = np.ceil((max(allmb) + 0.03) / 0.05) * 0.05
    fig = plt.figure(figsize=(W, 8.0))
    ax = fig.add_axes([0.3, 0.75, 0.3, 0.18])
    bx = fig.add_axes([0.8, 0.75, 0.13, 0.18])
    cx = fig.add_axes([0.3, 0.45, 0.4, 0.17])
    dx = fig.add_axes([0.3, 0.1, 0.4, 0.22])
    drawn = {}
    n = len(ORDER)
    for i, m in enumerate(ORDER):
        y = n - 1 - i
        v = np.array(SS['models'][m]['y'], dtype=float)
        c = S.MODEL_COLOUR[m]
        if v.max() - v.min() > 0:
            ax.barh(y, v.max() - v.min(), left=v.min(), height=0.62, color=tint(c, 0.22), edgecolor=c, lw=0.5, zorder=1)
        ax.plot(v, [y] * len(v), ls='none', marker='o', ms=2.4, mfc=S.MUTED, mec='none', mew=0.0, zorder=3)
        ax.plot(MEAN[m], y, ls='none', marker=S.MODEL_MARKER[m], ms=mk(m, True) + 0.8, color=c, mec='white', mew=0.5, zorder=4)
    ax.set_yticks(range(n))
    ax.set_yticklabels([NAME[m] for m in ORDER[::-1]])
    for t, m in zip(ax.get_yticklabels(), ORDER[::-1]):
        t.set_color(ink(m))
    FC.house_ticks(ax, 'y')
    ax.set_ylim(-0.7, n - 0.3)
    alla = [v for m in ORDER for v in SS['models'][m]['y']]
    a0 = np.floor(min(alla) / 0.2) * 0.2
    a1 = np.ceil(max(alla) / 0.2) * 0.2
    ax.set_xlim(a0 - 0.04, a1 + 0.04)
    xt = np.round(np.arange(a0, a1 + 1e-09, 0.2), 1)
    ax.set_xticks(xt)
    ax.set_xticklabels([('%.1f' % t).replace('-', S.MINUS) if abs(t) > 1e-09 else '0' for t in xt])
    ax.axvline(0, color=S.REF_INK, lw=0.6, zorder=2)
    ax.annotate('Zero skill', xy=(0.0, 1.0), xycoords=blended_transform_factory(ax.transData, ax.transAxes), xytext=(2.5, -2.0), textcoords='offset points', ha='left', va='top', fontsize=S.FS_TICK, color=S.INK, gid='zero_label')
    ax.set_xlabel('Skill score')
    ax.set_ylabel('Model')
    FC.frame(ax)
    FC.tag(ax, 'a', 'Skill score by model')
    drawn['a_xlim'] = list(ax.get_xlim())
    rows = [('M', 2016), ('M', 2019), ('P', 2016), ('P', 2019)]
    allb = []
    drawn['b_rows'] = []
    for j, (m, yr) in enumerate(rows):
        y = 3 - j
        vals = SS['seed_bars']['%s_%d' % (m, yr)]['values']
        sd = sorted((int(s) for s in vals))
        vv = np.array([float(vals[str(s)]) for s in sd])
        allb.extend(vv.tolist())
        jy = y + 0.12 * (np.arange(len(vv)) - (len(vv) - 1) / 2.0)
        bx.plot(vv, jy, ls='none', marker=S.MODEL_MARKER[m], ms=3.8, mfc='white', mec=S.MODEL_COLOUR[m], mew=0.8, zorder=3)
        if sd[0] != 0:
            raise SystemExit('STOP: seed 0 missing in (b)')
        bx.plot(vv[0], jy[0], ls='none', marker=S.MODEL_MARKER[m], ms=3.8, color=S.MODEL_COLOUR[m], zorder=4)
        drawn['b_rows'].append([m, yr, len(vv)])
    bx.set_yticks([3, 2, 1, 0])
    bx.set_yticklabels(['%d' % yr for _, yr in rows])
    FC.house_ticks(bx, 'y')
    bx.set_ylim(-0.6, 3.6)
    b0 = np.ceil((min(allb) - 0.01) / 0.05) * 0.05
    b1 = np.floor((max(allb) + 0.01) / 0.05) * 0.05
    bx.set_xlim(min(allb) - 0.01, max(allb) + 0.01)
    bt = np.round(np.arange(b0, b1 + 1e-09, 0.05), 2)
    bx.set_xticks(bt)
    bx.set_xticklabels(['%.2f' % t for t in bt])
    bx.set_xlabel('Skill score')
    FC.frame(bx)
    FC.tag(bx, 'b', 'Every seed')
    drawn['b_xlim'] = list(bx.get_xlim())
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    yw = max((t.get_window_extent(r).width for t in bx.get_yticklabels()))
    yw_pt = yw * 72.0 / fig.dpi
    gpad = S.TICK_LEN_PT + S.TICK_PAD_PT + yw_pt + 5.0
    bx.set_yticks([2.5, 0.5], minor=True)
    bx.set_yticklabels(['Residual U-Net', 'Reference U-Net'], minor=True)
    bx.tick_params(axis='y', which='minor', length=0, pad=gpad)
    for t, m in zip(bx.get_yticklabels(minor=True), ('M', 'P')):
        t.set_color(ink(m))
    bx.set_ylabel('Model and held-out year')
    dodge = dict(zip(ORDER, np.linspace(-0.26, 0.26, len(ORDER))))
    for m in ORDER[::-1]:
        md = RM['models'][m]
        x = np.array(YEARS, dtype=float) + dodge[m]
        y = np.array(md['y'], dtype=float)
        big = m in ('M', 'P')
        cx.plot(x, y, color=S.MODEL_COLOUR[m], lw=0.9 if big else 0.5, ls=S.MODEL_LS.get(m, '-'), zorder=3 if big else 2)
        cx.plot(x, y, ls='none', marker=S.MODEL_MARKER[m], ms=mk(m), color=S.MODEL_COLOUR[m], mec='white', mew=0.4, zorder=5 if big else 4)
    cx.set_xlim(YEARS[0] - 0.45, YEARS[-1] + 0.45)
    cx.set_xticks(YEARS)
    cx.set_xticklabels([str(y) for y in YEARS])
    allc = [v for m in ORDER for v in RM['models'][m]['y']]
    c0 = np.floor(min(allc) / 0.25) * 0.25
    c1 = np.ceil(max(allc) / 0.25) * 0.25
    cx.set_ylim(c0 - 0.06, c1 + 0.06)
    ct = np.round(np.arange(c0, c1 + 1e-09, 0.25), 2)
    cx.set_yticks(ct)
    cx.set_yticklabels(['%.2f' % t for t in ct])
    cx.set_xlabel('Held-out year')
    cx.set_ylabel('RMSE (ppb)')
    FC.frame(cx)
    FC.tag(cx, 'c', 'RMSE by held-out year')
    drawn['c_dodge_years'] = {m: float(dodge[m]) for m in ORDER}
    dx.set_xlim(X0, X1)
    dx.set_ylim(Y0, Y1)
    th = np.linspace(-np.pi / 2, np.pi / 2, 400)
    rings = [0.5, 0.75, 1.0, 1.25]
    yr_lab = Y1 - 0.025
    ring_ln = []
    trt = blended_transform_factory(dx.transData, dx.transAxes)
    ring_top = {}
    for rr in rings:
        ring_ln += dx.plot(rr * np.cos(th), rr * np.sin(th), color=RING_INK, lw=0.6, zorder=0)
        if rr <= Y1:
            raise SystemExit("STOP: a ring does not meet (d)'s top frame")
        xr_ = float(np.sqrt(rr * rr - Y1 * Y1))
        if not X0 < xr_ < X1:
            raise SystemExit('STOP: a ring meets the top outside (d)')
        ring_top['%.2f' % rr] = round(xr_, 6)
        dx.plot([xr_], [1.0], transform=trt, ls='none', marker=2, ms=S.TICK_LEN_PT, mew=S.TICK_W_PT, color=S.FRAME_INK, clip_on=False, zorder=6)
        dx.annotate('%.2f' % rr, xy=(xr_, 1.0), xycoords=trt, xytext=(0.0, S.TICK_LEN_PT + S.TICK_PAD_PT), textcoords='offset points', color=S.INK, fontsize=S.FS_TICK, ha='center', va='bottom', gid='ringscale')
    fig.canvas.draw()
    _pr = fig.text(0, 0, '0.50', fontsize=S.FS_TICK)
    _ph = _pr.get_window_extent(fig.canvas.get_renderer()).height * 72.0 / fig.dpi
    _pr.remove()
    dx.annotate('Equal RMSE (ppb)', xy=(0.5, 1.0), xycoords='axes fraction', xytext=(0.0, S.TICK_LEN_PT + S.TICK_PAD_PT + _ph + S.AXIS_LABELPAD_PT), textcoords='offset points', color=S.INK, fontsize=S.FS_LABEL, ha='center', va='bottom', gid='ringscale:title')
    drawn['d_ring_scale_top_x_ppb'] = ring_top
    dx.axhline(0, color=S.REF_INK, lw=0.6, zorder=1)
    for m in D4[::-1]:
        dx.plot(ub[m], mb[m], ls='none', marker=S.MODEL_MARKER[m], ms=mk(m), color=S.MODEL_COLOUR[m], mec='white', mew=0.4, zorder=4 if m in ('M', 'P') else 3)
    xt = [0.5, 0.75, 1.0, 1.25]
    dx.set_xticks(xt)
    dx.set_xticklabels(['%.2f' % t for t in xt])
    yt = np.round(np.arange(np.ceil(Y0 / 0.1) * 0.1, Y1 + 1e-09, 0.1), 1)
    dx.set_yticks(yt)
    dx.set_yticklabels([('%.1f' % t).replace('-', S.MINUS) if abs(t) > 1e-09 else '0' for t in yt])
    dx.set_xlabel('Unbiased RMSE (ppb)')
    dx.set_ylabel('Mean error (ppb)')
    FC.frame(dx)
    FC.tag(dx, 'd', 'Mean error and unbiased RMSE', gap_pt=FC.title_gap_over(dx))
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    E = {k: _ext(fig, a_, r) for k, a_ in (('a', ax), ('b', bx), ('c', cx), ('d', dx))}
    gut = (S.GUTTER_PT + 0.4) / 72.0
    bg = S.BLOCK_GAP_PT / 72.0
    xL0 = MARGIN + max(E['a'][0], E['c'][0], E['d'][0])
    xb1 = W - MARGIN - E['b'][1]
    xb0 = xb1 - WB
    xL1 = xb0 - E['b'][0] - gut - E['a'][1]
    wL = xL1 - xL0
    hd = wL * (Y1 - Y0) / (X1 - X0) * D_STRETCH
    d_top1 = MARGIN + max(E['a'][2], E['b'][2])
    d_bot1 = d_top1 + ha_
    t1 = d_bot1 + max(E['a'][3], E['b'][3])
    d_top2 = t1 + bg + E['c'][2]
    d_bot2 = d_top2 + hc
    t2 = d_bot2 + E['c'][3]
    d_top3 = t2 + bg + E['d'][2]
    d_bot3 = d_top3 + hd
    H = d_bot3 + E['d'][3] + MARGIN
    fig.set_size_inches(W, H)
    ax.set_position([xL0 / W, (H - d_bot1) / H, wL / W, ha_ / H])
    bx.set_position([xb0 / W, (H - d_bot1) / H, WB / W, ha_ / H])
    cx.set_position([xL0 / W, (H - d_bot2) / H, wL / W, hc / H])
    dx.set_position([xL0 / W, (H - d_bot3) / H, wL / W, hd / H])
    dx.set_aspect(D_STRETCH)
    drawn['figure_in'] = [W, float(H)]
    drawn['layout_columns_in'] = {'extents_l_r_t_b': {k: [round(float(v), 3) for v in e_] for k, e_ in E.items()}, 'left_frames_x0_x1': [round(xL0, 3), round(xL1, 3)], 'b_frame_x0_x1': [round(xb0, 3), round(xb1, 3)]}
    print('layout columns:', drawn['layout_columns_in'])
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    PX = fig.dpi / 72.0
    fdx = dx.get_window_extent(r)
    for t_ in dx.texts:
        if t_.get_gid() or t_.get_ha() != 'left':
            continue
        if t_.get_window_extent(r).x1 > fdx.x1 - 2.0 * PX:
            rr_ = float(t_.get_text())
            a2_ = np.arcsin((yr_lab + 0.015) / rr_)
            t_.xy = (rr_ * np.cos(a2_), yr_lab)
            t_.set_position((-2.5, 0))
            t_.set_ha('right')
    trb = blended_transform_factory(bx.transAxes, bx.transData)
    fb = bx.get_window_extent(r)
    L_ = (S.TICK_LEN_PT + S.TICK_PAD_PT + yw_pt + 2.0) * PX / fb.width
    bx.add_line(Line2D([0.0, -L_], [1.5, 1.5], transform=trb, color=S.FRAME_INK, lw=S.TICK_W_PT, clip_on=False))
    hk = []
    for m in ORDER:
        hk.append(Line2D([0], [0], color=S.MODEL_COLOUR[m], lw=0.9 if m in ('M', 'P') else 0.5, ls=S.MODEL_LS.get(m, '-'), marker=S.MODEL_MARKER[m], ms=mk(m), mec='white', mew=0.4))
    fc_ = cx.get_window_extent(r)
    fd_ = dx.get_window_extent(r)
    ycen = 0.5 * (fc_.y1 + fd_.y0) / fig.dpi
    leg = fig.legend(hk, [NAME[m] for m in ORDER], loc='center right', bbox_to_anchor=(xb1 / W, ycen / H), bbox_transform=fig.transFigure, fontsize=S.KEY_FS, frameon=False, borderpad=0.0, borderaxespad=0.0, handlelength=S.KEY_HANDLELENGTH_EM, handletextpad=S.KEY_TEXTPAD_EM, labelspacing=S.KEY_ROWSPACE_EM)
    for t, m in zip(leg.get_texts(), ORDER):
        t.set_color(ink(m))
    drawn['key'] = [NAME[m] for m in ORDER]
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    kb = leg.get_window_extent(r)
    ctb = Bbox.union([cx.get_tightbbox(r), dx.get_tightbbox(r)])
    key_rec = {'side': 'right of (c) and (d)', 'centre_offset_pt_vertical': round(float((0.5 * (kb.y0 + kb.y1) - 0.5 * (fc_.y1 + fd_.y0)) / PX), 2), 'right_edge_minus_b_frame_right_pt': round(float((kb.x1 - bx.get_window_extent(r).x1) / PX), 2), 'gap_to_c_d_tight_pt': round(float((kb.x0 - ctb.x1) / PX), 2)}
    if key_rec['gap_to_c_d_tight_pt'] < 2 * S.KEY_GAP_PT:
        raise SystemExit('STOP: the key crowds (c) and (d) %r' % key_rec)
    leg0 = fig.legend([Line2D([0], [0], color=S.REF_INK, lw=0.6)], ['Zero mean error'], loc='upper left', bbox_to_anchor=(kb.x0 / fig.dpi / W, (kb.y0 / fig.dpi - S.KEY_GAP_PT / 72.0) / H), bbox_transform=fig.transFigure, fontsize=S.KEY_FS, frameon=False, borderpad=0.0, borderaxespad=0.0, handlelength=S.KEY_HANDLELENGTH_EM, handletextpad=S.KEY_TEXTPAD_EM)
    fig.canvas.draw()
    kb0 = leg0.get_window_extent(fig.canvas.get_renderer())
    key_rec['zero_key_gap_pt'] = round(float((kb.y0 - kb0.y1) / PX), 2)
    key_rec['zero_key_left_minus_key_left_pt'] = round(float((kb0.x0 - kb.x0) / PX), 2)
    drawn['key_zero'] = 'Zero mean error'
    zx0, zx1, zy0, zy1 = ZOOM
    ratio = (zx1 - zx0) / (zy1 - zy0)
    allp = [(m, j) for m in D4 for j in range(5)]
    inbox = {(m, j): zx0 < ub[m][j] < zx1 and zy0 < mb[m][j] < zy1 for m, j in allp}
    R_MK = 2.6 * PX
    TT, TM = (3.0 * PX, 2.0 * PX)
    mk_d = np.array([dx.transData.transform((ub[m][j], mb[m][j])) for m, j in allp])
    ins = fig.add_axes([0.5, 0.1, 0.1, 0.1])
    ins.set_xlim(zx0, zx1)
    ins.set_ylim(zy0, zy1)
    ring_in = []
    for rr in rings:
        ring_in += ins.plot(rr * np.cos(th), rr * np.sin(th), color=RING_INK, lw=0.6, zorder=0)
    ins.axhline(0, color=S.REF_INK, lw=0.6, zorder=1)
    for m in D4[::-1]:
        ins.plot(ub[m], mb[m], ls='none', marker=S.MODEL_MARKER[m], ms=mk(m), color=S.MODEL_COLOUR[m], mec='white', mew=0.4, zorder=4 if m in ('M', 'P') else 3)
    ixt = [v for v in np.round(np.arange(0.4, 0.7, 0.05), 2) if zx0 <= v <= zx1]
    ins.set_xticks(ixt)
    ins.set_xticklabels(['%.2f' % t for t in ixt])
    iyt = [v for v in np.round(np.arange(-0.1, 0.1, 0.02), 2) if zy0 <= v <= zy1]
    ins.set_yticks(iyt)
    ins.set_yticklabels([('%.2f' % t).replace('-', S.MINUS) if abs(t) > 1e-09 else '0.00' for t in iyt])
    ins.yaxis.tick_right()
    ins.set_xlim(zx0, zx1)
    ins.set_ylim(zy0, zy1)
    FC.frame(ins)
    FC.house_ticks(ins, 'both')
    ins.tick_params(axis='both', labelsize=S.FS_TICK)
    for t_ in ins.get_xticklabels() + ins.get_yticklabels():
        t_.set_bbox(dict(fc='white', ec='none', pad=0.4))
    ins.set_zorder(6)
    zrect_d = Bbox([dx.transData.transform((zx0, zy0)), dx.transData.transform((zx1, zy1))])
    fdd = dx.get_window_extent(r)
    best = None
    tried = 0
    rej = {}
    for iy1 in np.arange(-0.055, -0.095, -0.005):
        for ix1 in np.arange(1.17, 0.94, -0.01):
            for iy0 in np.arange(Y0 + 0.02, Y0 + 0.13, 0.0025):
                h_ = iy1 - iy0
                w_ = ratio * h_
                ix0 = ix1 - w_
                p0 = dx.transData.transform((ix0, iy0))
                p1 = dx.transData.transform((ix1, iy1))
                tried += 1
                ins.set_position([p0[0] / fig.dpi / W, p0[1] / fig.dpi / H, (p1[0] - p0[0]) / fig.dpi / W, (p1[1] - p0[1]) / fig.dpi / H])
                tb = ins.get_tightbbox(r)
                if tb.x0 < fdd.x0 + 2 * PX or tb.x1 > fdd.x1 - 2 * PX or tb.y0 < fdd.y0 + 2 * PX or (tb.y1 > fdd.y1 - 2 * PX):
                    rej['frame'] = rej.get('frame', 0) + 1
                    continue
                dmk = min((_box_pt_d(tb, p) for p in mk_d))
                if dmk < R_MK + 2.0 * PX:
                    rej['marks'] = rej.get('marks', 0) + 1
                    continue
                if any((tb.overlaps(t_.get_window_extent(r)) for t_ in dx.texts)):
                    rej['texts'] = rej.get('texts', 0) + 1
                    continue
                fr_i = ins.get_window_extent(r)
                itl = [t_.get_window_extent(r) for t_ in ins.get_xticklabels() + ins.get_yticklabels() if t_.get_visible() and t_.get_text().strip()]
                bc = {'BL': np.array([zrect_d.x0, zrect_d.y0]), 'BR': np.array([zrect_d.x1, zrect_d.y0]), 'TR': np.array([zrect_d.x1, zrect_d.y1]), 'TL': np.array([zrect_d.x0, zrect_d.y1])}
                ic = {'BL': np.array([fr_i.x0, fr_i.y0]), 'BR': np.array([fr_i.x1, fr_i.y0]), 'TR': np.array([fr_i.x1, fr_i.y1]), 'TL': np.array([fr_i.x0, fr_i.y1])}
                cons = []
                for kb_, q0 in bc.items():
                    for ki, q1 in ic.items():
                        if _seg_hits_box(q0, q1, zrect_d) or _seg_hits_box(q0, q1, fr_i) or any((_seg_hits_box(q0, q1, b_, 1.0 * PX) for b_ in itl)):
                            continue
                        cl = min((_seg_d(q0, q1, p) for k_, p in enumerate(mk_d) if not inbox[allp[k_]]))
                        cons.append((cl, kb_, ki, q0, q1))
                pair = None
                for i_ in range(len(cons)):
                    for j_ in range(i_ + 1, len(cons)):
                        c1_, c2_ = (cons[i_], cons[j_])
                        if c1_[1] == c2_[1] or c1_[2] == c2_[2]:
                            continue
                        if _crosses(c1_[3], c1_[4], c2_[3], c2_[4]):
                            continue
                        cl = min(c1_[0], c2_[0])
                        if pair is None or cl > pair[0]:
                            pair = (cl, c1_, c2_)
                if pair is None or pair[0] < R_MK + 1.0 * PX:
                    rej['connectors'] = rej.get('connectors', 0) + 1
                    continue
                score = (round(h_, 3), pair[0])
                if best is None or score > best[0]:
                    best = (score, (ix0, ix1, iy0, iy1), pair, dmk)
    if best is None:
        raise SystemExit('STOP: no clear place for the inset of (d) %r' % rej)
    ix0, ix1, iy0, iy1 = best[1]
    p0 = dx.transData.transform((ix0, iy0))
    p1 = dx.transData.transform((ix1, iy1))
    ins.set_position([p0[0] / fig.dpi / W, p0[1] / fig.dpi / H, (p1[0] - p0[0]) / fig.dpi / W, (p1[1] - p0[1]) / fig.dpi / H])
    dx.add_patch(Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, fill=False, ec=LEAD_INK, lw=BOX_LW, zorder=5))
    inv_f = fig.transFigure.inverted()
    con_pts = []
    for c_ in best[2][1:]:
        q0, q1 = (c_[3], c_[4])
        f0, f1 = (inv_f.transform(q0), inv_f.transform(q1))
        fig.add_artist(Line2D([f0[0], f1[0]], [f0[1], f1[1]], transform=fig.transFigure, color=LEAD_INK, lw=LEAD_LW, zorder=5))
        con_pts.append((q0, q1))
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ib_ = ins.get_tightbbox(r).padded(2.0 * PX)
    for ln_ in ring_ln:
        xy_ = dx.transData.transform(np.asarray(ln_.get_xydata(), float))
        in_ = (xy_[:, 0] > ib_.x0) & (xy_[:, 0] < ib_.x1) & (xy_[:, 1] > ib_.y0) & (xy_[:, 1] < ib_.y1)
        yd_ = np.asarray(ln_.get_ydata(), float).copy()
        yd_[in_] = np.nan
        ln_.set_ydata(yd_)
    mag = (p1[0] - p0[0]) / (zrect_d.x1 - zrect_d.x0)
    drawn['layout_d_zoom'] = {'box_ppb': list(ZOOM), 'inset_in_d_ppb': [float(v) for v in best[1]], 'magnification': round(float(mag), 3), 'connectors': [best[2][1][1] + '-' + best[2][1][2], best[2][2][1] + '-' + best[2][2][2]], 'connector_to_mark_min_pt': round(float(best[2][0] / PX), 2), 'inset_to_mark_min_pt': round(float(best[3] / PX), 2), 'places_tried': tried, 'places_rejected': rej}
    print('zoom chosen:', drawn['layout_d_zoom'])
    groups = {}
    for m in D4:
        gl = []
        for j, y in enumerate(YEARS):
            for g in gl:
                k = YEARS.index(g[0])
                if np.hypot(ub[m][j] - ub[m][k], mb[m][j] - mb[m][k]) < GROUP_PPB:
                    g.append(y)
                    break
            else:
                gl.append([y])
        groups[m] = gl
    drawn['d_label_groups'] = groups
    for m in D4:
        for g in groups[m]:
            sides = set((inbox[m, YEARS.index(y)] for y in g))
            if len(sides) != 1:
                raise SystemExit('STOP: a label group split by the box')
    DIRS = {'GBT': 0.0, 'P': 180.0, 'M': 270.0, 'CLIM-WD': 0.0}
    SWEEP = [0] + [s_ * k_ for k_ in range(15, 181, 15) for s_ in (1, -1)]

    def axis_ctx(axx, in_inset):
        fr = axx.get_window_extent(r)
        mkp = np.array([axx.transData.transform((ub[m][j], mb[m][j])) for m, j in allp])
        lo_, hi_ = axx.get_xlim()
        lp = [axx.transData.transform((xz, 0.0)) for xz in np.linspace(lo_, hi_, 400)]
        th2 = np.linspace(-np.pi / 2, np.pi / 2, 1600)
        ibx = ins.get_tightbbox(r).padded(2.0 * PX)
        rp = []
        for rr in [] if in_inset else rings:
            for q in axx.transData.transform(np.column_stack([rr * np.cos(th2), rr * np.sin(th2)])):
                if fr.x0 <= q[0] <= fr.x1 and fr.y0 <= q[1] <= fr.y1 and (in_inset or not (ibx.x0 < q[0] < ibx.x1 and ibx.y0 < q[1] < ibx.y1)):
                    rp.append(q)
        placed = []
        if not in_inset:
            placed = [t_.get_window_extent(r) for t_ in dx.texts]
            placed.append(ins.get_tightbbox(r))
            for q0, q1 in con_pts:
                lp.extend(_seg_pts(q0, q1, 80))
            z = zrect_d
            for q0, q1 in (((z.x0, z.y0), (z.x1, z.y0)), ((z.x1, z.y0), (z.x1, z.y1)), ((z.x1, z.y1), (z.x0, z.y1)), ((z.x0, z.y1), (z.x0, z.y0))):
                lp.extend(_seg_pts(np.array(q0), np.array(q1), 80))
        vis = []
        for k_, (m, j) in enumerate(allp):
            p = mkp[k_]
            if fr.x0 - 1 <= p[0] <= fr.x1 + 1 and fr.y0 - 1 <= p[1] <= fr.y1 + 1:
                vis.append(k_)
        return {'axx': axx, 'fr': fr, 'mkp': mkp, 'lp': np.array(lp), 'rp': np.array(rp).reshape(-1, 2), 'placed': placed, 'vis': vis, 'leaders': [], 'in_inset': in_inset, 'zone': None if in_inset else zrect_d}

    def box_at(p, u, w, h, dist):
        lo_, hi_ = (0.0, dist + w + h)
        for _ in range(40):
            t_ = 0.5 * (lo_ + hi_)
            dxx = max(abs(u[0] * t_) - w / 2.0, 0.0)
            dyy = max(abs(u[1] * t_) - h / 2.0, 0.0)
            if np.hypot(dxx, dyy) < dist:
                lo_ = t_
            else:
                hi_ = t_
        c = p + u * hi_
        return Bbox([[c[0] - w / 2.0, c[1] - h / 2.0], [c[0] + w / 2.0, c[1] + h / 2.0]])

    def clear(ctx, bb, lead, own):
        fr = ctx['fr']
        if bb.x0 < fr.x0 + 2 * PX or bb.x1 > fr.x1 - 2 * PX or bb.y0 < fr.y0 + 2 * PX or (bb.y1 > fr.y1 - 2 * PX):
            return False
        if any((bb.padded(TT).overlaps(p) for p in ctx['placed'])):
            return False
        if ctx['zone'] is not None and bb.padded(TM).overlaps(ctx['zone']):
            return False
        e = bb.padded(TM)
        lp = ctx['lp']
        if len(lp) and np.any((lp[:, 0] > e.x0) & (lp[:, 0] < e.x1) & (lp[:, 1] > e.y0) & (lp[:, 1] < e.y1)):
            return False
        e2 = bb.padded(1.05 * PX)
        rp = ctx['rp']
        if len(rp) and np.any((rp[:, 0] > e2.x0) & (rp[:, 0] < e2.x1) & (rp[:, 1] > e2.y0) & (rp[:, 1] < e2.y1)):
            return False
        for k_ in ctx['vis']:
            if _box_pt_d(bb, ctx['mkp'][k_]) < R_MK + TM:
                return False
        q0, q1 = lead
        for p in ctx['placed']:
            if _seg_hits_box(q0, q1, p, TM * 0.5):
                return False
        for k_ in ctx['vis']:
            if k_ in own:
                continue
            if _seg_d(q0, q1, ctx['mkp'][k_]) < R_MK + 0.5 * PX:
                return False
        if ctx['zone'] is not None and _seg_hits_box(q0, q1, ctx['zone']):
            return False
        if any((_crosses(q0, q1, a0, a1) for a0, a1 in ctx['leaders'])):
            return False
        return True

    def crowd(m, grp):
        p = mk_d[D4.index(m) * 5 + YEARS.index(grp[0])]
        return int(np.sum(np.hypot(*(mk_d - p).T) < 14 * PX))
    items = [(m, g) for m in D4 for g in groups[m]]
    items.sort(key=lambda it: -crowd(*it))
    made = []
    for attempt in range(8):
        for a_ in made:
            a_.remove()
        made = []
        ctxs = {False: axis_ctx(dx, False), True: axis_ctx(ins, True)}
        unplaced = []
        rec = []
        for m, grp in items:
            j = YEARS.index(grp[0])
            ctx = ctxs[inbox[m, j]]
            axx = ctx['axx']
            own = set((D4.index(m) * 5 + YEARS.index(y) for y in grp))
            s_ = ', '.join(('%02d' % (y % 100) for y in grp))
            txt_ = s_
            if len(grp) > 2 and grp == list(range(grp[0], grp[-1] + 1)):
                txt_ = '%02d–%02d' % (grp[0] % 100, grp[-1] % 100)
            p = ctx['mkp'][D4.index(m) * 5 + j]
            t = axx.text(0, 0, txt_, fontsize=S.FS_TICK, color=ink(m), ha='center', va='center', zorder=7)
            t.set_position(axx.transData.inverted().transform(p))
            tb0 = t.get_window_extent(r)
            w, h = (tb0.width, tb0.height)
            done = False
            for dist in (LABEL_DIST_PT, LABEL_DIST_PT + 3.0, LABEL_DIST_PT + 6.0, LABEL_DIST_PT + 9.0, LABEL_DIST_PT + 13.0, LABEL_DIST_PT + 18.0, LABEL_DIST_PT + 24.0):
                for sw in SWEEP:
                    ang = np.deg2rad(DIRS[m] + sw)
                    u = np.array([np.cos(ang), np.sin(ang)])
                    bb = box_at(p, u, w, h, dist * PX)
                    q = np.array([np.clip(p[0], bb.x0, bb.x1), np.clip(p[1], bb.y0, bb.y1)])
                    v = (q - p) / max(np.hypot(*q - p), 1e-09)
                    lead = (p + v * (R_MK + 0.6 * PX), q - v * 0.8 * PX)
                    if clear(ctx, bb, lead, own):
                        c = np.array([(bb.x0 + bb.x1) / 2.0, (bb.y0 + bb.y1) / 2.0])
                        t.set_position(axx.transData.inverted().transform(c))
                        inv = axx.transData.inverted()
                        d0, d1 = (inv.transform(lead[0]), inv.transform(lead[1]))
                        ln_, = axx.plot([d0[0], d1[0]], [d0[1], d1[1]], color=LEAD_INK, lw=LEAD_LW, zorder=6, solid_capstyle='butt')
                        made.extend([t, ln_])
                        ctx['placed'].append(t.get_window_extent(r))
                        ctx['leaders'].append(lead)
                        ctx['lp'] = np.vstack([ctx['lp'], np.array(_seg_pts(*lead, 30))])
                        rec.append({'m': m, 'years': grp, 'in_inset': ctx['in_inset'], 'dist_pt': dist, 'angle_deg': float(DIRS[m] + sw), 'box': bb, 'lead': lead})
                        done = True
                        break
                if done:
                    break
            if not done:
                t.remove()
                unplaced.append([m, s_])
        if not unplaced:
            break
        bad = [it for it in items if [it[0], ', '.join(('%02d' % (y % 100) for y in it[1]))] in unplaced]
        items = bad + [it for it in items if it not in bad]
    drawn['d_label_layout_order'] = [[m, g] for m, g in items]
    drawn['d_label_layout_attempts'] = attempt + 1
    if unplaced:
        raise SystemExit('STOP: no clear place for year labels %r' % unplaced)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ilb = [t_.get_window_extent(r).padded(2.0 * PX) for t_ in ins.texts]
    th3 = np.linspace(-np.pi / 2, np.pi / 2, 4000)
    for ln_, rr in zip(ring_in, rings):
        xd_, yd_ = (rr * np.cos(th3), rr * np.sin(th3))
        q_ = ins.transData.transform(np.column_stack([xd_, yd_]))
        cut_ = np.zeros(len(q_), bool)
        for b_ in ilb:
            cut_ |= (q_[:, 0] > b_.x0) & (q_[:, 0] < b_.x1) & (q_[:, 1] > b_.y0) & (q_[:, 1] < b_.y1)
        idx_ = np.flatnonzero(cut_)
        if len(idx_):
            for a_, b_ in zip(idx_[:-1], idx_[1:]):
                if b_ - a_ > 1:
                    seg_ = q_[a_:b_ + 1]
                    ln_pt = np.sum(np.hypot(*np.diff(seg_, axis=0).T)) / PX
                    if ln_pt < 6.0:
                        cut_[a_:b_] = True
        yd_ = yd_.copy()
        yd_[cut_] = np.nan
        ln_.set_data(xd_, yd_)
    drawn['d_inset_ring_breaks_pt'] = 2.0
    fig.canvas.draw()
    lb = [x['box'] for x in rec]
    tt = min((np.hypot(max(lb[i_].x0 - lb[j_].x1, lb[j_].x0 - lb[i_].x1, 0.0), max(lb[i_].y0 - lb[j_].y1, lb[j_].y0 - lb[i_].y1, 0.0)) for i_ in range(len(lb)) for j_ in range(i_ + 1, len(lb)) if rec[i_]['in_inset'] == rec[j_]['in_inset']))
    nx = sum((1 for i_, a in enumerate(rec) for b in rec[i_ + 1:] if a['in_inset'] == b['in_inset'] and _crosses(a['lead'][0], a['lead'][1], b['lead'][0], b['lead'][1])))
    dists = sorted(set((x['dist_pt'] for x in rec)))
    lens = [float(np.hypot(*x['lead'][1] - x['lead'][0]) / PX) for x in rec]
    drawn['d_label_layout_clearances_pt'] = {'text_to_text_min': round(float(tt / PX), 2), 'label_distances_pt_used': dists, 'n_at_house_distance': sum((1 for x in rec if x['dist_pt'] == LABEL_DIST_PT)), 'n_labels': len(rec), 'n_in_inset': sum((1 for x in rec if x['in_inset'])), 'leader_length_pt_min_max': [round(min(lens), 2), round(max(lens), 2)], 'leaders': len(rec), 'leader_crossings': int(nx), 'rule': "every label at LABEL_DIST_PT from its mark's centre to its box (a longer step only where no place is clear, recorded), a thin gray leader from the mark's edge; 1.5 pt text to text, 2.0 pt text to mark and line; mark radius 2.6 pt"}
    print('labels:', drawn['d_label_layout_clearances_pt'])
    print('zoom:', drawn['layout_d_zoom'])
    print('key:', key_rec)
    drawn['layout_key'] = key_rec
    drawn['d_limits'] = [X0, X1, float(Y0), float(Y1)]
    drawn['d_rings_ppb'] = rings
    S.register_block(fig, 'a and b', [ax, bx])
    S.register_block(fig, 'c', [cx])
    S.register_block(fig, 'd', [dx])
    new = {'unbiased_rmse_ppb': ub, 'd_mean_bias_ppb_as_read': mb, 'rule': "sqrt(RMSE^2 - MB^2) of the stored metrics.json rows (stratum all, seed 0 or the baseline's fit), positions of (d) only"}
    return (fig, drawn, new)
