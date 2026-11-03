"""The drawing parts of figures fig_F3 and fig_A_F1: the city panels of daily series, the month labels and one key."""
import numpy as np
from matplotlib.lines import Line2D
import build_fig_common as C
import build_style as S
import matplotlib.pyplot as plt
W = 6.5
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
CMAQ_LW = 2.0
EMU_LW = 0.7
STRIP_LW = 0.45
STRIP_ALPHA = 0.3
LETTERS = 'abcdefghijklmnopqrst'

def months(year):
    st = C.month_starts(year) + [367 if year % 4 == 0 else 366]
    st = np.array(st, dtype=float)
    return (st[:-1], 0.5 * (st[:-1] + st[1:]))

def rect(H, x, ytop, w, h):
    return [x / W, (H - ytop - h) / H, w / W, h / H]

def key(fig, H, items, ncol=None):
    hs = [h for _, h in items]
    labs = [lab for lab, _ in items]
    leg = fig.legend(hs, labs, loc='upper center', bbox_to_anchor=(0.5, (H - 0.08) / H), ncol=ncol or len(items), fontsize=S.FS_LABEL, frameon=False, handlelength=2.6, columnspacing=2.2, handletextpad=0.6, borderaxespad=0.0)
    return leg

def h_cmaq():
    return Line2D([], [], color=S.CMAQ_LEVEL, lw=CMAQ_LW + 1.0, solid_capstyle='butt')

def h_model(m, line=True, marker=True):
    return Line2D([], [], color=S.MODEL_COLOUR[m], lw=1.0 if line else 0, linestyle=S.MODEL_LS[m] if line else 'none', marker=S.MODEL_MARKER[m] if marker else None, ms=3.6, mec='white', mew=0.3, solid_capstyle='butt', dash_capstyle='butt')

def level_axes(ax, doy, cm, em, m, B, mstart):
    x, (yc, ye) = C.with_gaps(doy, cm, em)
    ax.plot(x, yc, color=S.CMAQ_LEVEL, lw=CMAQ_LW, solid_capstyle='butt', zorder=2)
    ax.plot(x, ye, color=S.MODEL_COLOUR[m], lw=EMU_LW, linestyle=S.MODEL_LS[m], zorder=3)
    ax.set_xlim(0.5, 366.5)
    ax.set_ylim(0, B)
    ax.set_yticks(np.arange(0, B + 1, 20))
    ax.set_xticks(mstart)
    ax.tick_params(axis='x', length=S.TICK_LEN_PT, width=S.TICK_W_PT, labelbottom=False)
    C.frame(ax)

def month_labels(ax, mmid):
    ax.set_xticks(mmid, minor=True)
    ax.set_xticklabels(MONTHS, minor=True)
    ax.tick_params(axis='x', which='minor', length=0, labelbottom=True, pad=S.TICK_LEN_PT + S.TICK_PAD_PT)

def inward_edge_labels(ax):
    lo, hi = sorted(ax.get_xlim())
    span = hi - lo
    for tk, loc in zip(ax.xaxis.get_major_ticks(), ax.get_xticks()):
        if abs(loc - lo) <= 1e-09 * span:
            tk.label1.set_horizontalalignment('left')
        elif abs(loc - hi) <= 1e-09 * span:
            tk.label1.set_horizontalalignment('right')

def diffs_f3(arr, cities):
    return {c: {m: arr[m, c][1] - arr[m, c][0] for m in ('P', 'M')} for c in cities}

def draw_f3(arr, rows, doy, B, plotted, year, h_fig=None):
    cities = S.CITY_ORDER
    mstart, mmid = months(year)
    diff = diffs_f3(arr, cities)
    smax = float(np.ceil(max((np.abs(d).max() for c in cities for d in diff[c].values()))))
    plotted['daily_diff_unet_minus_cmaq'] = {'rule': 'row 1 minus row 0 of F5_city_fold2016_<M|P>_<city>.npy (the U-Net minus CMAQ) on each of the 339 test days', 'values': {c: {m: diff[c][m] for m in ('P', 'M')} for c in cities}, 'strip_half_range_ppb': smax, 'strip_half_range_rule': 'ceil of the largest absolute daily difference of either U-Net in any city'}
    X0, XR, GAPX = (0.62, 0.62, 0.5)
    CW = (W - X0 - XR - GAPX) / 2.0
    H_LEV, H_GAP, H_STR, H_ROW = (0.52, 0.13, 0.4, 0.27)
    TOP_AX = 0.5
    blk = 5 * (H_LEV + H_GAP + H_STR) + 4 * H_ROW
    BOT = 0.1
    H = TOP_AX + blk + 0.5 if h_fig is None else h_fig
    fig = plt.figure(figsize=(W, H))
    lev, stp = ({}, {})
    for i, c in enumerate(cities):
        col, row = (i % 2, i // 2)
        x = X0 + col * (CW + GAPX)
        ytop = TOP_AX + row * (H_LEV + H_GAP + H_STR + H_ROW)
        ax = fig.add_axes(rect(H, x, ytop, CW, H_LEV))
        sx = fig.add_axes(rect(H, x, ytop + H_LEV + H_GAP, CW, H_STR))
        lev[c], stp[c] = (ax, sx)
        level_axes(ax, doy, arr['M', c][0], arr['M', c][1], 'M', B, mstart)
        for m in ('P', 'M'):
            xx, (dd,) = C.with_gaps(doy, diff[c][m])
            sx.plot(xx, dd, color=S.MODEL_COLOUR[m], lw=STRIP_LW, linestyle=S.MODEL_LS[m])
        sx.axhline(0, color=S.REF_INK, lw=0.6, zorder=4)
        sx.set_ylim(-smax, smax)
        sx.set_yticks([-10, 0, 10])
        sx.set_yticklabels([S.MINUS + '10', '0', '10'])
        sx.yaxis.tick_right()
        sx.set_xlim(0.5, 366.5)
        sx.set_xticks(mstart)
        sx.tick_params(axis='x', length=S.TICK_LEN_PT, width=S.TICK_W_PT, labelbottom=False)
        if row == 4:
            month_labels(sx, mmid)
        C.frame(sx)
        C.tag(ax, LETTERS[i], S.CITY_NAME[c])
    block = list(lev.values()) + list(stp.values())
    C.block_title(fig, block, 'City daily mean NO$_2$ (ppb)', side='left')
    rt_ = C.block_title(fig, block, 'U-Net minus CMAQ (ppb)', side='right')
    fig.canvas.draw()
    rr = fig.canvas.get_renderer()
    l2 = max((tk.label2.get_window_extent(rr).x1 for sx_ in stp.values() for tk in sx_.yaxis.get_major_ticks() if tk.label2.get_visible() and tk.label2.get_text()))
    got = (rt_.get_window_extent(rr).x0 - l2) * 72.0 / fig.dpi
    x_, y_ = rt_.get_position()
    rt_.set_position((x_ + (S.AXIS_LABELPAD_PT - got) / 72.0 / W, y_))
    C.block_title(fig, [stp[cities[8]], stp[cities[9]]], 'Month of %d' % year, axis='x')
    hm = {m: h_model(m, marker=False) for m in ('M', 'P')}
    zero = Line2D([], [], color=S.REF_INK, lw=0.6)
    legs = C.house_key(fig, block, [h_cmaq(), hm['P'], hm['M'], zero], ['CMAQ', S.PLAIN['P'], S.PLAIN['M'], 'Zero difference'], side='above')
    leg = legs[0]
    for t, col in zip(leg.get_texts(), (S.INK, S.MODEL_TEXT['P'], S.MODEL_TEXT['M'], S.INK)):
        t.set_color(col)
    if h_fig is None:
        fig.canvas.draw()
        rr = fig.canvas.get_renderer()
        mt = [t for t in fig.texts if t.get_text().startswith('Month')][0]
        y0_in = mt.get_window_extent(rr).y0 / fig.dpi
        plt.close(fig)
        return draw_f3(arr, rows, doy, B, plotted, year, h_fig=H - (y0_in - BOT))
    return (fig, H, {'legend': leg, 'first_tag': lev[cities[0]].texts[0], 'row1': lev[cities[0]], 'row2_tag': lev[cities[2]].texts[0], 'strip1': stp[cities[0]], 'month': [t for t in fig.texts if t.get_text().startswith('Month')][0]})

def draw_f3k(arr, rows, plotted, year):
    cities = S.CITY_ORDER
    diff = diffs_f3(arr, cities)
    pct = {c: {m: np.percentile(diff[c][m], [10, 90]).tolist() for m in ('P', 'M')} for c in cities}
    plotted['p10_p90_by_city'] = {'rule': "numpy.percentile (linear) at 10 and 90 of each U-Net's 339 daily differences per city", 'values': pct}
    order = sorted(cities, key=lambda c: -float(np.mean(arr['M', c][0])))
    plotted['k_row_order'] = {'rule': "cities by CMAQ's city mean over the 339 test days, highest first (ordering only; the means not stored)", 'values': order}
    plotted['k_means_printed'] = {'rule': 'city_series.json MB (seed 0, fold 2016), as in Table A1; the positions of the marks; not printed (Table A1 prints the same values)', 'values': {c: {m: rows[m, c]['MB'] for m in ('P', 'M')} for c in cities}}
    plotted['k_means_drawn'] = {'rule': "numpy.mean of each U-Net's 339 daily differences per city (the x of each square and disc)", 'values': {c: {m: float(np.mean(diff[c][m])) for m in ('P', 'M')} for c in cities}}
    XR = 0.62
    TOP_AX = 0.5
    H_K = 2.6
    H = TOP_AX + H_K + 0.44
    fig = plt.figure(figsize=(W, H))
    KX0, KX1 = (1.02, W - XR)
    kx = fig.add_axes(rect(H, KX0, TOP_AX, KX1 - KX0, H_K))
    n = len(order)
    lo, hi = (0.0, 0.0)
    for r, c in enumerate(order):
        yb = n - 1 - r
        mean = {m: float(np.mean(diff[c][m])) for m in ('P', 'M')}
        dy = {'P': 0.2, 'M': -0.2}
        for m in ('P', 'M'):
            q = pct[c][m]
            lo, hi = (min(lo, q[0]), max(hi, q[1]))
            kx.plot(q, [yb + dy[m]] * 2, color=S.MODEL_COLOUR[m], lw=2.6, alpha=0.25, solid_capstyle='butt', zorder=2)
            for qq in q:
                kx.plot([qq, qq], [yb + dy[m] - 0.12, yb + dy[m] + 0.12], color=S.MODEL_COLOUR[m], lw=0.8, zorder=3)
        kx.plot([mean['P'], mean['M']], [yb + dy['P'], yb + dy['M']], color='#8A8A8A', lw=0.6, zorder=3)
        for m in ('P', 'M'):
            kx.plot(mean[m], yb + dy[m], S.MODEL_MARKER[m], ms=4.2, color=S.MODEL_COLOUR[m], mec='white', mew=0.4, zorder=4)
    kx.axvline(0, color=S.REF_INK, lw=0.6, zorder=1)
    kx.set_ylim(-0.6, n - 0.4)
    kx.set_yticks(np.arange(n)[::-1])
    kx.set_yticklabels([S.CITY_NAME[c] for c in order])
    lim = float(np.floor(max(-lo, hi)))
    span = 1.04 * max(-lo, hi)
    kx.set_xlim(-span, span)
    kx.set_xticks(np.arange(-lim, lim + 1))
    plotted['k_x_range'] = {'xlim': [-span, span], 'xticks': np.arange(-lim, lim + 1).tolist()}
    kx.set_xlabel('U-Net minus CMAQ (ppb)')
    kx.set_ylabel('City')
    C.frame(kx)
    C.house_ticks(kx, 'both')
    from matplotlib.transforms import offset_copy
    base = offset_copy(kx.transAxes, fig=fig, x=0.0, y=S.TITLE_GAP_PT, units='points')
    title = kx.text(0.0, 1.0, 'Mean error (U-Net minus CMAQ), %d' % year, transform=base, ha='left', va='bottom', fontsize=8, color=S.INK, gid='tagline:%d' % id(kx))
    hk = {m: (Line2D([], [], color=S.MODEL_COLOUR[m], lw=2.6, alpha=0.25, solid_capstyle='butt'), Line2D([], [], color=S.MODEL_COLOUR[m], lw=0, linestyle='none', marker=S.MODEL_MARKER[m], ms=4.2, mec='white', mew=0.4)) for m in ('P', 'M')}
    zero = Line2D([], [], color=S.REF_INK, lw=0.6)
    legs = C.house_key(fig, [kx], [hk['P'], hk['M'], zero], [S.PLAIN['P'], S.PLAIN['M'], 'Zero difference'], side='above')
    leg = legs[0]
    for t, col in zip(leg.get_texts(), (S.MODEL_TEXT['P'], S.MODEL_TEXT['M'], S.INK)):
        t.set_color(col)
    return (fig, H, {'legend': leg, 'title': title, 'kx': kx})

def draw_a3(arr, doy, B, plotted, year, s_top=None):
    cities = S.CITY_ORDER
    mstart, mmid = months(year)
    plotted['scatter_residual_unet_series'] = {'rule': 'row 1 of F5_city_fold2016_M_<city>.npy (the values of fig_F3 panels/<city>/model), drawn in (k)-(t)', 'values': {c: arr['M', c][1] for c in cities}}
    X0, XR, GAPX = (0.62, 0.16, 0.46)
    CW = (W - X0 - XR - GAPX) / 2.0
    H_LEV, H_ROW = (0.39, 0.27)
    TOP_AX = 0.5
    blk = 5 * H_LEV + 4 * H_ROW
    SG = 0.14
    SW = (W - X0 - XR - 4 * SG) / 5.0
    S_GAP = 0.3
    MO = 0.34
    S_TOP = TOP_AX + blk + MO + 0.26 if s_top is None else s_top
    H = S_TOP + 2 * SW + S_GAP + 0.44
    fig = plt.figure(figsize=(W, H))
    levs = []
    for i, c in enumerate(cities):
        col, row = (i % 2, i // 2)
        x = X0 + col * (CW + GAPX)
        ytop = TOP_AX + row * (H_LEV + H_ROW)
        ax = fig.add_axes(rect(H, x, ytop, CW, H_LEV))
        levs.append(ax)
        level_axes(ax, doy, arr['P', c][0], arr['P', c][1], 'P', B, mstart)
        if row == 4:
            month_labels(ax, mmid)
        C.tag(ax, LETTERS[i], S.CITY_NAME[c])
    C.block_title(fig, levs, 'City daily mean NO$_2$ (ppb)', axis='y')
    C.block_title(fig, [levs[8], levs[9]], 'Month of %d' % year, axis='x')
    scs = []
    for i, c in enumerate(cities):
        col, row = (i % 5, i // 5)
        x = X0 + col * (SW + SG)
        ytop = S_TOP + row * (SW + S_GAP)
        ax = fig.add_axes(rect(H, x, ytop, SW, SW))
        scs.append(ax)
        cm = arr['P', c][0]
        ax.plot([0, B], [0, B], color=S.REF_INK, lw=0.6, zorder=1)
        for m, z in (('P', 2), ('M', 3)):
            ax.plot(cm, arr[m, c][1], S.MODEL_MARKER[m], ms=1.4, color=S.MODEL_COLOUR[m], alpha=0.35, mew=0, zorder=z)
        ax.set_xlim(0, B)
        ax.set_ylim(0, B)
        ax.set_aspect('equal')
        ax.set_xticks(np.arange(0, B + 1, 20))
        ax.set_yticks(np.arange(0, B + 1, 20))
        ax.tick_params(labelleft=col == 0, labelbottom=row == 1)
        inward_edge_labels(ax)
        C.frame(ax)
        C.tag(ax, LETTERS[10 + i], S.CITY_NAME[c])
    C.block_title(fig, scs, 'U-Net city daily mean NO$_2$ (ppb)', axis='y')
    C.block_title(fig, scs[5:], 'CMAQ city daily mean NO$_2$ (ppb)', axis='x')
    one = Line2D([], [], color=S.REF_INK, lw=0.6)
    from matplotlib.legend_handler import HandlerBase

    class _HEnd(HandlerBase):

        def create_artists(self, legend, orig, xd, yd, w, h, fs, tr):
            ms = orig.get_markersize()
            return [Line2D([xd + ms / 2.0, xd + w / 2.0, xd + w - ms / 2.0], [yd + h / 2.0] * 3, ls='none', marker=orig.get_marker(), ms=ms, color=orig.get_color(), mec=orig.get_markeredgecolor(), mew=orig.get_markeredgewidth(), transform=tr)]
    hres = h_model('M', line=False)
    legs = C.house_key(fig, levs, [h_cmaq(), (h_model('P', marker=False), h_model('P', line=False)), hres, one], ['CMAQ', S.PLAIN['P'], S.PLAIN['M'], 'One-to-one'], side='above', handler_map={hres: _HEnd()})
    leg = legs[0]
    for t, col in zip(leg.get_texts(), (S.INK, S.MODEL_TEXT['P'], S.MODEL_TEXT['M'], S.INK)):
        t.set_color(col)
    if s_top is None:
        fig.canvas.draw()
        rr = fig.canvas.get_renderer()
        mt = [t for t in fig.texts if t.get_text().startswith('Month')][0]
        mb = mt.get_window_extent(rr).y0
        kt = max((t.get_window_extent(rr).y1 for a in scs[:5] for t in a.texts))
        d_in = (mb - kt) / fig.dpi - S.BLOCK_GAP_PT / 72.0
        plt.close(fig)
        return draw_a3(arr, doy, B, plotted, year, s_top=S_TOP - d_in)
    return (fig, H, {'legend': leg, 'first_tag': levs[0].texts[0], 'row1': levs[0], 'row2_tag': levs[2].texts[0], 'scatter_tag': scs[0].texts[0], 'last_level': levs[8]})
