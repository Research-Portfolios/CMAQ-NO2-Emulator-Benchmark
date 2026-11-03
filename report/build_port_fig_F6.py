"""The drawing part of figure fig_F6: the five-year mean error by CMAQ decile with each U-Net's range over the years, CMAQ's decile bounds on the top axis, and the top decile of each held-out year."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import build_style as S
import build_fig_common as FC
YEARS = S.YEARS
DECILES = np.arange(1, 11)
NAME = {'M': 'Residual U-Net', 'P': 'Reference U-Net', 'GBT': 'Gradient-boosted trees', 'CLIM-WD': 'Weekday climatology'}
ORDER = ['M', 'P', 'GBT', 'CLIM-WD']
BAND_M = 0.24
TAG_GAP = 37.0

def tint(hx, a):
    c = FC.hex2rgb(hx) * a + (1 - a)
    return '#%02X%02X%02X' % tuple((int(round(v * 255)) for v in c))

def dash(v):
    return ('%.2f' % v).replace('-', S.MINUS)

def blend(top, a, under):
    c = FC.hex2rgb(top) * a + FC.hex2rgb(under) * (1 - a)
    return '#%02X%02X%02X' % tuple((int(round(v * 255)) for v in c))
W = 6.5
TOP_PAD, KEYH, TAGL, TOPAX = (0.08, 0.34, 0.2, 0.4)
PH_A, XL_A, GAP_B, PH_B, XL_B, BOT = (2.7, 0.42, 0.3, 1.15, 0.42, 0.06)
LEFT, WIDTH = (0.62, 5.76)
BOT_C3 = 0.1
LABEL_GAP_PT = 3.0

def draw(arr, mean, bounds, lo, hi):
    H0 = 7.0
    fig = plt.figure(figsize=(W, H0))
    ax = fig.add_axes([LEFT / W, 3.6 / H0, WIDTH / W, PH_A / H0])
    bx = fig.add_axes([LEFT / W, 0.6 / H0, WIDTH / W, PH_B / H0])
    drawn = {}
    cM, cP = (S.MODEL_COLOUR['M'], S.MODEL_COLOUR['P'])
    yrsM = np.array([arr['M', y] for y in YEARS])
    yrsP = np.array([arr['P', y] for y in YEARS])
    ax.fill_between(DECILES, yrsM.min(0), yrsM.max(0), facecolor=cM, alpha=BAND_M, edgecolor='none', lw=0, zorder=1)
    ax.fill_between(DECILES, yrsP.min(0), yrsP.max(0), facecolor=cP, alpha=BAND_M, edgecolor='none', lw=0, zorder=2)
    ax.axhline(0.0, color=S.REF_INK, lw=0.6, zorder=1)
    clim = mean['CLIM-WD']
    ax.plot(DECILES, clim, color=S.MODEL_COLOUR['CLIM-WD'], lw=0.8, marker=S.MODEL_MARKER['CLIM-WD'], ms=4.0, mec='white', mew=0.4, zorder=3)
    ax.plot(DECILES, mean['GBT'], color=S.MODEL_COLOUR['GBT'], lw=0.8, marker=S.MODEL_MARKER['GBT'], ms=4.0, mec='white', mew=0.4, zorder=4)
    ax.plot(DECILES, mean['P'], color=cP, lw=1.3, ls=S.MODEL_LS['P'], marker=S.MODEL_MARKER['P'], ms=4.4, mec='white', mew=0.4, zorder=5)
    ax.plot(DECILES, mean['M'], color=cM, lw=1.3, ls=S.MODEL_LS['M'], marker=S.MODEL_MARKER['M'], ms=4.6, mec='white', mew=0.4, zorder=6)
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(lo, hi)
    ax.set_xticks(DECILES)
    yt = np.round(np.arange(lo, hi + 1e-09, 0.2), 6)
    ax.set_yticks(yt)
    ax.set_yticklabels([('%.1f' % t).replace('-', S.MINUS) if abs(t) > 1e-09 else '0' for t in yt])
    off = [int(d) for d, v in zip(DECILES, clim) if v < lo]
    for d in off:
        v = clim[d - 1]
        ax.annotate('', xy=(d, lo), xytext=(d, lo + 0.13), arrowprops=dict(arrowstyle='-|>', color=S.MODEL_COLOUR['CLIM-WD'], lw=0.8, mutation_scale=7), zorder=7)
        from matplotlib.transforms import offset_copy
        t_ = ax.text(d, lo + 0.035, dash(v), ha='right', va='center', fontsize=S.FS_TICK, color=S.INK, transform=offset_copy(ax.transData, fig=fig, x=-LABEL_GAP_PT, y=0.0, units='points'))
        fig.canvas.draw()
        rr_ = fig.canvas.get_renderer()
        seg_ = ax.transData.transform([(d - 1, clim[d - 2]), (d, v)])
        g_ = 2.0 * fig.dpi / 72.0
        yy_ = lo + 0.035
        for _ in range(40):
            b_ = t_.get_window_extent(rr_)
            if not S._seg_hits(seg_, (b_.x0 - g_, b_.y0 - g_, b_.x1 + g_, b_.y1 + g_)).any():
                break
            yy_ += 0.01
            t_.set_y(yy_)
            fig.canvas.draw()
        drawn['a_off_scale_label_y_ppb'] = round(float(yy_), 3)
    drawn['a_off_scale_value_printed'] = {str(d): float(clim[d - 1]) for d in off}
    ax.set_xlabel("Decile of CMAQ's daily mean NO$_2$ in each held-out year")
    ax.set_ylabel('Model minus CMAQ (ppb)')
    pos = [j + 0.5 for j in sorted(bounds)]
    ax.set_xticks(pos, minor=True)
    ax.set_xticklabels(['%s–%s' % ('%.2f' % bounds[j][0], '%.2f' % bounds[j][1]) for j in sorted(bounds)], minor=True, fontsize=S.FS_TICK)
    ax.tick_params(axis='x', which='minor', bottom=False, top=True, labelbottom=False, labeltop=True, direction='out', length=S.TICK_LEN_PT, width=S.TICK_W_PT, pad=S.TICK_PAD_PT)
    ax.tick_params(axis='x', which='major', top=False, labeltop=False)
    TOP_TITLE = 'Bound between deciles, range over years (ppb)'
    FC.frame(ax)
    rows = list(range(len(YEARS)))[::-1]
    for i, y in enumerate(YEARS):
        r = rows[i]
        vm, vp = (arr['M', y][9], arr['P', y][9])
        bx.plot([vm, vp], [r, r], color='#C8C8C8', lw=1.0, zorder=1)
        bx.plot(vm, r, ls='none', marker=S.MODEL_MARKER['M'], ms=4.6, color=cM, mec='white', mew=0.4, zorder=4)
        bx.plot(vp, r, ls='none', marker=S.MODEL_MARKER['P'], ms=4.4, color=cP, mec='white', mew=0.4, zorder=3)
    bx.axvline(0.0, color=S.REF_INK, lw=0.6, zorder=1)
    bx.set_yticks(rows)
    bx.set_yticklabels([str(y) for y in YEARS])
    bx.set_ylim(-0.6, len(YEARS) - 0.4)
    allb = [arr[m, y][9] for m in ('M', 'P') for y in YEARS]
    b0 = np.floor(min(allb) / 0.4) * 0.4
    bx.set_xlim(b0 - 0.06, 0.08)
    bt = np.round(np.arange(b0, 1e-09, 0.2), 1)
    bx.set_xticks(bt)
    bx.set_xticklabels([('%.1f' % t).replace('-', S.MINUS) if abs(t) > 1e-09 else '0' for t in bt])
    bx.set_xlabel('Model minus CMAQ, decile 10 (ppb)')
    bx.set_ylabel('Held-out year')
    FC.house_ticks(bx, 'y')
    FC.frame(bx)
    drawn['b_xlim'] = list(bx.get_xlim())
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fr = ax.get_window_extent(r)
    from matplotlib.transforms import offset_copy
    fr = ax.get_window_extent(r)
    ltop = max([fr.y1] + [t.get_window_extent(r).y1 for t in ax.get_xticklabels(minor=True) if t.get_visible() and t.get_text().strip()])
    lab_pt = (ltop - fr.y1) * 72.0 / fig.dpi + S.AXIS_LABELPAD_PT
    top_t = ax.text(0.5, 1.0, TOP_TITLE, transform=offset_copy(ax.transAxes, fig=fig, x=0.0, y=lab_pt, units='points'), ha='center', va='bottom', fontsize=S.FS_LABEL, color=S.INK, gid='axistitle_top')
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ttop = top_t.get_window_extent(r).y1
    FC.tag(ax, 'a', 'Five-year mean and range over years, by decile', gap_pt=S.TITLE_GAP_PT + (ttop - fr.y1) * 72.0 / fig.dpi)
    FC.tag(bx, 'b', 'Decile 10 by year')
    hs = [Line2D([], [], color=cM, lw=1.3, ls=S.MODEL_LS['M'], marker=S.MODEL_MARKER['M'], ms=4.6, mec='white', mew=0.4), Line2D([], [], color=tint(cM, BAND_M), lw=5.6, solid_capstyle='butt', marker='s', ms=4.4, mfc='none', mec='none'), Line2D([], [], color=cP, lw=1.3, ls=S.MODEL_LS['P'], marker=S.MODEL_MARKER['P'], ms=4.4, mec='white', mew=0.4), Line2D([], [], color=tint(cP, BAND_M), lw=5.6, solid_capstyle='butt', marker='s', ms=4.4, mfc='none', mec='none'), Line2D([], [], color=S.MODEL_COLOUR['GBT'], lw=0.8, marker=S.MODEL_MARKER['GBT'], ms=4.0, mec='white', mew=0.4), Line2D([], [], color=S.MODEL_COLOUR['CLIM-WD'], lw=0.8, marker=S.MODEL_MARKER['CLIM-WD'], ms=4.0, mec='white', mew=0.4)]
    ls = ['Residual U-Net', 'Its range over the years', 'Reference U-Net', 'Its range over the years', 'Gradient-boosted trees', 'Weekday climatology']
    hs.append(Line2D([], [], color=S.REF_INK, lw=0.6))
    ls.append('Zero difference')
    key_rows = [[(hs[0], ls[0]), (hs[2], ls[2]), (hs[4], ls[4])], [(hs[1], ls[1]), (hs[3], ls[3]), (hs[5], ls[5])], [(hs[6], ls[6])]]
    from matplotlib.transforms import Bbox
    probe = fig.legend(hs, ls, ncol=3, fontsize=S.KEY_FS, frameon=False, borderpad=0.0, labelspacing=S.KEY_ROWSPACE_EM, handlelength=S.KEY_HANDLELENGTH_EM)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    key_h = probe.get_window_extent(r).height / fig.dpi
    fig.legends.remove(probe)
    ext = {}
    for a in (ax, bx):
        f = a.get_window_extent(r)
        tb = a.get_tightbbox(r)
        ext[id(a)] = ((tb.y1 - f.y1) / fig.dpi, (f.y0 - tb.y0) / fig.dpi)
    (ab_a, bl_a), (ab_b, bl_b) = (ext[id(ax)], ext[id(bx)])
    gap = S.BLOCK_GAP_PT / 72.0
    H = TOP_PAD + key_h + S.KEY_GAP_PT / 72.0 + ab_a + PH_A + bl_a + gap + ab_b + PH_B + bl_b + BOT_C3
    fig.set_size_inches(W, H)
    ya = H - (TOP_PAD + key_h + S.KEY_GAP_PT / 72.0 + ab_a + PH_A)
    yb = BOT_C3 + bl_b
    ax.set_position([LEFT / W, ya / H, WIDTH / W, PH_A / H])
    bx.set_position([LEFT / W, yb / H, WIDTH / W, PH_B / H])
    drawn['figure_in'] = [W, float(round(H, 3))]
    S.register_block(fig, 'a', [ax])
    S.register_block(fig, 'b', [bx])
    legs = FC.house_key(fig, [ax], None, None, rows=key_rows, side='above')
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    top_in = max((lg.get_window_extent(r).y1 for lg in legs)) / fig.dpi
    need = top_in + FC.MARGIN_IN + 0.02 - H
    if need > 0:
        for lg in legs:
            fig.legends.remove(lg)
        H = H + need
        fig.set_size_inches(W, H)
        ax.set_position([LEFT / W, ya / H, WIDTH / W, PH_A / H])
        bx.set_position([LEFT / W, yb / H, WIDTH / W, PH_B / H])
        drawn['figure_in'] = [W, float(round(H, 3))]
        legs = FC.house_key(fig, [ax], None, None, rows=key_rows, side='above')
    for lg in legs:
        for t in lg.get_texts():
            c = {'Residual U-Net': 'M', 'Reference U-Net': 'P'}.get(t.get_text())
            t.set_color(S.MODEL_TEXT.get(c, S.INK) if c else S.INK)
    for lg in legs:
        txts = lg.get_texts()
        for t, nxt in zip(txts, txts[1:]):
            if nxt.get_text() == 'Its range over the years' and t.get_text() in ('Residual U-Net', 'Reference U-Net'):
                nxt.set_color(t.get_color())
    drawn['key'] = ls
    tm, tp = (tint(cM, BAND_M), tint(cP, BAND_M))
    over = blend(cP, BAND_M, tm)
    drawn['band_colours'] = {'orange_band': tm, 'blue_band': tp, 'overlap': over}
    drawn['band_overlap_de2000'] = FC.ramp_measure([tm, over, tp])
    return (fig, drawn)
