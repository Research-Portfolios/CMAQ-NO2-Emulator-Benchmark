"""The drawing part of figure fig_F2S: the two-way histograms of per-cell means on log axes with the 1:1 line, the cumulative curves of the differences, and their binned percentiles."""
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import build_fig_common as C
import build_style as S
import matplotlib.pyplot as plt
W = 6.5
NAME = {'M': 'Residual U-Net', 'P': 'Reference U-Net'}
LEVEL = 0.2
LEVEL_LS = (0, (3.0, 2.0))

def rect(H, x, ytop, w, h):
    return [x / W, (H - ytop - h) / H, w / W, h / H]

def lab(v):
    s = '%g' % v
    return s.replace('-', S.MINUS)

def nv7(diff):
    xa = np.round(np.linspace(0.0, 0.4, 801), 6)
    xs = np.round(np.linspace(-0.4, 0.4, 1601), 6)
    out = {'abs_x_ppb': xa, 'signed_x_ppb': xs, 'abs_share_pct': {}, 'signed_share_pct': {}, 'rule': "100 * (number of cells with |d| <= x, or d <= x) / 53,594, d each U-Net's 2016 cell mean minus CMAQ's"}
    for m in ('M', 'P'):
        n = diff[m].size
        a = np.sort(np.abs(diff[m]))
        b = np.sort(diff[m])
        out['abs_share_pct'][m] = 100.0 * np.searchsorted(a, xa, side='right') / n
        out['signed_share_pct'][m] = 100.0 * np.searchsorted(b, xs, side='right') / n
    return out

def nv8(cm, diff):
    lo, hi, step, mincells = (0.05, 30.0, 0.1, 300)
    be = 10.0 ** np.arange(np.log10(lo), np.log10(hi) + 1e-09, step)
    if be[-1] < hi * 0.999:
        be = np.append(be, hi)
    idx = np.digitize(cm, be) - 1
    nb = be.size - 1
    ncell = np.array([int((idx == b).sum()) for b in range(nb)])
    keep = ncell >= mincells
    q = {}
    for m in ('M', 'P'):
        qq = np.full((3, nb), np.nan)
        for b in range(nb):
            if keep[b]:
                qq[:, b] = np.percentile(diff[m][idx == b], [10, 50, 90])
        q[m] = qq
    return {'bin_edges_ppb': be, 'bin_centre_ppb': np.sqrt(be[:-1] * be[1:]), 'cells_per_bin': ncell, 'kept': keep, 'min_cells': mincells, 'p10_p50_p90': q, 'rule': "numpy.digitize of CMAQ's 2016 cell mean on 0.1-decade edges from 0.05 to 30 ppb; numpy.percentile (linear) at 10, 50 and 90 in bins of at least 300 cells"}

def draw(cm, means, diff, H2, edges, greys, class_lab, cnt, be_c, lim, mean_diff, within_txt, plotted):
    v7 = nv7(diff)
    v8 = nv8(cm, diff)
    plotted['cumulative_share'] = v7
    plotted['binned_quantiles'] = v8
    LX, RX = (0.62, 0.64)
    TOP = 0.48
    SQ = 0.96
    GAB = 0.42
    XC = LX + 2 * SQ + GAB + 0.74
    WC = W - RX - XC
    R1 = TOP
    R2 = R1 + SQ + 1.25
    H2_IN = 0.8
    W2 = (W - LX - RX - 0.3) / 2.0
    R3 = R2 + H2_IN + 0.42 + 0.24
    H3, H3S = (0.92, 0.3)
    H = R3 + H3 + 0.08 + H3S + 0.42 + 0.04
    fig = plt.figure(figsize=(W, H))
    cmap = ListedColormap(greys)
    norm = BoundaryNorm([1, 10, 100, 1000][:len(greys)] + [1000000000.0], cmap.N)
    rng = {}
    ab = []
    for m in ('P', 'M'):
        h = H2[m]
        rows = np.where(h.sum(axis=1) > 0)[0]
        cols = np.where(h.sum(axis=0) > 0)[0]
        i0 = int(min(rows.min(), cols.min()))
        i1 = int(max(rows.max(), cols.max())) + 1
        rng[m] = [float(edges[i0]), float(edges[i1])]
    lo = min((v[0] for v in rng.values()))
    hi = max((v[1] for v in rng.values()))
    for k, m in enumerate(('P', 'M')):
        h = H2[m]
        ax = fig.add_axes(rect(H, LX + k * (SQ + GAB), R1, SQ, SQ))
        ab.append(ax)
        ax.pcolormesh(edges, edges, np.ma.masked_less(h.T, 1), cmap=cmap, norm=norm, rasterized=True, shading='flat', linewidth=0, antialiased=False, zorder=1)
        ax.plot([lo, hi], [lo, hi], color=S.REF_INK, lw=0.6, zorder=3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        ticks = [t for t in (0.01, 0.1, 1.0, 10.0) if lo <= t <= hi]
        for f in (ax.set_xticks, ax.set_yticks):
            f(ticks)
        ax.minorticks_off()
        ax.set_xticklabels([lab(t) for t in ticks])
        ax.set_yticklabels(['' if t == lo else lab(t) for t in ticks])
        C.frame(ax)
        C.tag(ax, 'ab'[k], NAME[m])
    plotted['ab_occupied_range_ppb'] = {'rule': 'lower edge of the lowest and upper edge of the highest stored bin holding a cell on either axis of the panel', 'values': rng}
    plotted['ab_axis_range_drawn_ppb'] = {'rule': 'one range for (a) and (b): the lowest and highest of ab_occupied_range_ppb', 'values': {m: [lo, hi] for m in ('P', 'M')}}
    ab[0].set_ylabel('U-Net 2016 cell mean NO$_2$ (ppb)', fontsize=S.FS_LABEL)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    xt1 = C.block_title(fig, ab, 'CMAQ 2016 cell mean NO$_2$ (ppb)', axis='x')
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    from matplotlib.transforms import Bbox
    tlow = Bbox.union([a.get_tightbbox(r) for a in ab]).y0
    d_t = (tlow - xt1.get_window_extent(r).y0) * 72.0 / fig.dpi
    hs = [Patch(facecolor=g, edgecolor='none') for g in greys] + [Line2D([], [], color=S.REF_INK, lw=0.6)]
    labs1 = list(class_lab) + ['One-to-one']
    k1 = C.house_key(fig, ab, None, None, rows=[list(zip(hs[:-1], labs1[:-1])), [(hs[-1], labs1[-1])]], side='below', gap_pt=d_t + S.CLEAR_RATIO * S.AXIS_LABELPAD_PT + 1.0, title='Cells per bin')[0]
    axc = fig.add_axes(rect(H, XC, R1, WC, SQ))
    up = {'M': 1.0, 'P': -1.0}
    for m in ('M', 'P'):
        axc.stairs(up[m] * cnt[m], be_c, baseline=0.0, fill=True, facecolor=S.MODEL_COLOUR[m], edgecolor='none', alpha=0.45, zorder=2)
        axc.stairs(up[m] * cnt[m], be_c, baseline=0.0, fill=False, color=S.MODEL_COLOUR[m], lw=0.7, zorder=3)
    axc.axhline(0.0, color=S.INK2, lw=0.5, zorder=4)
    axc.axvline(0.0, color=S.REF_INK, lw=0.6, zorder=1)
    umax, dmax = (float(cnt['M'].max()), float(cnt['P'].max()))
    yl = 1.12 * max(umax, dmax)
    axc.set_ylim(-yl, yl)
    step = 5000.0
    yt = np.arange(-np.floor(yl / step) * step, yl, step)
    axc.set_yticks(yt)
    axc.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: '{:,.0f}'.format(abs(v))))
    axc.set_xlim(-lim, lim)
    axc.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    axc.set_xticklabels([lab(v) for v in (-1.0, -0.5, 0.0, 0.5, 1.0)])
    axc.set_xlabel('U-Net minus CMAQ (ppb)', fontsize=S.FS_LABEL)
    axc.set_ylabel('Cells', fontsize=S.FS_LABEL)
    for m in ('M', 'P'):
        md = mean_diff[m]
        sg = up[m]
        axc.plot([md, md], [0, sg * 0.1 * yl], color=S.MODEL_TEXT[m], lw=1.3, zorder=6, solid_capstyle='butt')
    C.frame(axc)
    C.tag(axc, 'c', 'Per-cell difference, 2016')
    ad = fig.add_axes(rect(H, LX, R2, W2, H2_IN))
    ae = fig.add_axes(rect(H, LX + W2 + 0.3, R2, W2, H2_IN))
    for m in ('P', 'M'):
        ad.plot(v7['abs_x_ppb'], v7['abs_share_pct'][m], color=S.MODEL_COLOUR[m], ls=S.MODEL_LS[m], lw=1.2, zorder=3)
        ae.plot(v7['signed_x_ppb'], v7['signed_share_pct'][m], color=S.MODEL_COLOUR[m], ls=S.MODEL_LS[m], lw=1.2, zorder=3)
    ad.axvline(LEVEL, color=S.REF_INK, lw=0.6, ls=LEVEL_LS, zorder=1)
    for m in ('M', 'P'):
        s_at = 100.0 * float(np.mean(np.abs(diff[m]) < LEVEL))
        ad.plot([LEVEL], [s_at], S.MODEL_MARKER[m], ms=4.2, color=S.MODEL_COLOUR[m], mec='white', mew=0.4, zorder=4)
    ad.set_xlim(0, 0.4)
    ad.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
    ad.set_xticklabels(['0', '0.1', '0.2', '0.3', '0.4'])
    ad.set_xlabel('Absolute difference, 2016 cell means (ppb)', fontsize=S.FS_LABEL)
    ad.set_ylabel('Percent of land cells', fontsize=S.FS_LABEL)
    ae.axvline(0, color=S.REF_INK, lw=0.6, zorder=1)
    ae.set_xlim(-0.4, 0.4)
    ae.set_xticks([-0.4, -0.2, 0, 0.2, 0.4])
    ae.set_xticklabels([lab(v) for v in (-0.4, -0.2, 0, 0.2, 0.4)])
    ae.set_xlabel('U-Net minus CMAQ, 2016 cell means (ppb)', fontsize=S.FS_LABEL)
    ae.tick_params(axis='y', labelleft=False)
    for ax in (ad, ae):
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        C.frame(ax)
    C.tag(ad, 'd', 'Absolute difference')
    C.tag(ae, 'e', 'Signed difference')
    af = fig.add_axes(rect(H, LX, R3, W - LX - RX, H3))
    an = fig.add_axes(rect(H, LX, R3 + H3 + 0.08, W - LX - RX, H3S))
    af.axhline(0, color=S.REF_INK, lw=0.6, zorder=1)
    ctr = v8['bin_centre_ppb']
    for m in ('P', 'M'):
        q = v8['p10_p50_p90'][m]
        for rr in (0, 2):
            af.plot(ctr, q[rr], color=S.MODEL_COLOUR[m], ls=S.MODEL_LS[m], lw=0.6, zorder=3)
        af.plot(ctr, q[1], color=S.MODEL_COLOUR[m], ls=S.MODEL_LS[m], lw=1.3, marker=S.MODEL_MARKER[m], ms=2.8, mew=0, zorder=5 if m == 'M' else 4)
    be = v8['bin_edges_ppb']
    keep = v8['kept']
    ncell = v8['cells_per_bin']
    an.bar(be[:-1][keep], ncell[keep], width=np.diff(be)[keep], align='edge', color='#B8B8B8', edgecolor='white', lw=0.3)
    an.bar(be[:-1][~keep], ncell[~keep], width=np.diff(be)[~keep], align='edge', color='#E3E3E3', edgecolor='white', lw=0.3)
    for ax in (af, an):
        ax.set_xscale('log')
        ax.set_xlim(be[0], be[-1])
        ax.minorticks_off()
        ax.set_xticks([0.1, 1, 10])
        C.frame(ax)
    af.set_xticklabels([])
    af.tick_params(axis='x', labelbottom=False)
    qa = np.concatenate([v8['p10_p50_p90'][m].ravel() for m in ('M', 'P')])
    ylim = 1.1 * float(np.nanmax(np.abs(qa)))
    af.set_ylim(-ylim, ylim)
    yt = [v for v in (-0.4, -0.2, 0.0, 0.2, 0.4) if abs(v) <= ylim]
    af.set_yticks(yt)
    af.set_yticklabels([lab(v) for v in yt])
    af.set_ylabel('U-Net minus CMAQ (ppb)', fontsize=S.FS_LABEL)
    an.set_ylim(0, 1.15 * float(ncell.max()))
    an.set_yticks([0, 5000])
    an.set_yticklabels(['0', '5,000'])
    an.set_xticklabels(['0.1', '1', '10'])
    an.yaxis.tick_right()
    an.yaxis.set_label_position('right')
    an.set_ylabel('Cells', fontsize=S.FS_LABEL)
    an.set_xlabel('CMAQ 2016 cell mean NO$_2$ (ppb)', fontsize=S.FS_LABEL)
    C.tag(af, 'f', 'Median and 10th to 90th percentiles per bin of at least 300 cells')
    hk = [(Line2D([], [], color=S.MODEL_COLOUR[m], ls=S.MODEL_LS[m], lw=1.2, solid_capstyle='butt', dash_capstyle='butt'), Line2D([], [], ls='none', marker=S.MODEL_MARKER[m], ms=3.6, color=S.MODEL_COLOUR[m], mec='white', mew=0.3)) for m in ('M', 'P')]
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    from matplotlib.transforms import Bbox
    row1 = ab + [axc]
    t_hi = Bbox.union([a.get_tightbbox(r) for a in row1]).y1
    tag_hi = max((t.get_window_extent(r).y1 for a in row1 for t in a.texts if (t.get_gid() or '').startswith('tagline:')))
    over = (t_hi - tag_hi) * 72.0 / fig.dpi
    hk += [Line2D([], [], color=S.REF_INK, lw=0.6), Line2D([], [], color=S.REF_INK, lw=0.6, ls=LEVEL_LS)]
    k2 = C.house_key(fig, row1, hk, [NAME['M'], NAME['P'], 'Zero difference', 'Level %g' % LEVEL], side='above', gap_pt=S.KEY_GAP_PT - over)[0]
    for t, c in zip(k2.get_texts(), (S.MODEL_TEXT['M'], S.MODEL_TEXT['P'], S.INK, S.INK)):
        t.set_color(c)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    k1_top = k1.get_window_extent(r).y1
    c_low = axc.get_tightbbox(r).y0
    hmean = [Line2D([], [], ls='none', marker='|', ms=6.0, mew=1.3, color=S.MODEL_TEXT[m]) for m in ('M', 'P')]
    from matplotlib.legend_handler import HandlerBase

    class _HRight(HandlerBase):

        def create_artists(self, legend, orig, xd, yd, w, h, fs, tr):
            return [Line2D([xd + w - orig.get_markeredgewidth()], [yd + h / 2.0], ls='none', marker=orig.get_marker(), ms=orig.get_markersize(), color=orig.get_color(), mew=orig.get_markeredgewidth(), transform=tr)]
    k3 = C.house_key(fig, [axc], hmean, [NAME['M'], NAME['P']], side='below', gap_pt=(c_low - k1_top) * 72.0 / fig.dpi, title='Mean difference', handler_map={h_: _HRight() for h_ in hmean})[0]
    for t, m in zip(k3.get_texts(), ('M', 'P')):
        t.set_color(S.MODEL_TEXT[m])
    return (fig, H, {'key': k2, 'cells_key': k1, 'mean_key': k3, 'ab': ab, 'c': axc, 'd': ad, 'f': af, 'n': an})
