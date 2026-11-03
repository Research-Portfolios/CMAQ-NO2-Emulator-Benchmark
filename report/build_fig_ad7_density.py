"""Draws figure fig_ad7_density: each U-Net's 2016 per-cell mean minus CMAQ's against CMAQ's mean as a two-dimensional histogram with a log axis, with the running median."""
import json
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_style as S
import build_fig_common as C
import numpy as np
import matplotlib.patheffects as PE
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerBase
STEM = 'fig_ad7_density'
F2 = os.path.join(S.FIG, 'fig_F2_plotted.json')
YL = 1.0
X0, X1 = (0.07, 25.0)
XEDGES = np.logspace(np.log10(X0), np.log10(X1), 49)
YEDGES = np.linspace(-YL, YL, 51)
MEDGES = np.logspace(np.log10(X0), np.log10(X1), 25)
CNT = [1, 3, 10, 30, 100, 300]
N_MIN, N_FULL = (10, 50)
PANELS = [('P', 'c_P_minus_cmaq'), ('M', 'd_M_minus_cmaq')]

class _Open:

    def __init__(self, m):
        self.m = m

class _OpenHandler(HandlerBase):

    def __init__(self, ms=3.0, mew=0.8):
        super().__init__()
        self.ms, self.mew = (ms, mew)

    def create_artists(self, legend, h, xd, yd, w, ht, fs, trans):
        y = (ht - yd) / 2.0
        return [Line2D([-xd, -xd + w], [y, y], color='none', lw=0, transform=trans), Line2D([-xd + w - 2.0], [y], ls='none', marker=S.MODEL_MARKER[h.m], ms=self.ms, mfc='white', mec=S.MODEL_COLOUR[h.m], mew=self.mew, transform=trans)]

class _Bound:
    pass

class _BoundHandler(HandlerBase):

    def create_artists(self, legend, h, xd, yd, w, ht, fs, trans):
        y = (ht - yd) / 2.0
        x1 = -xd + w
        return [Line2D([-xd, x1], [y, y], color='none', lw=0, transform=trans), Line2D([x1 - 7.2, x1], [y + 2.9, y + 2.9], color=S.INK, lw=0.8, transform=trans), Line2D([x1 - 3.6, x1 - 3.6], [y + 2.9, y - 2.2], color=S.INK, lw=0.8, transform=trans)]

class _Zero:
    pass

class _ZeroHandler(HandlerBase):

    def create_artists(self, legend, h, xd, yd, w, ht, fs, trans):
        y = (ht - yd) / 2.0
        return [Line2D([-xd, -xd + w], [y, y], color=S.REF_INK, lw=0.6, solid_capstyle='butt', transform=trans)]

class _Med:

    def __init__(self, m, halo):
        self.m, self.halo = (m, halo)

class _MedHandler(HandlerBase):

    def create_artists(self, legend, h, xd, yd, w, ht, fs, trans):
        y = (ht - yd) / 2.0
        col = S.MODEL_COLOUR[h.m]
        return [Line2D([-xd, -xd + w], [y, y], color=col, lw=1.3, ls=S.MODEL_LS[h.m], path_effects=h.halo, transform=trans), Line2D([-xd + w / 2.0], [y], ls='none', marker=S.MODEL_MARKER[h.m], ms=3.0, mfc=col, mec='white', mew=0.5, transform=trans)]

def _below_title_gap(fig, axes, xt):
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    tight = Bbox.union([a.get_tightbbox(r) for a in axes])
    xb = xt.get_window_extent(r)
    return (tight.y0 - xb.y0) * 72.0 / fig.dpi + S.CLEAR_RATIO * S.AXIS_LABELPAD_PT + S.GAP_TOL_PT

def run_median(x, d):
    out = []
    for a, b in zip(MEDGES[:-1], MEDGES[1:]):
        sel = (x >= a) & (x < b)
        n = int(sel.sum())
        out.append([float(np.sqrt(a * b)), float(np.median(d[sel])) if n >= N_MIN else None, n])
    return out

def main():
    with open(F2, encoding='utf-8') as f:
        bounds = json.load(f)['plotted']['level_bounds']
    src = {m: C.sealed('F4_mean_fold2016_%s.npy' % m) for m in ('M', 'P')}
    a = {m: np.load(p).astype(np.float64) for m, p in src.items()}
    cm = a['M'][0]
    assert np.array_equal(a['P'][0], cm)
    assert np.isfinite(cm).all() and cm.min() >= X0 and (cm.max() < X1), (cm.min(), cm.max())
    hist, med, outside, shade = ({}, {}, {}, {})
    for m, k in PANELS:
        d = a[m][1] - a[m][0]
        assert d.shape == cm.shape and np.isfinite(d).all()
        h, _, _ = np.histogram2d(cm, d, bins=[XEDGES, YEDGES])
        hist[m] = h.astype(int)
        med[m] = run_median(cm, d)
        outside[m] = int((np.abs(d) > YL).sum())
        assert int(h.sum()) + outside[m] == cm.size
        shade[m] = np.where(hist[m] >= 1, np.searchsorted(CNT, hist[m], side='right') - 1, -1)
    ramp = list(S.GREYS6)
    ramp_m = C.ramp_measure(ramp)
    assert ramp_m['pass'], ramp_m
    cmap = ListedColormap(ramp)
    norm = BoundaryNorm(CNT + [10 ** 6], len(ramp))
    C.setup()
    W = 6.5
    L, R, GUT = (0.58, 0.14, (S.GUTTER_PT + S.TICK_LEN_PT) / 72.0)
    pw = (W - L - R - GUT) / 2.0
    ph = 2.45
    halo = [PE.withStroke(linewidth=2.6, foreground='white')]

    def draw(top, H):
        fig = S.plt.figure(figsize=(W, H))
        bot = H - top - ph
        axes = []
        for i, (m, k) in enumerate(PANELS):
            x0 = L + i * (pw + GUT)
            ax = fig.add_axes([x0 / W, bot / H, pw / W, ph / H])
            hm = np.ma.masked_less(hist[m].T, 1)
            ax.pcolormesh(XEDGES, YEDGES, hm, cmap=cmap, norm=norm, rasterized=True, edgecolors='none')
            ax.axhline(0.0, color=S.REF_INK, lw=0.6, zorder=3)
            xs = np.array([r[0] for r in med[m] if r[1] is not None])
            ys = np.array([r[1] for r in med[m] if r[1] is not None])
            ns = np.array([r[2] for r in med[m] if r[1] is not None])
            col = S.MODEL_COLOUR[m]
            ax.plot(xs, ys, color=col, lw=1.3, ls=S.MODEL_LS[m], zorder=4, path_effects=halo)
            full = ns >= N_FULL
            ax.plot(xs[full], ys[full], ls='none', marker=S.MODEL_MARKER[m], ms=3.0, mfc=col, mec='white', mew=0.5, zorder=5)
            ax.plot(xs[~full], ys[~full], ls='none', marker=S.MODEL_MARKER[m], ms=3.0, mfc='white', mec=col, mew=0.8, zorder=5)
            ax.set_xscale('log')
            ax.set_xlim(X0, X1)
            ax.set_ylim(-YL, YL)
            ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10, 20])
            ax.set_xticklabels(['0.1', '0.2', '0.5', '1', '2', '5', '10', '20'])
            ax.minorticks_off()
            yt = [-1.0, -0.5, 0.0, 0.5, 1.0]
            ax.set_yticks(yt)
            if i == 0:
                ax.set_yticklabels([('%.1f' % t).replace('-', S.MINUS) if t else '0' for t in yt])
                ax.set_ylabel('U-Net minus CMAQ (ppb)', fontsize=S.FS_LABEL)
            else:
                ax.set_yticklabels([])
            for b in bounds:
                ax.plot([b, b], [YL, YL - 0.07], color=S.INK, lw=0.8, zorder=6)
            C.frame(ax)
            C.tag(ax, 'ab'[i], '%s minus CMAQ' % S.PLAIN[m])
            axes.append(ax)
        xt = C.block_title(fig, axes, 'CMAQ 2016 test-day mean NO$_2$ (ppb, log scale)', axis='x')
        hmap = {_Open: _OpenHandler(), _Bound: _BoundHandler(), _Med: _MedHandler(), _Zero: _ZeroHandler()}
        legs = C.house_key(fig, axes, None, None, rows=[[(_Zero(), 'Zero difference')]], side='above', handler_map=hmap)
        fig.canvas.draw()
        r0_ = fig.canvas.get_renderer()
        up0_ = legs[0].get_window_extent(r0_).y1

        def _row2_gap(blk):
            from matplotlib.transforms import Bbox
            t_ = Bbox.union([a_.get_tightbbox(r0_) for a_ in blk])
            return (up0_ - t_.y1) * 72.0 / fig.dpi + S.KEY_ROWSPACE_EM * S.KEY_FS
        for i, (m, k) in enumerate(PANELS):
            legs += C.house_key(fig, [axes[i]], None, None, rows=[[(_Open(m), 'Under 50 cells')]], side='above', handler_map=hmap, handlelength=0.5, gap_pt=_row2_gap([axes[i]]))
        legs += C.house_key(fig, axes, None, None, rows=[[(_Bound(), 'Level class bound')]], side='above', handler_map=hmap, handlelength=0.9, gap_pt=_row2_gap(axes))
        fig.canvas.draw()
        r_ = fig.canvas.get_renderer()
        up_ = max((lg_.get_window_extent(r_).y1 for lg_ in legs))
        for i, (m, k) in enumerate(PANELS):
            ax_ = axes[i]
            gap_ = (up_ - ax_.get_tightbbox(r_).y1) * 72.0 / fig.dpi + S.KEY_ROWSPACE_EM * S.KEY_FS
            legs += C.house_key(fig, [ax_], None, None, rows=[[(_Med(m, halo), '%s, running median' % S.PLAIN[m])]], side='above', gap_pt=gap_, handler_map=hmap)
        for lg_ in legs:
            for t_ in lg_.get_texts():
                for m_ in ('M', 'P'):
                    if t_.get_text().startswith(S.PLAIN[m_]):
                        t_.set_color(S.MODEL_TEXT[m_])
        bw, bh = (0.42, 0.11)
        rect = C.key_rect(fig, axes, 6 * bw, bh, side='below', gap_pt=_below_title_gap(fig, axes, xt))
        bk = C.stepped_bar_key(fig, rect, ramp, CNT, title='Land cells per bin', open_low=False, open_high=True, ticklabels=['%d' % c for c in CNT])
        bk['cax'].set_gid('noframe')
        return (fig, axes, legs, bk, xt)
    MARG = 0.1
    top0, H0 = (1.0, 5.5)
    fig, axes, legs, bk, xt = draw(top0, H0)
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    S.plt.close(fig)
    top = top0 - (H0 - tb.y1) + MARG
    H = round(tb.y1 - tb.y0 + 2 * MARG, 3)
    fig, axes, legs, bk, xt = draw(top, H)
    plotted = {'label': 'DESC, added after the plan was fixed', 'source': 'results/evaluation_test_years/F4_mean_fold2016_M.npy and F4_mean_fold2016_P.npy (row 0 CMAQ mean, row 1 model mean; difference row 1 minus row 0, float64, unrounded); level_bounds from figures\\fig_F2_plotted.json', 'bins_orientation': 'nv_cells_per_bin_<m>[i][j]: i the CMAQ bin (x_edges_ppb), j the difference bin (y_edges_ppb); right edge open but the last', 'x_edges_ppb': XEDGES, 'y_edges_ppb': YEDGES, 'median_edges_ppb': MEDGES, 'median_min_cells': N_MIN, 'median_open_below_cells': N_FULL, 'count_class_lower': CNT, 'count_ramp': ramp, 'count_ramp_measure': ramp_m, 'level_bounds_ticks': bounds, 'panels': '(a) P, (b) M', 'figure_in': [W, H], 'nv_cells_outside_view': {'P': outside['P'], 'M': outside['M'], 'rule': '|difference| > 1 ppb'}}
    for m, _k in PANELS:
        plotted['nv_cells_per_bin_' + m] = hist[m]
        plotted['nv_running_median_' + m] = med[m]
        plotted['shade_class_per_bin_' + m] = shade[m]
    plotted['fc_check'] = C.fc_check(fig)
    plotted['text_check'] = C.text_check(fig)
    fcc, tc = (plotted['fc_check'], plotted['text_check'])
    print('fc_check pass', fcc['pass'], 'case_pass', fcc['case_pass'], ascii(fcc['case']), ascii(fcc['overlaps']), ascii(fcc['outside']), fcc['small'], fcc['open_frames'])
    print('text_check pass', tc['pass'], ascii(tc['overlaps']), ascii(tc['within_0.08in_of_edge']), 'fig', tc['figure_in'])
    print('outside', outside, 'ramp', ramp_m)
    S.save(fig, STEM, plotted, [src['M'], src['P'], F2])
    cap = "\\textbf{Difference of each U-Net's 2016 test-day mean modeled surface NO$_2$ from CMAQ's (ppb) against CMAQ's 2016 test-day mean (ppb, log scale), per land cell of the contiguous US, reference U-Net (a) and residual U-Net (b).} The values are those mapped in Figure~\\ref{fig:F2}(c) and (d), from each U-Net at seed 0 trained without 2016. Gray shading counts the land cells in each bin of CMAQ's mean and the difference, darker for more cells (key below). The line with marks is the running median of the difference over bins of CMAQ's mean (all land cells of a bin, those beyond the vertical range included). It is drawn where a bin holds 10 or more cells, vermilion, solid and with discs for the residual U-Net and blue, dashed and with squares for the reference U-Net; open marks are medians from fewer than 50 cells. The black line marks zero difference, and the ticks inside the top edge mark the level class bounds of Figure~\\ref{fig:F2}. The vertical range leaves out %d cells for the reference U-Net and %d for the residual U-Net, whose differences lie beyond it. " % (outside['P'], outside['M']) + "Grouping by CMAQ's own value lets regression toward the mean make a smoother model read low where CMAQ is high. These results are descriptive and were not specified in advance."
    alt = "Two density panels: each U-Net's 2016 mean minus CMAQ's against CMAQ's mean per land cell, with running medians."
    longd = "Two side-by-side panels, (a) reference U-Net and (b) residual U-Net. Horizontal axis: CMAQ's 2016 test-day mean NO2 per land cell, 0.07 to 25 ppb on a log scale; vertical axis: the U-Net's mean minus CMAQ's, minus 1 to 1 ppb. Gray shading shows how many land cells fall in each bin, densest near zero difference between about 0.5 and 3 ppb. A black line marks zero; short ticks inside the top edge mark the level class bounds 1, 2, 4, 8 and 16 ppb. The reference U-Net's running median (blue, dashed, squares) lies a little below zero at low CMAQ levels and rises above zero at the highest; the residual U-Net's (vermilion, solid, discs) stays near zero and falls below it at the highest levels. Open marks show medians from fewer than 50 cells. A key above names the marks and the zero line; a key below gives the gray classes, with bounds at %s and %d cells. The range of plus or minus %g ppb leaves out %d cells for the reference U-Net and %d for the residual U-Net." % (', '.join(('%d' % c for c in CNT[:-1])), CNT[-1], YL, outside['P'], outside['M'])
    C.write_text(STEM, cap, alt, longd)
    print('words', len(cap.split()))
if __name__ == '__main__':
    main()
