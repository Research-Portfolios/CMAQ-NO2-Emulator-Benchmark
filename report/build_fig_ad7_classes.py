"""Draws figure fig_ad7_classes: mirrored distributions of each U-Net's 2016 per-cell mean minus CMAQ's by CMAQ level class, with the median and the 5th and 95th percentiles."""
import json
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_style as S
import build_fig_common as C
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerBase
STEM = 'fig_ad7_classes'
F2 = os.path.join(S.FIG, 'fig_F2_plotted.json')
YL = 1.0
YBINS = np.arange(-1.2, 1.3 + 1e-09, 0.04)
HALF = 0.44
FEW = 200
SIDES = [('P', 'c_P_minus_cmaq', -1), ('M', 'd_M_minus_cmaq', 1)]
ZERO_LS = (0, (3.0, 2.0))

class _Half:

    def __init__(self, m):
        self.m = m

class _HalfHandler(HandlerBase):

    def __init__(self, fill):
        super().__init__()
        self.fill = fill

    def create_artists(self, legend, h, xd, yd, w, ht, fs, trans):
        y = (ht - yd) / 2.0
        col = S.MODEL_COLOUR[h.m]
        return [Line2D([-xd, -xd + w], [y, y], color='none', lw=0, transform=trans), Rectangle((-xd, y - 3.6), w - 6.5, 7.2, facecolor=self.fill[h.m], edgecolor=col, lw=0.6, transform=trans), Line2D([-xd + w - 2.0], [y], ls='none', marker=S.MODEL_MARKER[h.m], ms=3.2, mfc=self.fill[h.m], mec=col, mew=0.8, transform=trans)]

class _Tick:

    def __init__(self, lw, length):
        self.lw, self.length = (lw, length)

class _TickHandler(HandlerBase):

    def create_artists(self, legend, h, xd, yd, w, ht, fs, trans):
        y = (ht - yd) / 2.0
        x1 = -xd + w
        x0 = -xd if h.length is None else x1 - h.length
        out = [Line2D([x0, x1], [y, y], color=S.INK, lw=h.lw, solid_capstyle='butt', transform=trans)]
        if h.length is not None:
            out.insert(0, Line2D([-xd, x1], [y, y], color='none', lw=0, transform=trans))
        return out

class _Zero:
    pass

class _ZeroHandler(HandlerBase):

    def create_artists(self, legend, h, xd, yd, w, ht, fs, trans):
        y = (ht - yd) / 2.0
        x1 = -xd + w
        x0 = x1 - 0.12 * 72.0
        return [Line2D([-xd, x1], [y, y], color='none', lw=0, transform=trans), Line2D([x0, x1], [y, y], color=S.REF_INK, lw=0.6, ls=ZERO_LS, solid_capstyle='butt', transform=trans)]

def tint(hexc, k=0.28):
    c = C.hex2rgb(hexc)
    return C.rgb2hex(1.0 - k * (1.0 - c))

def main():
    with open(F2, encoding='utf-8') as f:
        f2 = json.load(f)['plotted']
    v = f2['values_in_mask_order']
    bounds = f2['level_bounds']
    cm = np.asarray(v['a_cmaq'], float)
    dd = {m: np.asarray(v[k], float) for m, k, _s in SIDES}
    edges = [-np.inf] + list(bounds) + [np.inf]
    labels = ['Under %g' % bounds[0]]
    labels += ['%g–%g' % (a, b) for a, b in zip(bounds[:-1], bounds[1:])]
    labels += ['%g and over' % bounds[-1]]
    ncls, hists, pct, few_pts = ([], {}, {}, {})
    outside = {m: int((np.abs(dd[m]) > YL).sum()) for m in dd}
    cls = np.searchsorted(np.asarray(bounds), cm, side='right')
    target = [int(round(s * cm.size)) for s in f2['level_class_shares']['cmaq']]
    assert sum(target) == cm.size
    moved = []
    for j, bnd in enumerate(bounds):
        on = np.nonzero(cm == bnd)[0]
        have = int((cls == j).sum())
        need = target[j] - have
        assert 0 <= need <= on.size, (bnd, need, on.size)
        for idx in on[:need]:
            cls[idx] = j
            moved.append([int(idx), float(bnd)])
        assert need in (0, on.size), (bnd, need, on.size)
    assert [int((cls == j).sum()) for j in range(6)] == target
    for i in range(len(edges) - 1):
        sel = cls == i
        n = int(sel.sum())
        ncls.append(n)
        for m, _k, _s in SIDES:
            d = dd[m][sel]
            hists[i, m] = np.histogram(d, bins=YBINS)[0]
            pct[i, m] = [float(x) for x in np.percentile(d, [5, 50, 95])]
            if n < FEW:
                few_pts[i, m] = d[np.abs(d) <= YL]
    assert sum(ncls) == cm.size
    C.setup()
    W = 6.5
    L, R = (0.62, 0.14)
    aw = W - L - R
    ah = 2.65
    fill = {m: tint(S.MODEL_COLOUR[m]) for m in dd}

    def draw(top, H):
        fig = S.plt.figure(figsize=(W, H))
        bot = H - top - ah
        ax = fig.add_axes([L / W, bot / H, aw / W, ah / H])
        ax.axhline(0.0, color=S.REF_INK, lw=0.6, ls=ZERO_LS, zorder=2.8)
        widths = {}
        dividers = {}
        for i in range(len(ncls)):
            xc = float(i)
            top_h = max(hists[i, 'P'].max(), hists[i, 'M'].max())
            for m, _k, side in SIDES:
                col = S.MODEL_COLOUR[m]
                if ncls[i] < FEW:
                    p = few_pts[i, m]
                    ax.plot(np.full(p.size, xc + side * 0.16), p, ls='none', marker=S.MODEL_MARKER[m], ms=3.2, mfc=fill[m], mec=col, mew=0.8, zorder=3)
                    continue
                w = HALF * hists[i, m] / top_h
                widths[i, m] = w
                ax.stairs(xc + side * w, YBINS, baseline=xc, orientation='horizontal', fill=True, facecolor=fill[m], edgecolor='none', zorder=2)
                hc = hists[i, m]
                j0 = 0
                while j0 < hc.size:
                    if hc[j0] == 0:
                        j0 += 1
                        continue
                    j1 = j0
                    while j1 < hc.size and hc[j1] > 0:
                        j1 += 1
                    ax.stairs(xc + side * w[j0:j1], YBINS[j0:j1 + 1], baseline=xc, orientation='horizontal', fill=False, edgecolor=col, lw=0.6, zorder=2.5)
                    j0 = j1
                p5, p50, p95 = pct[i, m]
                for yv, ln, lw in ((p5, 0.12, 0.9), (p95, 0.12, 0.9), (p50, 0.3, 1.6)):
                    ax.plot([xc, xc + side * ln], [yv, yv], color=S.INK, lw=lw, solid_capstyle='butt', zorder=4)
            if ncls[i] < FEW:
                pv = np.concatenate([few_pts[i, m] for m, _k, _s in SIDES])
                lo_, hi_ = (float(pv.min()), float(pv.max()))
            else:
                nz = np.nonzero(hists[i, 'P'] + hists[i, 'M'])[0]
                lo_ = max(-YL, float(YBINS[nz[0]]))
                hi_ = min(YL, float(YBINS[nz[-1] + 1]))
            dividers[i] = [lo_, hi_]
            ax.plot([xc, xc], [lo_, hi_], color=S.FRAME_INK, lw=0.4, zorder=1.5)
        ax.set_xlim(-0.55, len(ncls) - 0.45)
        ax.set_ylim(-YL, YL)
        ax.set_xticks(range(len(ncls)))
        ax.set_xticklabels(labels)
        C.house_ticks(ax, 'x')
        yt = [-1.0, -0.5, 0.0, 0.5, 1.0]
        ax.set_yticks(yt)
        ax.set_yticklabels([('%.1f' % t).replace('-', S.MINUS) if t else '0' for t in yt])
        ax.set_ylabel('U-Net minus CMAQ (ppb)', fontsize=S.FS_LABEL)
        C.frame(ax)
        fig.canvas.draw()
        rr = fig.canvas.get_renderer()
        ylo = min((t.get_window_extent(rr).y0 for t in ax.get_xticklabels()))
        y_in = ylo / fig.dpi - 1.5 / 72.0
        nax = fig.add_axes([L / W, y_in / H, aw / W, 0.0001])
        nax.set_gid('noframe')
        for s in nax.spines.values():
            s.set_visible(False)
        nax.patch.set_visible(False)
        nax.set_xlim(ax.get_xlim())
        nax.set_yticks([])
        nax.set_xticks(range(len(ncls)))
        nax.set_xticklabels([C.thousands(n) for n in ncls])
        nax.tick_params(axis='x', length=0, pad=0, labelcolor=S.INK2)
        nax.set_xlabel('CMAQ 2016 test-day mean NO$_2$ class (ppb)', fontsize=S.FS_LABEL)
        fig.canvas.draw()
        rr = fig.canvas.get_renderer()
        g_now = (ylo - max((t.get_window_extent(rr).y1 for t in nax.get_xticklabels()))) * 72.0 / fig.dpi
        shift = (S.KEY_ROWSPACE_EM * S.KEY_FS - g_now) / 72.0
        pn = nax.get_position()
        nax.set_position([pn.x0, pn.y0 - shift / H, pn.width, pn.height])
        rows = [[(_Half('P'), '%s, left half' % S.PLAIN['P']), (_Half('M'), '%s, right half' % S.PLAIN['M'])], [(_Tick(1.6, None), 'Median'), (_Tick(0.9, 0.12 * 72.0), '5th and 95th percentiles'), (_Zero(), 'Zero difference')]]
        legs = C.house_key(fig, [ax], None, None, rows=rows, side='above', handler_map={_Half: _HalfHandler(fill), _Tick: _TickHandler(), _Zero: _ZeroHandler()})
        for lg_ in legs:
            for t_ in lg_.get_texts():
                for m_ in ('M', 'P'):
                    if t_.get_text().startswith(S.PLAIN[m_]):
                        t_.set_color(S.MODEL_TEXT[m_])
        return (fig, ax, nax, legs, widths, dividers)
    MARG = 0.1
    top0, H0 = (0.9, 5.0)
    fig, ax, nax, legs, widths, dividers = draw(top0, H0)
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    S.plt.close(fig)
    top = top0 - (H0 - tb.y1) + MARG
    H = round(tb.y1 - tb.y0 + 2 * MARG, 3)
    fig, ax, nax, legs, widths, dividers = draw(top, H)
    plotted = {'label': 'DESC, added after the plan was fixed', 'source': 'figures\\fig_F2_plotted.json values_in_mask_order a_cmaq, c_P_minus_cmaq, d_M_minus_cmaq; level_bounds', 'class_labels': labels, 'level_bounds': bounds, 'y_bins_ppb': YBINS, 'half_width': HALF, 'one_mark_per_cell_below_cells': FEW, 'fills': fill, 'figure_in': [W, H], 'on_bound_cells_moved_to_lower_class': moved, 'class_counts_figure2': target, 'nv_cells_per_class': ncls, 'nv_cells_outside_view': {'P': outside['P'], 'M': outside['M'], 'rule': '|difference| > 1 ppb'}, 'nv_percentiles_5_50_95': {'%d_%s' % k: pv for k, pv in pct.items()}, 'nv_hist_cells_per_bin': {'%d_%s' % k: h for k, h in hists.items()}, 'drawn_widths': {'%d_%s' % k: w for k, w in widths.items()}, 'one_mark_per_cell_values': {'%d_%s' % k: p for k, p in few_pts.items()}, 'layout': {'dividers_ppb': {str(i): v for i, v in dividers.items()}}}
    plotted['fc_check'] = C.fc_check(fig)
    plotted['text_check'] = C.text_check(fig)
    fcc, tc = (plotted['fc_check'], plotted['text_check'])
    print('fc_check pass', fcc['pass'], 'case_pass', fcc['case_pass'], ascii(fcc['case']), ascii(fcc['overlaps']), ascii(fcc['outside']), fcc['small'], fcc['open_frames'])
    print('text_check pass', tc['pass'], ascii(tc['overlaps']), ascii(tc['within_0.08in_of_edge']), 'fig', tc['figure_in'])
    print('ncls', ncls, 'outside', outside)
    S.save(fig, STEM, plotted, [F2])
    cap = "\\textbf{Distribution over land cells of each U-Net's 2016 test-day mean modeled surface NO$_2$ minus CMAQ's (ppb), by CMAQ's 2016 level class (ppb), contiguous US.} The values are those mapped in Figure~\\ref{fig:F2}(c) and (d), from each U-Net at seed 0 trained without 2016, and the classes are the level classes of Figure~\\ref{fig:F2}. In each class the left half (blue outline, pale blue fill) shows the reference U-Net and the right half (vermilion outline, pale vermilion fill) the residual U-Net. Each half's width is proportional to the land cells in each 0.04 ppb bin of the difference, on one scale for the two halves of a class. Each class is scaled to its own widest bin, so widths compare within a class and not between classes. Black ticks mark each half's median (long) and its 5th and 95th percentiles (short), computed over all land cells of the class. The highest class holds few cells and is drawn with one mark per cell (squares for the reference U-Net, discs for the residual U-Net). The black dashed line marks zero difference; a thin dark gray line divides each class's two halves. The number under each class name is its count of land cells. The vertical range leaves out %d cells for the reference U-Net and %d for the residual U-Net, whose differences lie beyond it. " % (outside['P'], outside['M']) + "Grouping by CMAQ's own value lets regression toward the mean make a smoother model read low where CMAQ is high. These results are descriptive and were not specified in advance."
    alt = "Mirrored histograms of each U-Net's 2016 mean minus CMAQ's per land cell in six CMAQ level classes."
    longd = "One panel with six CMAQ level classes along the horizontal axis (under 1, 1 to 2, 2 to 4, 4 to 8, 8 to 16, and 16 ppb and over), holding %s, %s, %s, %s, %s and %s land cells. Vertical axis: the U-Net's 2016 test-day mean minus CMAQ's, minus 1 to 1 ppb, with a black dashed line at zero. In each class a histogram of the reference U-Net (blue, left) mirrors one of the residual U-Net (vermilion, right); black ticks mark each median and the 5th and 95th percentiles, and a thin dark gray line divides each class's halves. Where a bin holds very few cells its width is near zero and its outline shows as a thin colored stroke along the divider. The spreads widen from the lowest to the higher classes. The highest class shows one square or disc per cell. A key above names the halves, the ticks and the zero line. The range of plus or minus %g ppb leaves out %d cells for the reference U-Net and %d for the residual U-Net." % (tuple((C.thousands(n) for n in ncls)) + (YL, outside['P'], outside['M']))
    C.write_text(STEM, cap, alt, longd)
    print('words', len(cap.split()))
if __name__ == '__main__':
    main()
