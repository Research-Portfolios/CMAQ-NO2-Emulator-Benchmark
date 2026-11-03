"""Draws figure fig_ad8_tail: each U-Net minus CMAQ by CMAQ's daily value and by the emulator's own value, as binned medians and 5th to 95th percentiles from the counts of fig_F1."""
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
STEM = 'fig_ad8_tail'
F1 = os.path.join(S.FIG, 'fig_F1_plotted.json')
BW = 0.5
PS = (0.05, 0.5, 0.95)
CUT_LO = 26.0
STRIP_EDGES = [100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0, 100000000.0]
FEW = 1000.0
YLO, YHI = (-10.0, 8.0)
ORDER = ['M', 'P']

def qlin(w, lower, p):
    tot = float(w.sum())
    if tot <= 0:
        return np.nan
    cum = np.concatenate([[0.0], np.cumsum(w)])
    t = p * tot
    k = int(np.searchsorted(cum, t, side='right') - 1)
    k = min(max(k, 0), len(w) - 1)
    while w[k] == 0 and k > 0:
        k -= 1
    return lower[k] + BW * (t - cum[k]) / w[k]

def conditional(c, axis):
    nb = c.shape[0]
    q = np.full((nb, 3), np.nan)
    n = np.zeros(nb)
    for b in range(nb):
        if axis == 0:
            w = c[b, :]
            cen = (np.arange(nb) - b) * BW
        else:
            w = c[::-1, b]
            cen = (b - np.arange(nb)[::-1]) * BW
        n[b] = w.sum()
        if n[b] > 0:
            q[b] = [qlin(w, cen - BW / 2.0, p) for p in PS]
    return (q, n)

class _Pct:

    def __init__(self, col, ls):
        self.col, self.ls = (col, ls)

class _PctHandler(HandlerBase):

    def create_artists(self, legend, h, xd, yd, w, ht, fs, trans):
        y = (ht - yd) / 2.0
        return [Line2D([-xd, -xd + w], [y + dy, y + dy], color=h.col, lw=0.7, ls=h.ls, transform=trans) for dy in (2.6, -2.6)]

class _Open:
    pass

class _OpenHandler(HandlerBase):

    def create_artists(self, legend, h, xd, yd, w, ht, fs, trans):
        y = (ht - yd) / 2.0
        span = Line2D([-xd, -xd + w], [y, y], color='none', lw=0, transform=trans)
        out = [span]
        for f, m in ((0.3, 'M'), (0.72, 'P')):
            out.append(Line2D([-xd + f * w], [y], ls='none', marker=S.MODEL_MARKER[m], ms=3.2, mfc='white', mec=S.MODEL_COLOUR[m], mew=0.7, transform=trans))
        return out

def main():
    with open(F1, encoding='utf-8') as f:
        f1 = json.load(f)['plotted']
    assert f1['bin_ppb'] == BW and f1['A_ppb'] == 30
    pan = f1['panels']
    counts = {}
    for m in ORDER:
        tot = None
        for y in S.YEARS:
            c = pan['%s_%d' % (m, y)]['counts_drawn']
            if isinstance(c, str):
                c = json.loads(c)
            c = np.asarray(c, dtype=float)
            tot = c if tot is None else tot + c
        counts[m] = tot
    NB = counts['M'].shape[0]
    xc = (np.arange(NB) + 0.5) * BW
    res = {}
    for pi, axis in enumerate((0, 1)):
        for m in ORDER:
            q, n = conditional(counts[m], axis)
            res['ab'[pi], m] = (q, n)
    for k, (q, n) in res.items():
        assert n.min() >= STRIP_EDGES[0] and n.max() < STRIP_EDGES[-1], k
    strip_class = {k: np.searchsorted(STRIP_EDGES, v[1], side='right') - 1 for k, v in res.items()}
    ramp = list(S.GREYS6)
    ramp_m = C.ramp_measure(ramp)
    assert ramp_m['pass'], ramp_m
    C.setup()
    W = 6.5
    L, R, GUT = (0.86, 0.16, 0.62)
    aw = (W - L - R - GUT) / 2.0
    ah = 2.55
    sh, sg, s0 = (0.12, 0.05, 0.08)

    def draw(top, H):
        fig = S.plt.figure(figsize=(W, H))
        axes, strips = ([], [])
        bot = H - top - ah
        for pi, axis in enumerate((0, 1)):
            x0 = L + pi * (aw + GUT)
            ax = fig.add_axes([x0 / W, bot / H, aw / W, ah / H])
            axes.append(ax)
            tagk = 'ab'[pi]
            ax.add_patch(Rectangle((CUT_LO, YLO), NB * BW - CUT_LO, YHI - YLO, facecolor='white', edgecolor='#9A9A9A', hatch='////', lw=0, zorder=0.5))
            ax.axhline(0.0, color=S.REF_INK, lw=0.6, zorder=1)
            for m in reversed(ORDER):
                q, n = res[tagk, m]
                col, ls = (S.MODEL_COLOUR[m], S.MODEL_LS[m])
                for k in (0, 2):
                    ax.plot(xc, q[:, k], color=col, lw=0.7, ls=ls, zorder=2)
                ax.plot(xc, q[:, 1], color=col, lw=1.5, ls=ls, zorder=3)
                few = n < FEW
                ax.plot(xc[few], q[few, 1], ls='none', marker=S.MODEL_MARKER[m], ms=3.2, mfc='white', mec=col, mew=0.7, zorder=4)
            ax.set_xlim(0, NB * BW)
            ax.set_ylim(YLO, YHI)
            ax.set_xticks(np.arange(0, 31, 5))
            ax.set_xticklabels([])
            yt = np.arange(-8, 9, 2)
            ax.set_yticks(yt)
            ax.set_yticklabels([('%d' % v).replace('-', S.MINUS) for v in yt])
            if pi == 0:
                ax.set_ylabel('U-Net minus CMAQ (ppb)', fontsize=S.FS_LABEL)
            C.frame(ax)
            C.tag(ax, tagk, "Binned by CMAQ's daily value" if axis == 0 else "Binned by the U-Net's daily value")
            for si, m in enumerate(ORDER):
                yb = bot - s0 - (si + 1) * sh - si * sg
                sax = fig.add_axes([x0 / W, yb / H, aw / W, sh / H])
                strips.append(sax)
                cls = strip_class[tagk, m]
                for b in range(NB):
                    sax.add_patch(Rectangle((b * BW, 0), BW, 1, facecolor=ramp[cls[b]], edgecolor=ramp[cls[b]], lw=0.2))
                sax.set_xlim(0, NB * BW)
                sax.set_ylim(0, 1)
                sax.set_xticks(np.arange(0, 31, 5))
                if si == 0:
                    sax.set_xticklabels([])
                    sax.tick_params(axis='x', length=0)
                else:
                    sax.set_xlabel('CMAQ daily value (ppb)' if axis == 0 else 'U-Net daily value (ppb)', fontsize=S.FS_LABEL)
                C.frame(sax)
                sax.set_yticks([0.5])
                sax.set_yticklabels(['Residual' if m == 'M' else 'Reference'], fontsize=S.FS_TICK)
                sax.get_yticklabels()[0].set_color(S.MODEL_TEXT[m])
                C.house_ticks(sax, 'y')
        C.block_title(fig, strips[:2], 'U-Net', axis='y')

        def med(m):
            return Line2D([], [], color=S.MODEL_COLOUR[m], lw=1.5, ls=S.MODEL_LS[m])

        def pct(m):
            return _Pct(S.MODEL_COLOUR[m], S.MODEL_LS[m])
        hatch = Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='#9A9A9A', hatch='////', lw=0.4)
        zero = Line2D([], [], color=S.REF_INK, lw=0.6)
        rows = [[(med('M'), '%s, median' % S.PLAIN['M']), (pct('M'), '5th to 95th percentile'), (hatch, 'Near the 30 ppb cut')], [(med('P'), '%s, median' % S.PLAIN['P']), (pct('P'), '5th to 95th percentile'), (zero, 'Zero difference')], [(_Open(), "Median of a bin in the strip's lightest class")]]
        legs = C.house_key(fig, axes, None, None, rows=rows, side='above', handler_map={_Pct: _PctHandler(), _Open: _OpenHandler()})
        for lg_ in legs:
            for t_ in lg_.get_texts():
                for m_ in ('M', 'P'):
                    if t_.get_text().startswith(S.PLAIN[m_]):
                        t_.set_color(S.MODEL_TEXT[m_])
        cw, chh = (0.36, 0.12)
        rect = C.key_rect(fig, axes + strips, 6 * cw, chh, side='below', gap_pt=S.CLEAR_RATIO * S.AXIS_LABELPAD_PT + S.GAP_TOL_PT)
        sk = fig.add_axes(rect)
        for k, colr in enumerate(ramp):
            sk.add_patch(Rectangle((k, 0), 1, 1, facecolor=colr, edgecolor='none', lw=0))
        sk.set_xlim(0, 6)
        sk.set_ylim(0, 1)
        sk.set_yticks([])
        sk.set_xticks(range(7))
        sk.set_xticklabels(['$10^{%d}$' % e for e in range(2, 9)], fontsize=S.FS_LABEL)
        C.house_ticks(sk, 'x')
        C.frame(sk)
        bk = C.bar_key(fig, sk, 'Cell-days per 0.5 ppb bin')
        return (fig, axes, strips, legs, bk)
    MARG = 0.1
    top0, H0 = (1.2, 6.0)
    fig, axes, strips, legs, bk = draw(top0, H0)
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    S.plt.close(fig)
    top = top0 - (H0 - tb.y1) + MARG
    H = round(tb.y1 - tb.y0 + 2 * MARG, 3)
    fig, axes, strips, legs, bk = draw(top, H)
    plotted = {'label': 'DESC, added after the plan was fixed', 'source': 'figures\\fig_F1_plotted.json panels.<P|M>_<year>.counts_drawn, five years summed', 'bin_ppb': BW, 'cut_ppb': 30, 'bin_centres_ppb': xc, 'derivation': 'per conditioning bin b: (a) emulator bins j, difference (j-b)*0.5; (b) CMAQ bins i, difference (b-i)*0.5; percentiles linear within 0.5 ppb bins', 'percentiles': list(PS), 'hatch_from_ppb': CUT_LO, 'few_threshold_cell_days': FEW, 'y_limits_ppb': [YLO, YHI], 'strip_class_edges': STRIP_EDGES, 'strip_ramp': ramp, 'strip_ramp_measure': ramp_m, 'figure_in': [W, H]}
    for (tk, m), (q, n) in sorted(res.items()):
        key = '%s_%s' % (tk, m)
        plotted['p05_ppb_' + key] = q[:, 0]
        plotted['median_ppb_' + key] = q[:, 1]
        plotted['p95_ppb_' + key] = q[:, 2]
        plotted['cell_days_per_bin_' + key] = n.astype(int)
        plotted['strip_class_' + key] = strip_class[tk, m]
        plotted['open_mark_bins_' + key] = np.nonzero(n < FEW)[0]
    plotted['fc_check'] = C.fc_check(fig)
    plotted['text_check'] = C.text_check(fig)
    fcc, tc = (plotted['fc_check'], plotted['text_check'])
    print('fc_check pass', fcc['pass'], 'case_pass', fcc['case_pass'], fcc['case'], fcc['overlaps'], fcc['outside'], fcc['small'], fcc['open_frames'])
    print('text_check pass', tc['pass'], tc['overlaps'], tc['within_0.08in_of_edge'], 'fig', tc['figure_in'])
    print('near', ascii(tc['text_blocks_nearer_than_4pt'][:12]))
    print('ramp', ramp_m)
    S.save(fig, STEM, plotted, [F1])
    cap = "\\textbf{Modeled surface NO$_2$ of each U-Net minus CMAQ (ppb) by CMAQ's daily value (a) and by the U-Net's own daily value (b), land cell-days of the test days of the five held-out years (2015 to 2019) pooled.} The cell-days are those of Figure~1, in 0.5 ppb bins with both values under 30 ppb. The bins condition on CMAQ's daily value in (a) and on the U-Net's own value in (b). In each bin, the thick line is the median of the U-Net minus CMAQ and the two thin lines of the same color and style bound its 5th to 95th percentiles (the spread of the cell-days); vermilion solid lines and discs show the residual U-Net, blue dashed lines and squares the reference U-Net. These lines, from Figure~1's joint bins, are approximate to about 0.5~ppb. Open marks are medians of bins in the lightest class of the strip, which rest on few cell-days. The black line marks zero difference. The hatched band marks bins near the 30 ppb cut, where leaving out cell-days with either value at or above 30 ppb biases the spread. The two strips under each panel give the cell-days per bin for the residual U-Net (upper) and the reference U-Net (lower), in gray classes by powers of ten (key below). Grouping by CMAQ's own value, as in (a), lets regression toward the mean make a smoother model read low where CMAQ is high; grouping by the U-Net's own value, as in (b), can tilt the medians the other way. These results are descriptive and were not specified in advance."
    alt = "Two panels of each U-Net minus CMAQ by CMAQ's daily value and by the U-Net's own value, with medians and percentile lines."
    longd = "Two side-by-side panels, each with its own vertical ticks on one scale, U-Net minus CMAQ in ppb, from minus 10 to 8, titled once at the far left. Panel (a) bins the cell-days by CMAQ's daily value, 0 to 30 ppb; panel (b) by the U-Net's own daily value. In each panel the residual U-Net (vermilion, solid, discs) and the reference U-Net (blue, dashed, squares) have a thick median line and two thin lines at the 5th and 95th percentiles; open marks show medians of bins with few cell-days; a black line marks zero and a hatched band covers the bins near the 30 ppb cut. In (a) the medians fall below zero as CMAQ's value rises; in (b) the reference U-Net's median rises above zero with the U-Net's own value while the residual U-Net's stays near zero up to the hatched band. Under each panel two gray strips give the cell-days per bin for the residual (upper) and reference (lower) U-Net, from over ten million in the lowest bins to a few hundred or fewer near 30 ppb; a key below gives the gray classes by powers of ten."
    C.write_text(STEM, cap, alt, longd)
    print('words', len(cap.split()))
if __name__ == '__main__':
    main()
