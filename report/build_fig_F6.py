"""Draws figure fig_F6: the mean error by CMAQ decile for the two U-Nets, with the gradient-boosted trees and the weekday climatology for comparison."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import build_fig_common as C
import build_style as S
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
STEM = 'fig_F6'
PANELS = ['M', 'P']
CONTEXT = ['GBT', 'CLIM-WD']
DECILES = np.arange(1, 11)
CTX_LW = 0.8
CTX_MS = 2.8

def legacy():
    C.setup()
    rows = C.rows('deciles.json')
    srcs = [C.sealed('deciles.json')]
    mb, edges = ({}, {})
    for r in rows:
        m = r['model']
        if m not in PANELS + CONTEXT or r['seed'] not in (0, None):
            continue
        if m in PANELS and r['seed'] != 0:
            continue
        key = (m, r['fold'])
        mb.setdefault(key, {})[r['decile']] = r['MB']
        e = edges.setdefault(r['fold'], {})
        lo = (r['decile'], 'lower')
        hi = (r['decile'], 'upper')
        for kk, v in ((lo, r['lower']), (hi, r['upper'])):
            if kk in e:
                assert e[kk] == v, (r['fold'], kk)
            e[kk] = v
    for m in PANELS + CONTEXT:
        for y in S.YEARS:
            assert sorted(mb[m, y]) == list(DECILES), (m, y)
    arr = {k: np.array([v[d] for d in DECILES], dtype=float) for k, v in mb.items()}
    mean = {m: np.mean([arr[m, y] for y in S.YEARS], axis=0) for m in PANELS + CONTEXT}
    bounds = {}
    for j in range(1, 10):
        vals = []
        for y in S.YEARS:
            u = edges[y][j, 'upper']
            lw = edges[y][j + 1, 'lower']
            assert u == lw, (y, j)
            vals.append(u)
        bounds[j] = (min(vals), max(vals), vals)
    allv = np.concatenate([arr[m, y] for m in PANELS for y in S.YEARS] + [mean[m] for m in PANELS + ['GBT']])
    lo = np.floor(allv.min() / 0.2 - 1e-12) * 0.2
    hi = np.ceil(max(allv.max(), mean['CLIM-WD'].max()) / 0.2 - 1e-12) * 0.2
    hi = max(hi, 0.2)
    off_scale = {c: [int(d) for d, v in zip(DECILES, mean[c]) if v < lo or v > hi] for c in CONTEXT}
    print('off scale', off_scale)
    W = 6.5
    left, right, gapx = (0.62, 0.1, 0.5)
    pw = (W - left - right - gapx) / 2.0
    ph = 2.6
    leg_h = 0.47
    top = 0.08 + leg_h + 0.07 + 0.16
    below = 0.4
    Ht = top + ph + below + 0.08
    fig = plt.figure(figsize=(W, Ht))
    axes = []
    plotted = {'x': 'CMAQ decile 1..10', 'y': 'MB (model minus CMAQ, ppb)', 'mean_rule': 'unweighted mean over the five held-out years of the per-year MB of each decile', 'years': S.YEARS, 'per_year': {}, 'mean': {}, 'decile_bounds_ppb': {}}
    for k, m in enumerate(PANELS):
        ax = fig.add_axes([(left + k * (pw + gapx)) / W, (below + 0.08) / Ht, pw / W, ph / Ht])
        axes.append(ax)
        ax.axhline(0.0, color=S.INK, lw=C.REF_LW, linestyle=C.REF_DASH, zorder=1)
        for c in CONTEXT:
            ax.plot(DECILES, mean[c], color=S.MODEL_COLOUR[c], lw=CTX_LW, marker=S.MODEL_MARKER[c], ms=CTX_MS, mfc=S.MODEL_COLOUR[c], mec='white', mew=0.4, zorder=2)
            for d in off_scale[c]:
                v = mean[c][d - 1]
                ax.plot([d], [lo if v < lo else hi], ls='none', marker='v' if v < lo else '^', ms=5.0, mfc=S.MODEL_COLOUR[c], mec=S.MODEL_COLOUR[c], mew=0.4, clip_on=False, zorder=6)
        for y in S.YEARS:
            ax.plot(DECILES, arr[m, y], color=S.MODEL_COLOUR[m], lw=0.6, zorder=3)
            plotted['per_year']['%s_%d' % (m, y)] = arr[m, y]
        ax.plot(DECILES, mean[m], color=S.MODEL_COLOUR[m], lw=1.8, marker=S.MODEL_MARKER[m], ms=4.2, mfc=S.MODEL_COLOUR[m], mec='white', mew=0.4, zorder=4)
        ax.set_xlim(0.5, 10.5)
        ax.set_xticks(DECILES)
        ax.set_ylim(lo, hi)
        ax.set_yticks(np.round(np.arange(lo, hi + 1e-09, 0.2), 6))
        C.tag(ax, 'ab'[k], S.PLAIN[m])
        plotted['ylim'] = [float(lo), float(hi)]
    for m in PANELS + CONTEXT:
        plotted['mean'][m] = mean[m]
    for j, (a, b, v) in bounds.items():
        plotted['decile_bounds_ppb']['edge_%d_%d' % (j, j + 1)] = {'min_over_years': a, 'max_over_years': b, 'by_year': dict(zip([str(y) for y in S.YEARS], v))}
    for c in CONTEXT:
        for y in S.YEARS:
            plotted['per_year']['%s_%d (not drawn; enters the mean)' % (c, y)] = arr[c, y]
    y_mid = (below + 0.08 + ph / 2.0) / Ht
    fig.text((left - 0.48) / W, y_mid, 'Mean error (ppb)', rotation=90, ha='center', va='center', fontsize=8, color=S.INK)
    xmid = (left + (2 * pw + gapx) / 2.0) / W
    fig.text(xmid, 0.1 / Ht, 'CMAQ decile of cell-days', ha='center', va='bottom', fontsize=8, color=S.INK)
    hs, ls = ([], [])
    for m, c in zip(PANELS, CONTEXT):
        hs.append(Line2D([], [], color=S.MODEL_COLOUR[m], lw=0.6))
        ls.append('%s, each held-out year' % S.PLAIN[m])
        hs.append(Line2D([], [], color=S.MODEL_COLOUR[m], lw=1.8, marker=S.MODEL_MARKER[m], ms=4.2, mfc=S.MODEL_COLOUR[m], mec='white', mew=0.4))
        ls.append('%s, five-year mean' % S.PLAIN[m])
        hs.append(Line2D([], [], color=S.MODEL_COLOUR[c], lw=CTX_LW, marker=S.MODEL_MARKER[c], ms=CTX_MS, mfc=S.MODEL_COLOUR[c], mec='white', mew=0.4))
        ls.append('%s, five-year mean' % S.PLAIN[c])
    leg = fig.legend(hs, ls, loc='upper center', bbox_to_anchor=(xmid, (Ht - 0.08) / Ht), ncol=2, fontsize=7, frameon=False, handlelength=2.4, columnspacing=1.6, handletextpad=0.6, labelspacing=0.62, borderaxespad=0.0)
    chk = C.text_check(fig)
    plotted['off_scale_deciles_marked'] = off_scale
    plotted['text_check'] = chk
    plotted['figure_in'] = [W, Ht]
    plotted['gaps_pt'] = C.gaps_report(fig, [['key to tag line', leg, axes[0].texts[0]], ['tick labels to x label', axes[0], [t for t in fig.texts if t.get_text().startswith('CMAQ')][0]]])
    plt.close(fig)
    return (plotted, srcs, arr, mean, bounds, off_scale, lo, hi)

def main():
    import re
    import build_port_fig_F6 as PORT
    plotted, srcs, arr, mean, bounds, off_scale, lo, hi = legacy()
    if off_scale != {'GBT': [], 'CLIM-WD': [10]}:
        raise SystemExit("STOP: the caption names one off-scale point, the climatology's decile 10: %r" % off_scale)
    fig, drawn = PORT.draw(arr, mean, bounds, lo, hi)
    plotted['drawn_build1029'] = drawn
    plotted['fc_check'] = C.fc_check(fig)
    plotted['text_check'] = C.text_check(fig)
    fcc, tc = (plotted['fc_check'], plotted['text_check'])
    print('fc_check pass', fcc['pass'], 'case_pass', fcc['case_pass'], fcc['case'], fcc['overlaps'], fcc['outside'], fcc['small'])
    print('text_check pass', tc['pass'], tc['overlaps'], tc['within_0.08in_of_edge'], 'fig', tc['figure_in'])
    print('band overlap', drawn['band_overlap_de2000'])
    S.save(fig, STEM, plotted, srcs)
    cap = "\\textbf{Mean error of modeled daily mean NO$_2$ (ppb, the model minus CMAQ) in each decile of CMAQ's cell-days, contiguous US land cells of the 12 km grid, test days of the held-out years 2015 to 2019.} Deciles are set within each held-out year (10 the highest); the top axis of (a) gives each bound's range over the years. (a) Lines show the unweighted five-year mean (U-Nets at seed 0), and the light vermilion and light blue bands the residual and the reference U-Net's range over the years (lowest to highest), gray where they overlap. The weekday climatology's decile-10 mean, below the axis, is marked by an arrow with its value. (b) Decile 10 in each held-out year; a gray line joins one year's two U-Net values. The black line marks zero difference. Grouping by CMAQ's own value lets regression toward the mean make a smoother model read low where CMAQ is high. Panel (b) was not specified in advance."
    cap = re.sub('\\s+', ' ', cap).strip()
    print('caption words', len(re.sub('\\\\[a-zA-Z]+|[{}$]', ' ', cap).split()))
    alt = 'Mean error by CMAQ decile for two U-Nets with year ranges, two baselines, and decile 10 by year.'
    bl = '; '.join(('%d to %d: %.2f to %.2f' % (j, j + 1, bounds[j][0], bounds[j][1]) for j in sorted(bounds)))
    longd = "Key above, three columns, the last entry, a black line, centred in a row of its own: %s.\nPanel (a) Five-year mean and range over years, by decile: horizontal axis Decile of CMAQ's daily mean NO2 in each held-out year, ticks 1 to 10; top axis Bound between deciles, range over years (ppb), between deciles %s; vertical axis Model minus CMAQ (ppb); a line at 0. The light vermilion band and the light blue band both widen toward decile 10, where all four mean lines fall below 0; the weekday climatology's line leaves the panel at its lower edge, marked by a downward arrow and the value %s.\nPanel (b) Decile 10 by year, below panel (a) at the same width: rows 2015 to 2019 (vertical axis Held-out year); horizontal axis Model minus CMAQ, decile 10 (ppb); a disc for the residual U-Net and a square for the reference U-Net in each year, joined by a gray line; a line at 0." % ('; '.join(drawn['key']), bl, ('%.2f' % mean['CLIM-WD'][9]).replace('-', 'minus '))
    C.write_text(STEM, cap, alt, longd)
if __name__ == '__main__':
    main()
