"""Draws figure fig_F2S: each land cell's mean over the 2016 test days, emulator against CMAQ, for the two U-Nets, and the distribution of the per-cell differences."""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_fig_common as C
import build_style as S
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
C.setup()
src = {m: C.sealed('F4_mean_fold2016_%s.npy' % m) for m in ('P', 'M')}
arr = {m: np.load(p).astype(np.float64) for m, p in src.items()}
cm = arr['M'][0]
assert np.array_equal(arr['P'][0], cm)
n_cells = cm.size
diff = {m: arr[m][1] - arr[m][0] for m in ('P', 'M')}
for m in ('P', 'M'):
    assert np.allclose(diff[m], arr[m][2].astype(np.float64), atol=1e-05)
lo_all = float(min(cm.min(), arr['M'][1].min(), arr['P'][1].min()))
hi_all = float(max(cm.max(), arr['M'][1].max(), arr['P'][1].max()))
if lo_all <= 0.0:
    raise SystemExit('STOP: a cell mean at or below 0 ppb; a logarithmic axis cannot hold it')
PER_DEC = 40
SERIES = {'CMAQ': cm, 'M': arr['M'][1], 'P': arr['P'][1]}
bot = float(10.0 ** np.floor(np.log10(lo_all)))
while True:
    nxt = bot * 10.0
    if max((int((v < nxt).sum()) for v in SERIES.values())) <= 0.0001 * n_cells:
        bot = nxt
    else:
        break
below_bot = {s: int((v < bot).sum()) for s, v in SERIES.items()}
top = float(np.ceil(hi_all / 10.0) * 10.0)
k0 = int(round(np.log10(bot) * PER_DEC))
k1 = int(np.ceil(np.log10(top) * PER_DEC - 1e-09))
edges = 10.0 ** (np.arange(k0, k1 + 1) / float(PER_DEC))
assert edges[-1] >= hi_all
print('cell means %.4f to %.3f ppb; log axes %g to %g ppb; %d bins per axis; below the bottom: %s' % (lo_all, hi_all, bot, top, edges.size - 1, below_bot))

def in_axis(v):
    return np.maximum(v, edges[0])
CLASS = [1, 10, 100, 1000]
GREYS = ['#C7C7C7', '#8C8C8C', '#4D4D4D', '#000000']
CLASS_LAB = ['1 to 9', '10 to 99', '100 to 999', '1,000 or more']
H2 = {m: np.histogram2d(in_axis(cm), in_axis(arr[m][1]), bins=[edges, edges])[0] for m in ('P', 'M')}
hmax = max((int(h.max()) for h in H2.values()))
n_cls = sum((1 for c in CLASS if c <= hmax))
GREYS_USED = GREYS[:n_cls]
CLASS_USED = CLASS_LAB[:n_cls]
cmap = ListedColormap(GREYS_USED)
norm = BoundaryNorm(CLASS[:n_cls] + [1000000000.0], cmap.N)
print('largest count per bin %d; classes kept %s' % (hmax, CLASS_USED))
import build_port_fig_F2S as PF
plotted = {'bottom_ppb': bot, 'top_ppb': top, 'bins_per_decade': PER_DEC, 'bin_edges_ppb': edges, 'axes': 'logarithmic, x and y', 'n_cells': n_cells, 'cells_below_bottom_in_lowest_bin': below_bot, 'panels': {}}
names = {'P': 'Reference U-Net', 'M': 'Residual U-Net'}

def tick_txt(t):
    return '%g' % t
for m in ('P', 'M'):
    h = H2[m]
    plotted['panels'][m] = {'cells_per_bin_nonzero': int((h > 0).sum()), 'cells_drawn': int(h.sum()), 'max_cells_per_bin': int(h.max()), 'r_mean_maps': float(np.corrcoef(cm, arr[m][1])[0, 1]), 'rmse_mean_maps': float(np.sqrt(np.mean(diff[m] ** 2))), 'mean_diff': float(diff[m].mean())}
    assert int(h.sum()) == n_cells
plotted['cells_per_2d_bin'] = {'rule': 'numpy.histogram2d of (CMAQ, U-Net) 2016 cell means on bin_edges_ppb, means below the bottom in the lowest bin; counts[i][j]: i the CMAQ bin, j the U-Net bin', 'values': {m: H2[m].astype(int) for m in ('P', 'M')}}
lim = float(np.ceil(np.percentile(np.abs(np.concatenate([diff['P'], diff['M']])), 99.9) * 10) / 10)
be = np.linspace(-lim, lim, 61)
cnt = {}
for m in ('P', 'M'):
    c, _ = np.histogram(np.clip(diff[m], -lim, lim), bins=be)
    cnt[m] = c
    plotted['panels'][m]['diff_hist_counts'] = c
    plotted['panels'][m]['diff_p05_p95'] = [float(np.percentile(diff[m], 5)), float(np.percentile(diff[m], 95))]
    plotted['panels'][m]['share_abs_le_0_25'] = float(np.mean(np.abs(diff[m]) <= 0.25))
plotted['diff_edges'] = be
plotted['diff_clip_ppb'] = lim
within_txt = {}
with open(os.path.join(S.ROOT, 'results', 'report', 'numbers.tex'), encoding='ascii') as f:
    _num = f.read()
for m, key in (('M', 'map.f2.m.within'), ('P', 'map.f2.p.within')):
    v = '%.1f' % (100.0 * float(np.mean(np.abs(diff[m]) < 0.2)))
    if '\\NUMDEF{%s}{%s}' % (key, v) not in _num:
        raise SystemExit("STOP: share within 0.2 ppb %s differs from the report's %s" % (v, key))
    within_txt[m] = v
fig, H_IN, art = PF.draw(cm, {m: arr[m][1] for m in ('P', 'M')}, diff, H2, edges, GREYS_USED, CLASS_USED, cnt, be, lim, {m: plotted['panels'][m]['mean_diff'] for m in ('P', 'M')}, within_txt, plotted)
plotted['text_check'] = C.text_check(fig)
plotted['fc_check'] = C.fc_check(fig)
plotted['layout'] = {'figure_in': [6.5, H_IN]}
pm, pp = (plotted['panels']['M'], plotted['panels']['P'])
_who = {'CMAQ': 'CMAQ', 'P': 'the reference U-Net', 'M': 'the residual U-Net'}
_bl = [(n, _who[s]) for s, n in below_bot.items() if n > 0]
if not _bl:
    below_txt = ''
else:
    below_txt = ' (%s below %s~ppb are counted in the lowest bin)' % (' and '.join(('%d %s of %s' % (n, 'cell' if n == 1 else 'cells', w) for n, w in _bl)), tick_txt(bot))
mean_tex = {m: ('%g' % round(plotted['panels'][m]['mean_diff'], 3)).replace('-', '$-$') for m in ('M', 'P')}
cap = "\\textbf{Modeled mean surface NO$_2$ (ppb) of each land cell over the 2016 test days: the two U-Nets against CMAQ, contiguous US land cells of the 12~km grid, 2016 held out of training.} In (a) and (b), each of the %s cells is placed by its 2016 CMAQ mean (x) and the U-Net's mean (y), the means mapped in Figure~\\ref{fig:F2}, in square bins of one fortieth of a decade on logarithmic axes, shaded by the cells per bin in %s classes (key). Both panels share one range, and the black line is the one-to-one line. (c) The per-cell difference, U-Net minus CMAQ, in 60 bins from $-$%.1f to %.1f~ppb, the residual U-Net above the dark gray baseline and the reference U-Net below it. Short marks give each U-Net's mean difference (%s~ppb for the residual and %s~ppb for the reference U-Net), and the reference U-Net's most frequent bin lies below zero. (d) The share of land cells whose absolute difference is at or below each value, with marks at the dashed 0.2~ppb level (%s percent for the residual U-Net, %s percent for the reference U-Net), and (e) the share at or below each signed difference. (f) The median (heavy line, marks) and the 10th and 90th percentiles (thin lines) of the difference in tenth-of-a-decade bins of CMAQ's cell mean of at least 300 cells, with the cells per bin below. The black line marks zero difference, and values outside the axes are counted in the end bins. Grouping by CMAQ's own value lets regression toward the mean make a smoother model read low where CMAQ is high. These results are descriptive and were not specified in advance." % ('{:,}'.format(n_cells), {2: 'two', 3: 'three', 4: 'four'}[n_cls], lim, lim, mean_tex['M'], mean_tex['P'], within_txt['M'], within_txt['P'])
alt = 'Per-cell 2016 mean NO2 of two U-Nets against CMAQ, the spread of their differences, and the differences by CMAQ level.'
cls_txt = ', '.join(CLASS_USED[:-1]) + ' and ' + CLASS_USED[-1]
long_desc = "Six panels in three rows, with a key at the top: Residual U-Net (vermilion solid line, disc), Reference U-Net (blue dashed line, square), Zero difference (black line) and Level 0.2 (black dashed line). Row 1. Panels a (reference U-Net) and b (residual U-Net): x axis CMAQ 2016 cell mean NO2 (ppb), y axis U-Net 2016 cell mean NO2 (ppb), both logarithmic over one range shared by the two panels, gray bins in classes of %s cells per bin and a black one-to-one line on top; key under them: Cells per bin, %s, and One-to-one. Panel c, Per-cell difference, 2016: back-to-back step histograms of U-Net minus CMAQ (ppb), the residual U-Net above a dark gray baseline and the reference U-Net below it, y axis Cells, a black zero line; each U-Net's mean difference marked by a short stroke on the baseline, keyed under the panel as Mean difference (residual U-Net %s ppb, reference U-Net %s ppb). Row 2. Panel d, Absolute difference: share of land cells (%%) against the absolute difference (ppb) from 0 to 0.4, a black dashed line at 0.2 ppb with the shares %s%% (residual) and %s%% (reference) marked; panel e, Signed difference: the same share against U-Net minus CMAQ from minus 0.4 to 0.4 ppb, a black zero line. Row 3. Panel f: median and 10th and 90th percentiles of U-Net minus CMAQ (ppb) against CMAQ 2016 cell mean NO2 (ppb, logarithmic, 0.1-decade bins), a black zero line, and a gray bar strip of cells per bin below, its ticks and title Cells at the right." % (cls_txt, cls_txt, C.fmt_plain(pm['mean_diff'], 3), C.fmt_plain(pp['mean_diff'], 3), within_txt['M'], within_txt['P'])
plotted['classes_kept'] = {'labels': CLASS_USED, 'colors': GREYS_USED, 'largest_count_per_bin': hmax}
print('caption words', len(cap.split()))
C.write_text('fig_F2S', cap, alt, long_desc)
S.save(fig, 'fig_F2S', plotted, list(src.values()))
print('F2S built; top', top, 'clip', lim, 'R', pp['r_mean_maps'], pm['r_mean_maps'], 'size 6.5', round(H_IN, 3))
print(ascii(['text check', plotted['text_check']['pass'], plotted['text_check']['overlaps'], plotted['text_check']['within_0.08in_of_edge'], plotted['text_check']['text_blocks_nearer_than_4pt'][:8]]))
print(ascii({k: v for k, v in plotted['fc_check'].items() if v}))
