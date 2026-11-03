"""Draws figure fig_F5: RMSE and skill score against the weekday climatology by held-out year for nine models, with the spread over training seeds of the two U-Nets."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import build_fig_common as C
import build_style as S
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
STEM = 'fig_F5'
LEGEND_ORDER = ['M', 'P', 'PIX-R', 'GBT', 'GBT-RIDGE', 'MET-REG', 'CLIM-WD', 'CLIM-EM', 'CLIM-C']
DODGE_ORDER = ['M', 'PIX-R', 'CLIM-EM', 'GBT', 'GBT-RIDGE', 'CLIM-C', 'MET-REG', 'CLIM-WD', 'P']
LISTED_ORDER = ['M', 'P', 'GBT', 'MET-REG', 'PIX-R', 'GBT-RIDGE', 'CLIM-EM', 'CLIM-WD', 'CLIM-C']
STEP = 0.0
BAR_OFF = 0.13
SEED_YEARS = (2016, 2019)
MS = 4.2
LINE_LW = 0.6
LW = {'M': 1.5, 'P': 1.5, 'GBT': 1.0}
MSZ = {'M': MS, 'P': MS, 'GBT': 3.6}
LSTYLE = {'M': '-', 'P': '-', 'GBT': '-', 'PIX-R': (0, (4.0, 1.5)), 'GBT-RIDGE': (0, (1.0, 1.2)), 'MET-REG': (0, (5.0, 1.5, 1.0, 1.5)), 'CLIM-WD': (0, (2.5, 1.5)), 'CLIM-EM': (0, (6.0, 2.0)), 'CLIM-C': (0, (1.0, 1.5, 1.0, 1.5, 4.0, 1.5))}
ZO = {'M': 6, 'P': 5, 'GBT': 4}
LABEL_GAP_PT = 11.0

def spread(t, gap, lo, hi):
    order = sorted(range(len(t)), key=lambda i: t[i])
    y = [float(t[i]) for i in order]
    for _ in range(500):
        moved = False
        for j in range(1, len(y)):
            d = y[j] - y[j - 1]
            if d < gap - 1e-09:
                k = (gap - d) / 2.0
                y[j - 1] -= k
                y[j] += k
                moved = True
        y[0] = max(y[0], lo)
        y[-1] = min(y[-1], hi)
        if not moved:
            break
    out = [0.0] * len(t)
    for j, i in enumerate(order):
        out[i] = y[j]
    return out

def nice(lo, hi, step):
    return (np.floor(lo / step - 1e-12) * step, np.ceil(hi / step - 1e-12) * step)

def legacy():
    C.setup()
    mets = C.rows('metrics.json')
    srcs = [C.sealed('metrics.json')]
    pts = {}
    seeds = {}
    for r in mets:
        if r['stratum_kind'] != 'all' or r['status'] != 'RUN':
            continue
        m = r['model']
        if m not in LEGEND_ORDER:
            continue
        if r['seed'] in (0, None):
            key = (m, r['fold'])
            assert key not in pts, key
            pts[key] = r
        if m in ('M', 'P') and r['seed'] is not None and (r['fold'] in SEED_YEARS):
            seeds.setdefault((m, r['fold']), []).append(r)
    for m in LEGEND_ORDER:
        for y in S.YEARS:
            assert (m, y) in pts, (m, y)
            if pts[m, y]['kind'] == 'network':
                assert pts[m, y]['seed'] == 0
            else:
                assert pts[m, y]['kind'] == 'baseline'
                assert pts[m, y]['seed'] is None
    ss_mean = {m: float(np.mean([pts[m, y]['SS'] for y in S.YEARS])) for m in LEGEND_ORDER}
    LEGEND_ORDER[:] = sorted(LEGEND_ORDER, key=lambda m: -ss_mean[m])
    print('key order (mean skill score, highest first):', ['%s %.3f' % (m, ss_mean[m]) for m in LEGEND_ORDER])
    if LEGEND_ORDER != LISTED_ORDER:
        print('the key order differs from the listed order', LISTED_ORDER)
    off = {m: (i - (len(DODGE_ORDER) - 1) / 2.0) * STEP for i, m in enumerate(DODGE_ORDER)}
    bar_x = {'M': off['M'] - BAR_OFF, 'P': off['P'] + BAR_OFF}
    W = 6.5
    lab_w = 1.62
    left, right, gapx = (0.55, 0.08, 0.55)
    pw = (W - left - right - gapx - lab_w) / 2.0
    ph = 3.0
    top = 0.08 + 0.2
    below = 0.4
    Ht = top + ph + below + 0.08
    fig = plt.figure(figsize=(W, Ht))
    axes = []
    plotted = {'x': 'held-out year + offset (offset carries no value)', 'offsets': off, 'seed_bar_x_offsets': bar_x, 'panels': {}}
    specs = [('RMSE', 'Root mean square error', 'RMSE (ppb)', 0.2), ('SS', 'Skill against the weekday climatology', 'Skill score', 0.2)]
    seed_ranges = {}
    for k, (fld, name, ylab, step) in enumerate(specs):
        ax = fig.add_axes([(left + k * (pw + gapx)) / W, (below + 0.08) / Ht, pw / W, ph / Ht])
        axes.append(ax)
        pan = {'field': fld, 'models': {}, 'seed_bars': {}}
        vals = []
        for m in LEGEND_ORDER:
            xs = np.array([y + off[m] for y in S.YEARS])
            ys = np.array([pts[m, y][fld] for y in S.YEARS], dtype=float)
            vals.extend(ys.tolist())
            joined = not (fld == 'SS' and m == 'CLIM-WD')
            z = ZO.get(m, 2)
            if joined:
                ax.plot(xs, ys, color=S.MODEL_COLOUR[m], lw=LW.get(m, 0.7), linestyle=LSTYLE[m], zorder=z)
            ax.plot(xs, ys, linestyle='none', marker=S.MODEL_MARKER[m], ms=MSZ.get(m, 3.0), mfc=S.MODEL_COLOUR[m], mec='white', mew=0.4, zorder=z + 0.5)
            pan['models'][m] = {'plain': S.PLAIN[m], 'x': xs, 'y': ys, 'joining_line': joined, 'seed': pts[m, S.YEARS[0]]['seed'], 'n': [pts[m, y]['n'] for y in S.YEARS]}
        for m in ('M', 'P'):
            for y in SEED_YEARS:
                rs = seeds[m, y]
                v = np.array([r[fld] for r in rs], dtype=float)
                lo, hi = (float(v.min()), float(v.max()))
                mid = (lo + hi) / 2.0
                ax.errorbar([y + bar_x[m]], [mid], yerr=[[mid - lo], [hi - mid]], fmt='none', ecolor=S.MODEL_COLOUR[m], elinewidth=1.0, capsize=0, zorder=7)
                vals.extend([lo, hi])
                pan['seed_bars']['%s_%d' % (m, y)] = {'x': y + bar_x[m], 'min': lo, 'max': hi, 'seeds': sorted((int(r['seed']) for r in rs)), 'values': {str(r['seed']): r[fld] for r in rs}}
                seed_ranges[fld, m, y] = (hi - lo, len(rs))
        lo, hi = nice(min(vals), max(vals), step)
        pad = 0.03 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_yticks(np.round(np.arange(lo, hi + 1e-09, step), 6))
        ax.set_xlim(S.YEARS[0] - 0.4, S.YEARS[-1] + 0.4)
        ax.set_xticks(S.YEARS)
        ax.set_xticklabels([str(y) for y in S.YEARS])
        seq = [(m, ZO.get(m, 2) + 0.5, i) for i, m in enumerate(LEGEND_ORDER)]
        pan['outlined_on_top'] = []
        for yi, y in enumerate(S.YEARS):
            pt_y = {m: ax.transData.transform((y, pan['models'][m]['y'][yi]))[1] * 72.0 / fig.dpi for m in LEGEND_ORDER}
            for m, zm, im in seq:
                cover = [n for n, zn, i_n in seq if n != m and (zn, i_n) > (zm, im) and (abs(pt_y[n] - pt_y[m]) < (MSZ.get(n, 3.0) + MSZ.get(m, 3.0)) / 2.0)]
                if cover:
                    ax.plot([pan['models'][m]['x'][yi]], [pan['models'][m]['y'][yi]], linestyle='none', marker=S.MODEL_MARKER[m], ms=MSZ.get(m, 3.0), mfc='none', mec=S.MODEL_COLOUR[m], mew=0.9, zorder=8)
                    pan['outlined_on_top'].append([m, y, cover])
        if fld == 'SS':
            ax.axhline(0.0, color=S.INK, lw=C.REF_LW, linestyle=C.REF_DASH, zorder=1)
        ax.set_ylabel(ylab, fontsize=8)
        C.tag(ax, 'ab'[k], name)
        pan['ylim'] = list(ax.get_ylim())
        plotted['panels'][fld] = pan
    xmid = (left + (2 * pw + gapx) / 2.0) / W
    fig.text(xmid, 0.1 / Ht, 'Held-out year', ha='center', va='bottom', fontsize=8, color=S.INK)
    from matplotlib.lines import Line2D as _L
    axb = axes[1]
    fig.canvas.draw()
    k_pt = 72.0 / fig.dpi
    tgt = []
    for m in LEGEND_ORDER:
        yv = float(pts[m, S.YEARS[-1]]['SS'])
        tgt.append(axb.transData.transform((S.YEARS[-1], yv))[1] * k_pt)
    y0b = axb.transAxes.transform((0, 0))[1] * k_pt
    y1b = axb.transAxes.transform((0, 1))[1] * k_pt
    pos = spread(tgt, LABEL_GAP_PT, y0b, y1b)
    xr = axb.transAxes.transform((1, 0))[0] * k_pt
    x_a, x_b, x_t = (xr + 2.0, xr + 14.0, xr + 17.0)
    direct = {}
    for m, ty, py in zip(LEGEND_ORDER, tgt, pos):
        fig.add_artist(_L([x_a / 72.0 / W, x_b / 72.0 / W], [ty / 72.0 / Ht, py / 72.0 / Ht], transform=fig.transFigure, color=S.MODEL_COLOUR[m], lw=min(LW.get(m, 0.7), 1.0), linestyle=LSTYLE[m]))
        fig.text(x_t / 72.0 / W, py / 72.0 / Ht, S.PLAIN[m], ha='left', va='center', fontsize=7, color=S.INK)
        direct[m] = {'target_pt': ty, 'label_pt': py, 'shift_pt': py - ty}
    plotted['direct_labels_b'] = direct
    fig.canvas.draw()
    close = {}
    for k, ax in enumerate(axes):
        pan = plotted['panels'][specs[k][0]]
        P = []
        for m, d in pan['models'].items():
            for x, y in zip(d['x'], d['y']):
                P.append((m, ax.transData.transform((x, y))))
        best = (1000000000.0, None)
        for i in range(len(P)):
            for j in range(i + 1, len(P)):
                dd = np.hypot(*P[i][1] - P[j][1]) * 72.0 / fig.dpi
                if dd < best[0]:
                    best = (dd, (P[i][0], P[j][0]))
        close[specs[k][0]] = {'min_centre_distance_pt': round(best[0], 2), 'pair': best[1], 'marker_size_pt': MS}
    plotted['marker_spacing'] = close
    bar_pt = {}
    for k, ax in enumerate(axes):
        for key, b in plotted['panels'][specs[k][0]]['seed_bars'].items():
            p0 = ax.transData.transform((b['x'], b['min']))
            p1 = ax.transData.transform((b['x'], b['max']))
            b['drawn_length_pt'] = float(abs(p1[1] - p0[1]) * 72.0 / fig.dpi)
            bar_pt['%s %s' % (specs[k][0], key)] = b['drawn_length_pt']
    n_short = sum((1 for v in bar_pt.values() if v < MS))
    plotted['seed_bars_shorter_than_marker'] = [n_short, len(bar_pt)]
    plotted['seed_bar_check'] = {'marker_size_pt': MS, 'rule': 'a bar is shorter than a marker is wide when its drawn length (pt, at the printed size) is below the marker size (pt)', 'drawn_length_pt': bar_pt, 'n_shorter': n_short, 'n_bars': len(bar_pt)}
    plotted['key_order'] = {'models_down_columns': LEGEND_ORDER, 'mean_skill_score': ss_mean, 'listed_order_J37': LISTED_ORDER, 'equal_to_listed_order': LEGEND_ORDER == LISTED_ORDER}
    chk = C.text_check(fig)
    plotted['text_check'] = chk
    plotted['figure_in'] = [W, Ht]
    plotted['gaps_pt'] = C.gaps_report(fig, [['tick labels to x label', axes[0], [t for t in fig.texts if t.get_text().startswith('Held')][0]]])
    plt.close(fig)
    return (plotted, srcs, pts, seeds, seed_ranges, ss_mean)

def main():
    import re
    import build_port_fig_F5 as PORT
    plotted, srcs, pts, seeds, seed_ranges, ss_mean = legacy()
    fig, drawn, new = PORT.draw(plotted, pts)
    plotted['drawn_build1029'] = drawn
    plotted.update(new)
    plotted['fc_check'] = C.fc_check(fig)
    plotted['text_check'] = C.text_check(fig)
    fcc, tc = (plotted['fc_check'], plotted['text_check'])
    print('fc_check pass', fcc['pass'], 'case_pass', fcc['case_pass'], fcc['case'], fcc['overlaps'], fcc['outside'], fcc['small'])
    print('text_check pass', tc['pass'], tc['overlaps'], tc['within_0.08in_of_edge'], 'fig', tc['figure_in'])
    print('nv5', {m: [round(v, 4) for v in vv] for m, vv in new['unbiased_rmse_ppb'].items()})
    S.save(fig, STEM, plotted, srcs)
    for m in S.MODEL_ORDER:
        r = [pts[m, y]['RMSE'] for y in S.YEARS]
        if not all((r[0] > v for v in r[1:])):
            raise SystemExit('STOP: RMSE of %s not highest in 2015' % m)
    ns = {(m, y): seed_ranges['RMSE', m, y][1] for m in ('M', 'P') for y in SEED_YEARS}

    def nword(n):
        return {2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'}.get(int(n), '%d' % n)

    def seedtxt(m):
        a, b = (ns[m, 2016], ns[m, 2019])
        if a == b:
            return '%s seeds in each year' % nword(a)
        return '%s seeds in 2016 and %s in 2019' % (nword(a), nword(b))
    grp = [g for g in drawn['d_label_groups']['M'] if len(g) > 1]
    shared = ''
    if grp:
        shared = '; coinciding marks share one label'
    cap = '\\textbf{Skill score, RMSE and mean error of modeled daily mean NO$_2$ against CMAQ by held-out year, 2015 to 2019, test days, contiguous US land cells of the 12 km grid.} Networks at seed 0; baselines one fit per fold. (a) Skill score against the weekday climatology; the large mark is the five-year mean, small gray dots the single years and the pale band their range (lowest to highest). (b) Every seed, 2016 and 2019 (residual U-Net, %s; reference U-Net, %s), with seed 0 filled. (c) RMSE (ppb), offset within years. (d) Mean error against centered RMSE (Equation~\\eqref{eq:stats}; ppb) of the two U-Nets, the gradient-boosted trees and the weekday climatology, with marks labeled by year (two digits, each joined to its mark by a gray line%s). Gray rings join points of equal RMSE, with their values (ppb) on the scale over the panel, and mean error is drawn at 1.5 times the scale of centered RMSE. The inset magnifies the boxed area. The mean error is a domain mean; urban and rural errors can cancel. Panels (b) and (d) were not specified in advance.' % (seedtxt('M'), seedtxt('P'), shared)
    cap = re.sub('\\s+', ' ', cap).strip()
    print('caption words', len(re.sub('\\\\[a-zA-Z]+|[{}$]', ' ', cap).split()))
    alt = 'Four panels: skill score by model with year ranges, U-Net seeds, RMSE by held-out year, and mean error against centered RMSE.'
    PL = PORT.NAME
    longd = "Panel (a) Skill score by model: one row per model, from the top %s; vertical axis Model; horizontal axis Skill score; each row a large mark at the five-year mean, small filled gray dots for each year and a pale band over their range; a vertical line at 0, labeled Zero skill at its upper end.\nPanel (b) Every seed: vertical axis Model and held-out year, rows 2016 and 2019 under Residual U-Net, then 2016 and 2019 under Reference U-Net; horizontal axis Skill score; open marks for each seed, filled for seed 0.\nPanel (c) RMSE by held-out year: vertical axis RMSE (ppb); horizontal axis Held-out year, 2015 to 2019; each model a line of marks, set apart within each year. The key at the right of (c) and (d) names the nine models in the order of (a); under it a solid black line is keyed Zero mean error.\nPanel (d) Mean error and centered RMSE: horizontal axis Centered RMSE (ppb), vertical axis Mean error (ppb); gray rings of equal RMSE, with values 0.50, 0.75, 1.00 and 1.25 on a scale over the frame titled Equal RMSE (ppb), each value over a tick where its ring meets the top of the frame; a line at 0, keyed Zero mean error under the models' key; marks of %s, each labeled by its year's last two digits and joined to it by a short gray line; a gray box around the crowded marks of the U-Nets and the trees near zero mean error, drawn magnified in an inset in the lower middle, joined to the box by two gray lines; the marks in the box are labeled in the inset; mean error is drawn at 1.5 times the scale of centered RMSE, here and in the inset, so the rings appear as stretched arcs." % ('; '.join((PL[m] for m in S.MODEL_ORDER)), ', '.join((PL[m] for m in PORT.D4)))
    C.write_text(STEM, cap, alt, longd)
if __name__ == '__main__':
    main()
