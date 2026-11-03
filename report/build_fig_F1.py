"""Draws figure fig_F1: two-dimensional histograms of emulated against CMAQ daily mean NO2 for each held-out year, for the reference and the residual U-Net, with the 1:1 line."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import build_fig_common as C
import build_style as S
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
STEM = 'fig_F1'
ROWS = ['P', 'M']
BIN = 0.5
CLASS_LABELS = ['1 to 9', '10 to 99', '100 to 999', '1,000 to 9,999', '10,000 to 99,999', '100,000 or more']

def main():
    C.setup()
    srcs = []
    H = {}
    for y in S.YEARS:
        for m in ROWS:
            n = 'F3_hist_fold%d_%s.npy' % (y, m)
            p = C.sealed(n)
            srcs.append(p)
            h = np.load(p)
            assert h.shape == (200, 200) and h.dtype == np.int64
            H[y, m] = h
    diag = C.rows('diagnostics.json')
    srcs.append(C.sealed('diagnostics.json'))
    out100 = {}
    for r in diag:
        if 'F3_outside_0_100' in r and r['model'] in ROWS:
            out100[r['fold'], r['model']] = int(r['F3_outside_0_100'])
    mets = C.rows('metrics.json')
    srcs.append(C.sealed('metrics.json'))
    nrow = {}
    for r in mets:
        if r['stratum_kind'] == 'all' and r['model'] in ROWS and (r['seed'] == 0):
            nrow[r['fold'], r['model']] = (int(r['n']), int(r['cells']))
    cen = (np.arange(200) + 0.5) * BIN
    FIT = {}
    for key, h in H.items():
        ii, jj = np.nonzero(h)
        cnt = h[ii, jj].astype(np.float64)
        s_fit, b_fit = np.polyfit(cen[ii], cen[jj], 1, w=np.sqrt(cnt))
        FIT[key] = (float(s_fit), float(b_fit))
    print('fit slopes', {'%s_%d' % (k[1], k[0]): round(v[0], 4) for k, v in sorted(FIT.items())})
    pool = np.zeros(200, dtype=np.int64)
    for h in H.values():
        pool += h.sum(axis=1) + h.sum(axis=0)
    cum = np.cumsum(pool)
    k99 = int(np.searchsorted(cum, 0.9999 * pool.sum(), side='left'))
    upper = (k99 + 1) * BIN
    A = int(np.ceil(upper / 10.0 - 1e-12) * 10)
    kA = int(round(A / BIN))
    beyond = {}
    tot = {}
    for key, h in H.items():
        inside_total = int(h.sum())
        n_out = out100[key]
        total = inside_total + n_out
        tot[key] = total
        beyond[key] = total - int(h[:kA, :kA].sum())
        assert total == nrow[key][0], key
    n_all = sum(tot.values())
    b_all = sum(beyond.values())
    out_all = sum(out100.values())
    k1 = int(round(1.0 / BIN))
    share_lt1 = {}
    for y in S.YEARS:
        mp = H[y, 'P'].sum(axis=1)
        mm = H[y, 'M'].sum(axis=1)
        if not np.array_equal(mp, mm):
            raise SystemExit("STOP: CMAQ's margin differs between the rows in %d" % y)
        share_lt1[y] = 100.0 * float(mp[:k1].sum()) / float(tot[y, 'P'])
    print('share of cell-days with CMAQ below 1 ppb:', share_lt1)
    W = 6.5
    left, right = (0.5, 0.24)
    gapx = (S.GUTTER_PT + 6.0) / 72.0
    pw = (W - left - right - 4 * gapx) / 5.0
    ph = pw
    cmap = ListedColormap(S.GREYS6)
    norm = BoundaryNorm(np.arange(-0.5, 6.5, 1.0), cmap.N)
    edges = np.arange(kA + 1) * BIN
    from matplotlib.transforms import offset_copy

    def draw(top_pad, Ht, gapy):
        fig = plt.figure(figsize=(W, Ht))
        axes = {}
        heads = []
        for ri, m in enumerate(ROWS):
            for ci, y in enumerate(S.YEARS):
                x0 = left + ci * (pw + gapx)
                y0 = Ht - top_pad - ph - ri * (ph + gapy)
                ax = fig.add_axes([x0 / W, y0 / Ht, pw / W, ph / Ht])
                axes[ri, ci] = ax
                h = H[y, m][:kA, :kA]
                cls = np.full(h.shape, -1, dtype=int)
                nz = h > 0
                cls[nz] = np.clip(np.floor(np.log10(h[nz])).astype(int), 0, 5)
                Cm = np.ma.masked_less(cls.T.astype(float), 0)
                ax.pcolormesh(edges, edges, Cm, cmap=cmap, norm=norm, rasterized=True, shading='flat', linewidth=0)
                ax.plot([0, A], [0, A], color='white', lw=1.4, solid_capstyle='butt', zorder=3)
                ax.plot([0, A], [0, A], color=S.INK, lw=0.6, linestyle=C.REF_DASH, zorder=4)
                s_f, b_f = FIT[y, m]
                xs = np.array([0.0, float(A)])
                ax.plot(xs, b_f + s_f * xs, color='white', lw=1.6, solid_capstyle='butt', zorder=3, gid='fit:halo')
                ax.plot(xs, b_f + s_f * xs, color=S.MODEL_COLOUR[m], lw=0.8, solid_capstyle='butt', zorder=5, gid='fit')
                ax.set_xlim(0, A)
                ax.set_ylim(0, A)
                ax.set_aspect('equal')
                ax.set_xticks(np.arange(0, A + 1, 10))
                ax.set_yticks(np.arange(0, A + 1, 10))
                C.frame(ax)
                ax.tick_params(labelleft=ci == 0, labelbottom=ri == 1)
                if ci == 0:
                    C.tag(ax, 'ab'[ri])
                trs = offset_copy(ax.transAxes, fig=fig, x=0.0, y=S.TITLE_GAP_PT, units='points')
                ax.text(1.0, 1.0, 'Slope %.2f' % s_f, transform=trs, ha='right', va='bottom', fontsize=S.FS_TICK, color=S.MODEL_TEXT[m], gid='slope')
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        k = 72.0 / fig.dpi
        tg = axes[0, 0].texts[0].get_window_extent(r)
        fr = axes[0, 0].get_window_extent(r)
        yh = (tg.y1 - fr.y1) * k + 2.0 * S.TITLE_GAP_PT
        for ci, y in enumerate(S.YEARS):
            ax = axes[0, ci]
            trh = offset_copy(ax.transAxes, fig=fig, x=0.0, y=yh, units='points')
            heads.append(ax.text(0.5, 1.0, '%d' % y, transform=trh, ha='center', va='bottom', fontsize=8, color=S.INK, gid='colhead'))
        fig.canvas.draw()
        hb = heads[0].get_window_extent(r)
        a0 = axes[0, 0].get_window_extent(r)
        a4 = axes[0, 4].get_window_extent(r)
        inv = fig.transFigure.inverted()
        from matplotlib.lines import Line2D
        ry = hb.y1 + S.TITLE_GAP_PT / k
        rx0, ry0 = inv.transform((a0.x0, ry))
        rx1, _ = inv.transform((a4.x1, ry))
        fig.add_artist(Line2D([rx0, rx1], [ry0, ry0], color=S.FRAME_INK, lw=S.FRAME_LW, gid='colhead:rule'))
        fx, fy = inv.transform((0.5 * (a0.x0 + a4.x1), ry + S.TITLE_GAP_PT / k))
        span = fig.text(fx, fy, 'Held-out year', ha='center', va='bottom', fontsize=8, color=S.INK, gid='colhead:span')
        rowheads = []
        for ri, m in enumerate(ROWS):
            ax = axes[ri, 4]
            trr = offset_copy(ax.transAxes, fig=fig, x=S.AXIS_LABELPAD_PT, y=0.0, units='points')
            rowheads.append(ax.text(1.0, 0.5, S.PLAIN[m], transform=trr, rotation=90, ha='left', va='center', fontsize=8, color=S.INK, gid='rowhead'))
        yl = C.block_title(fig, [axes[0, 0], axes[1, 0]], 'Emulator daily mean NO$_2$ (ppb)', axis='y')
        axes[1, 2].set_xlabel('CMAQ daily mean NO$_2$ (ppb)')
        xl = axes[1, 2].xaxis.label
        block = [axes[1, ci] for ci in range(5)]
        lhand = [Line2D([], [], color=S.INK, lw=0.6, linestyle=C.REF_DASH), Line2D([], [], color=S.MODEL_COLOUR['P'], lw=0.8), Line2D([], [], color=S.MODEL_COLOUR['M'], lw=0.8)]
        llab = ['One-to-one', 'Least-squares fit, reference U-Net', 'Least-squares fit, residual U-Net']
        legs_l = C.house_key(fig, block, lhand, llab, ncol=3, side='below', gap_pt=S.CLEAR_RATIO * S.AXIS_LABELPAD_PT + 1.0)
        fig.canvas.draw()
        _, _, tgt = C._block_box(fig, block, r)
        ly0 = min((lg.get_window_extent(r).y0 for lg in legs_l))
        handles = [Patch(facecolor=c, edgecolor='none') for c in S.GREYS6]
        legs_b = C.house_key(fig, block, handles, CLASS_LABELS, ncol=3, side='below', gap_pt=(tgt.y0 - ly0) * k + S.KEY_GAP_PT, title='Cell-days per bin', handleheight=1.0)
        fig._f1keys = (legs_l, legs_b)
        legs = legs_l + legs_b
        for lg in legs:
            if lg.get_title() is not None:
                lg.get_title().set_color(S.INK)
        fig.canvas.draw()
        kb = min((lg.get_window_extent(r).y0 for lg in legs))
        st = span.get_window_extent(r).y1
        return (fig, axes, legs, xl, yl, heads, span, rowheads, kb / fig.dpi, Ht - st / fig.dpi)
    pad_top, pad_bot = (0.04, 0.04)
    tp0, H0 = (0.8, 5.0)
    gapy = (S.GUTTER_PT + S.TICK_LEN_PT + S.TITLE_GAP_PT + 9.0) / 72.0
    fig, axes, legs, xl, yl, heads, span, rowheads, kb, st = draw(tp0, H0, gapy)
    plt.close(fig)
    top_pad = tp0 - st + pad_top
    Ht = H0 - (tp0 - top_pad) - kb + pad_bot
    fig, axes, legs, xl, yl, heads, span, rowheads, kb, st = draw(top_pad, Ht, gapy)
    legs_l, legs_b = fig._f1keys
    plotted = {'A_ppb': A, 'rule': 'A = smallest multiple of 10 ppb >= upper edge of the bin holding the 99.99th percentile of the pooled CMAQ (row sums) and emulator (column sums) margins of all ten panels', 'k99_bin_index_0based': k99, 'k99_bin_upper_edge_ppb': upper, 'bin_ppb': BIN, 'orientation': 'counts[i][j]: i the CMAQ bin (x), j the emulator bin (y), bins [k*0.5, (k+1)*0.5) ppb for k < A/0.5', 'classes': 'class c = floor(log10(count)) clipped to 0..5 for count >= 1 (c=0: 1-9, ..., c=5: >= 100,000); count 0 white (not drawn)', 'colors': S.GREYS6, 'panels': {}}
    letters = 'abcdefghij'
    for ri, m in enumerate(ROWS):
        for ci, y in enumerate(S.YEARS):
            h = H[y, m][:kA, :kA]
            cls = np.full(h.shape, -1, dtype=int)
            nz = h > 0
            cls[nz] = np.clip(np.floor(np.log10(h[nz])).astype(int), 0, 5)
            plotted['panels']['%s_%d' % (m, y)] = {'tag': '(%s)' % letters[ri * 5 + ci], 'model': m, 'plain': S.PLAIN[m], 'fold': y, 'counts_drawn': h, 'class_drawn': cls, 'cell_days_total': tot[y, m], 'cell_days_beyond_A': beyond[y, m], 'F3_outside_0_100': out100[y, m], 'metrics_n': nrow[y, m][0], 'metrics_cells': nrow[y, m][1], 'row_tag_drawn': '(%s)' % 'ab'[ri], 'fit_slope_binned': round(FIT[y, m][0], 6), 'fit_intercept_binned_ppb': round(FIT[y, m][1], 6), 'fit_slope_printed': '%.2f' % FIT[y, m][0]}
    plotted['fit_rule'] = 'least-squares line of the emulator (y) on CMAQ (x) fitted to the stored counts of all 200 x 200 bins (0-100 ppb): bin centres (k + 0.5) * 0.5 ppb weighted by their counts; drawn from 0 to A ppb; slope printed with two decimals above the panel'
    chk = C.text_check(fig)
    plotted['text_check'] = chk
    plotted['fc_check'] = C.fc_check(fig)
    plotted['layout'] = {'figure_in': [W, Ht], 'gaps_pt': C.gaps_report(fig, [['x label to line key (below)', xl, legs_l[0]], ['line key to key title (below)', legs_l[-1], legs_b[0].get_title()], ['row 1 axes to row 2 tag line', axes[0, 0], axes[1, 0].texts[0]], ['row 2 tick labels to x label', axes[1, 0], xl]])}
    print('gaps', plotted['layout']['gaps_pt'], 'near', chk['text_blocks_nearer_than_4pt'])
    print('fc_check', {k: v for k, v in plotted['fc_check'].items() if v})
    plotted['cmaq_below_1ppb_percent_by_year'] = {'rule': "100 * (CMAQ bins 0 and 1, i.e. below 1.0 ppb, summed over the emulator bins) / the year's cell-days; the reference U-Net's counts (equal margin in both rows, tested)", 'values': share_lt1}
    S.save(fig, STEM, plotted, srcs)
    days = {}
    for y in S.YEARS:
        doy, _ = C.test_days(y)
        days[y] = len(doy)
    cells = sorted(set((v[1] for v in nrow.values())))
    assert len(cells) == 1
    for (y, m), (n, c) in nrow.items():
        assert n == days[y] * c
    bmax_key = max(beyond, key=lambda k: beyond[k])
    share = 100.0 * b_all / n_all
    dlist = '; '.join(('%d: %d' % (y, days[y]) for y in S.YEARS))
    cap = '\\textbf{Modeled daily mean NO$_2$ (ppb) of the two U-Net emulators against CMAQ, contiguous US land cells of the 12 km grid, test days of the held-out years 2015 to 2019.} Columns are held-out years; rows show (a) the reference and (b) the residual U-Net, both at seed 0. Each square is a 0.5 by 0.5 ppb bin shaded by its cell-day count in six classes (key); white bins hold none. Each solid line is a least-squares fit of the emulator on CMAQ to all stored bins, 0 to 100~ppb (bin centers weighted by counts), not to exact cell-day values, with its slope above the panel. Both axes run from 0 to %d ppb; %s of the %s cell-days of the 10 panels (%s percent) have a CMAQ or emulator value at or above %d ppb and are not drawn (at most %s in one panel, %s). Table~\\ref{tab:yearly} gives the metrics.' % (A, C.thousands_tex(b_all), C.thousands_tex(n_all), C.fmt(share, 3), A, C.thousands_tex(beyond[bmax_key]), '%s, %d' % (S.PLAIN[bmax_key[1]].lower().replace('u-net', 'U-Net'), bmax_key[0]))
    moved = {'share_lt1': [C.fmt(min(share_lt1.values()), 1), C.fmt(max(share_lt1.values()), 1)], 'max_beyond': [beyond[bmax_key], bmax_key[1], bmax_key[0]], 'cells': cells[0], 'days': dlist}
    print('moved', moved)
    alt = 'Ten binned scatter plots of emulator against CMAQ daily mean NO2 for two U-Nets and five held-out years, to judge agreement.'
    ticks = ', '.join((str(t) for t in range(0, A + 1, 10)))
    longd = "Figure of ten square panels in two rows and five columns. Column heads over the top row, under one spanning head Held-out year: 2015, 2016, 2017, 2018, 2019. Row heads at the right edge of each row, read upward: Reference U-Net (top row, tagged (a) at its top-left), Residual U-Net (bottom row, tagged (b)). Horizontal axis label, shared: CMAQ daily mean NO2 (ppb), tick labels on the bottom row. Vertical axis label, shared: Emulator daily mean NO2 (ppb), tick labels on the left column. Both axes of every panel have ticks %s. Each panel shows gray square bins of 0.5 ppb whose shade gives the count of cell-days, a dashed black one-to-one line on a thin white band from 0 to %d ppb, and a solid least-squares line fitted to the binned values, blue in the top row and vermilion in the bottom row, also on a thin white band. Above each panel's upper right corner its slope is printed: %s. A line key directly under the axis label names the dashed line One-to-one and the solid lines Least-squares fit, reference U-Net and Least-squares fit, residual U-Net. Under it, a key titled Cell-days per bin has six gray swatches from light to black: %s." % (ticks, A, '; '.join(('%s %d, %s' % (S.PLAIN[m], y, '%.2f' % FIT[y, m][0]) for m in ROWS for y in S.YEARS)), ', '.join(CLASS_LABELS))
    C.write_text(STEM, cap, alt, longd)
    print('A', A, 'upper', upper, 'beyond', b_all, 'of', n_all, 'max', beyond[bmax_key], bmax_key, 'out100', out_all)
    print('text check pass', chk['pass'], 'min pt', chk['min_size_pt'], 'overlaps', chk['overlaps'][:5], 'edge', chk['within_0.08in_of_edge'])
    print('figure in', W, round(Ht, 3))
if __name__ == '__main__':
    main()
