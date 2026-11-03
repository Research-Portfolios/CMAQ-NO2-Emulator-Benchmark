"""Draws figure fig_F4: the four comparisons fixed in advance with their intervals, for the test days and the held-back days."""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_style as S
import build_c2_sealed as SEAL
import build_c2_common as C
import build_fig_common as FC
from matplotlib.transforms import offset_copy
STEM = 'fig_F4'
KS = ['C-1', 'C-2', 'C-3', 'C-4']
NAMES = {'C-1': 'C-1: skill score of the residual U-Net against the weekday climatology', 'C-2': 'C-2: RMSE of the gradient-boosted trees minus that of the residual U-Net', 'C-3': 'C-3: RMSE of the reference U-Net minus that of the residual U-Net', 'C-4': 'C-4: mean-map RMSE of the weekday climatology minus that of the residual U-Net'}
XLAB = {'C-1': 'Skill score (unitless)', 'C-2': 'Difference in RMSE (ppb)', 'C-3': 'Difference in RMSE (ppb)', 'C-4': 'Difference in mean-map RMSE (ppb)'}
XLAB_DRAWN = {'C-1': 'Skill score (unitless); higher: residual U-Net closer to CMAQ', 'C-2': 'Difference in RMSE (ppb); positive: residual U-Net closer to CMAQ', 'C-3': 'Difference in RMSE (ppb); positive: residual U-Net closer to CMAQ', 'C-4': 'Difference in mean-map RMSE (ppb); positive: residual U-Net closer to CMAQ'}
TITLE_SHORT = {'C-1': 'C-1: weekday climatology', 'C-2': 'C-2: gradient-boosted trees', 'C-3': 'C-3: reference U-Net', 'C-4': 'C-4: weekday climatology, mean map'}
XLAB_SHORT = {'C-1': 'Skill score', 'C-2': 'RMSE difference (ppb)', 'C-3': 'RMSE difference (ppb)', 'C-4': 'Mean-map RMSE difference (ppb)'}
DIRECTION = 'Residual U-Net closer to CMAQ'
ROWS = ['conf', 'sens', 'fwd', 'lb']
STYLE = {'conf': dict(color=S.MODEL_COLOUR['M'], lw=1.1, marker='o', ms=5.0, mfc=S.MODEL_COLOUR['M'], mec=S.MODEL_COLOUR['M'], mew=0.9), 'sens': dict(color=S.INK2, lw=1.1, marker='o', ms=4.2, mfc='white', mec=S.INK2, mew=0.9), 'fwd': dict(color=S.INK2, lw=1.1, marker='^', ms=4.4, mfc='white', mec=S.INK2, mew=0.9), 'lb': dict(color=S.INK2, lw=1.1, marker='D', ms=3.8, mfc='white', mec=S.INK2, mew=0.9)}
YPOS = {'conf': 3.0, 'sens': 2.0, 'fwd': 1.0, 'lb': 0.0}
LABEL_FWD = 'forward year, exploratory'
LABEL_LB = 'further check'
AX_H = {'C-1': 0.74, 'C-2': 0.74, 'C-3': 0.74, 'C-4': 0.74}

def val3(v):
    s = '%.3f' % float(v)
    return s.replace('-', '−')

def main():
    cvp, = SEAL.verify(S.CV, ['family.json'])
    lbp, = SEAL.verify(S.LB, ['family.json'])
    cv = S.load_json(S.CV, 'family.json')
    lb = S.load_json(S.LB, 'family.json')
    ff = cv['five_fold']
    fw = cv['forward_2019']
    lk = lb['lockbox']
    if ff.get('status') != 'RUN' or 'rows' not in fw or 'rows' not in lk:
        raise SystemExit('STOP: a family part is not RUN')
    vals = {}
    for k in KS:
        a = ff['rows'][k]
        b = a['block_14']
        f = fw['rows'][k]
        g = lk['rows'][k]
        if abs(float(cv['margins'][k]) - float(a['margin'])) > 0:
            raise SystemExit('STOP: margins/%s differs from rows' % k)
        vals[k] = {'margin': float(cv['margins'][k]), 'conf': {'point': a['point'], 'lo': a['lo'], 'hi': a['hi'], 'level': a['level'], 'holm_rank': a['holm_rank'], 'passes_holm_step': a['passes_holm_step'], 'reading': a['reading'], 'key': 'CV family.json five_fold/rows/%s' % k}, 'sens': {'point': a['point'], 'lo': b['lo'], 'hi': b['hi'], 'level': b['level'], 'reading': b['reading'], 'key': 'CV family.json five_fold/rows/%s/block_14 (point: five_fold/rows/%s/point)' % (k, k)}, 'fwd': {'point': f['point'], 'lo': f['lo'], 'hi': f['hi'], 'level': f['level'], 'label': f['label'], 'word': f['word'], 'key': 'CV family.json forward_2019/rows/%s' % k}, 'lb': {'point': g['point'], 'lo': g['lo'], 'hi': g['hi'], 'level': g['level'], 'word': g['word'], 'key': 'LB family.json lockbox/rows/%s' % k}, 'definition': cv['definitions'][k]}
    lv_conf = [C.pct(vals[k]['conf']['level']) for k in KS]
    lv_sens = [C.pct(vals[k]['sens']['level']) for k in KS]
    lv_fwd = sorted({C.pct(vals[k]['fwd']['level']) for k in KS})
    lv_lb = sorted({C.pct(vals[k]['lb']['level']) for k in KS})
    if len(lv_fwd) != 1 or len(lv_lb) != 1:
        raise SystemExit('STOP: the 2019 or held-back levels differ by contrast; the key must list them')
    blk = sorted({d['block_used'] for d in ff['draws']})
    blk14 = sorted({d['block_used'] for d in ff['draws_14']})
    nres = sorted({int(vals_n) for vals_n in [ff['rows'][k]['n_resamples'] for k in KS]})
    nres_f = sorted({int(fw['rows'][k]['n_resamples']) for k in KS})
    nres_l = sorted({int(lk['rows'][k]['n_resamples']) for k in KS})
    blk_f = sorted({d['block_used'] for d in fw['draws']})
    blk_l = sorted({d['block_used'] for d in lk['draws']})
    lb_days = [int(d['n_days']) for d in lk['draws']]
    lb_years = [int(d['year']) for d in lk['draws']]
    cv_days = [int(d['n_days']) for d in ff['draws']]
    if not len(blk) == len(blk14) == len(nres) == len(nres_f) == len(nres_l) == len(blk_f) == len(blk_l) == 1:
        raise SystemExit('STOP: mixed resampling settings')
    m = int(cv['m'])
    lb_ff_equal = lb.get('five_fold', {}).get('rows') == ff['rows']
    for k in KS:
        if not str(vals[k]['fwd']['label']).startswith('exploratory'):
            raise SystemExit('STOP: the stored 2019 label of %s is not exploratory; the key must follow it' % k)
    if lv_fwd[0] in lv_conf[:-1] or len(set(lv_conf)) == 1:
        raise SystemExit('STOP: the unadjusted level cannot be told from the Holm levels; reword the caption')
    FC.setup()
    W = 6.5
    EDGE = 0.09
    AXH = AX_H['C-1']
    WB = 0.62
    GAPB = 0.08
    YLIM = (-0.6, 3.6)
    fig = S.plt.figure(figsize=(W, 7.0))
    lo_a = min((vals['C-1'][r]['lo'] for r in ROWS))
    hi_a = max((vals['C-1'][r]['hi'] for r in ROWS))
    if lo_a <= 0.0:
        raise SystemExit('an interval of (a) reaches 0: draw 0 on (a)')
    t0 = round(0.05 * int(lo_a / 0.05 + 1e-09), 2)
    t1 = round(0.05 * (int(hi_a / 0.05 - 1e-09) + 1), 2)
    lim_a = (t0 - 0.004, t1 + 0.004)
    hi_b = max((vals[k][r]['hi'] for k in KS[1:] for r in ROWS))
    lo_b = min((vals[k][r]['lo'] for k in KS[1:] for r in ROWS))
    if lo_b < 0:
        raise SystemExit('an interval below 0 in (b) to (d): widen the axis rule before drawing')
    top_b = 0.05 * (int(hi_b / 0.05) + 1)
    lim_b = (-0.01, top_b + 0.005)
    xt_a = [round(t0 + 0.05 * j, 2) for j in range(int(round((t1 - t0) / 0.05)) + 1)]
    xt_b = [round(0.05 * j, 2) for j in range(int(round(top_b / 0.05)) + 1)]
    plotted = {'panels': {}, 'x_limits': {}, 'rows_top_to_bottom': ROWS}
    mg_b = sorted({vals[k]['margin'] for k in KS[1:]})
    if len(mg_b) != 1:
        raise SystemExit('STOP: margins differ in (b) to (d)')
    margin_label = 'Margin, %s ppb' % C.num_plain(mg_b[0], 2)
    if vals['C-1']['margin'] != 0.0:
        raise SystemExit("STOP: the margin of C-1 is not 0; the caption and (a)'s dashed line at 0 assume it")
    row_name = {'conf': 'Confirmatory', 'sens': '%d-day blocks' % blk14[0], 'fwd': 'Forward year', 'lb': 'Held-back days'}
    drawn_levels = {}
    P = {}
    for i, k in enumerate(KS):
        b0 = 7.0 - (i + 1) * 1.6
        if k == 'C-1':
            ax0 = fig.add_axes([0.2, b0 / 7.0, 0.1, AXH / 7.0])
            ax = fig.add_axes([0.32, b0 / 7.0, 0.5, AXH / 7.0])
            ax0.set_xlim(-0.01, 0.05)
            ax0.set_xticks([0.0])
            ax0.set_xticklabels(['0'])
            ax0.set_ylim(*YLIM)
            ax0.axvline(0.0, color=S.REF_INK, lw=0.6, ls=(0, (4, 2.5)), zorder=1)
            FC.frame(ax0)
            lab_ax = ax0
            ax.set_yticks([])
        else:
            ax0 = None
            ax = fig.add_axes([0.2, b0 / 7.0, 0.62, AXH / 7.0])
            lab_ax = ax
        lim = lim_a if k == 'C-1' else lim_b
        ax.set_xlim(*lim)
        ax.set_ylim(*YLIM)
        ax.set_xticks(xt_a if k == 'C-1' else xt_b)
        ax.set_xticklabels([C.tick(t, 2) for t in (xt_a if k == 'C-1' else xt_b)])
        if k != 'C-1':
            ax.axvline(0.0, color=S.REF_INK, lw=0.6, zorder=1)
        mg = vals[k]['margin']
        if mg != 0.0:
            ax.axvline(mg, color=S.REF_INK, lw=0.6, ls=(0, (4, 2.5)), zorder=1)
        for r in ROWS:
            v = vals[k][r]
            st = STYLE[r]
            y = YPOS[r]
            ax.plot([v['lo'], v['hi']], [y, y], color=st['color'], lw=st['lw'], solid_capstyle='butt', zorder=2)
            ax.plot([v['point']], [y], ls='none', marker=st['marker'], ms=st['ms'], mfc=st['mfc'], mec=st['mec'], mew=st['mew'], zorder=3)
        lab_ax.set_yticks([YPOS[r] for r in ROWS])
        lab_ax.set_yticklabels([row_name[r] for r in ROWS])
        for t, r in zip(lab_ax.get_yticklabels(), ROWS):
            t.set_color(S.MODEL_TEXT['M'] if r == 'conf' else S.INK)
        FC.house_ticks(lab_ax, 'y')
        lab_ax.set_ylabel('Analysis')
        rax = ax.twinx()
        rax.set_ylim(*YLIM)
        rax.set_yticks([YPOS[r] for r in ROWS])
        lv = [C.pct(vals[k][r]['level']) for r in ROWS]
        drawn_levels[k] = lv
        rax.set_yticklabels(['%s%%' % s for s in lv])
        for t, r in zip(rax.get_yticklabels(), ROWS):
            t.set_color(S.MODEL_TEXT['M'] if r == 'conf' else S.INK)
        FC.house_ticks(rax, 'y')
        rax.set_ylabel('Interval level')
        FC.frame(ax)
        FC.frame(rax)
        ax.set_xlabel(XLAB_SHORT[k])
        FC.tag(lab_ax, 'abcd'[i], TITLE_SHORT[k])
        plotted['panels'][k] = vals[k]
        plotted['x_limits'][k] = list(lim)
        P[k] = {'ax0': ax0, 'ax': ax, 'lab': lab_ax, 'rax': rax}
    fig.canvas.draw()
    rr = fig.canvas.get_renderer()
    hy = P['C-1']['rax'].yaxis.label.get_window_extent(rr).height
    right_title = 'Interval level'
    if hy > AXH * fig.dpi - 2.0:
        right_title = 'Level'
        for k in KS:
            P[k]['rax'].set_ylabel(right_title)
        fig.canvas.draw()
        rr = fig.canvas.get_renderer()
    from matplotlib.transforms import Bbox
    ext = {}
    for k in KS:
        p = P[k]
        parts = [a for a in (p['ax0'], p['ax'], p['rax']) if a is not None]
        fr = Bbox.union([a.get_window_extent(rr) for a in parts])
        tb = Bbox.union([a.get_tightbbox(rr) for a in parts])
        ext[k] = {'above': (tb.y1 - fr.y1) / fig.dpi, 'below': (fr.y0 - tb.y0) / fig.dpi, 'left': (fr.x0 - tb.x0) / fig.dpi, 'right': (tb.x1 - fr.x1) / fig.dpi}
    L = EDGE + max((e['left'] for e in ext.values()))
    R = W - EDGE - max((e['right'] for e in ext.values()))
    tt = fig.text(0, 0, 'Hg0', fontsize=S.KEY_FS)
    key_h = tt.get_window_extent(rr).height / fig.dpi
    tt.remove()
    top_res = EDGE + key_h + S.KEY_GAP_PT / 72.0
    gap = S.BLOCK_GAP_PT / 72.0
    tpos = {}
    t = top_res
    for k in KS:
        ftop = t + ext[k]['above']
        tpos[k] = ftop + AXH
        t = tpos[k] + ext[k]['below'] + gap
    H = t - gap + EDGE
    fig.set_size_inches(W, H)
    for k in KS:
        p = P[k]
        yb = (H - tpos[k]) / H
        if p['ax0'] is not None:
            p['ax0'].set_position([L / W, yb, WB / W, AXH / H])
            pos = [(L + WB + GAPB) / W, yb, (R - L - WB - GAPB) / W, AXH / H]
        else:
            pos = [L / W, yb, (R - L) / W, AXH / H]
        p['ax'].set_position(pos)
        p['rax'].set_position(pos)
    ax0, ax = (P['C-1']['ax0'], P['C-1']['ax'])
    for a, xs in ((ax0, 1.0), (ax, 0.0)):
        wi = a.get_position().width * W
        for yb in (0.0, 1.0):
            a.plot([xs - 0.03 / wi, xs + 0.03 / wi], [yb - 0.06, yb + 0.06], transform=a.transAxes, color=S.FRAME_INK, lw=0.6, clip_on=False, zorder=5)
    ax.xaxis.label.set_x(((L + R) / 2.0 - (L + WB + GAPB)) / (R - L - WB - GAPB))
    for k in KS:
        P[k]['lab'].set_ylabel('')
        P[k]['rax'].set_ylabel('')
    yl_left = FC.block_title(fig, [P[k]['lab'] for k in KS], 'Analysis', axis='y', side='left')
    yl_right = FC.block_title(fig, [P[k]['rax'] for k in KS], right_title, axis='y', side='right')
    note_drawn = {}
    from matplotlib.lines import Line2D
    h_zero = Line2D([], [], color=S.REF_INK, lw=0.6)
    h_marg = Line2D([], [], color=S.REF_INK, lw=0.6, ls=(0, (4, 2.5)))
    KEY_DRAWN = ['Zero', 'Margin']
    legs = FC.house_key(fig, [ax0, ax], [h_zero, h_marg], KEY_DRAWN, side='above')
    for i, k in enumerate(KS):
        S.register_block(fig, 'panel (%s)' % 'abcd'[i], [P[k]['ax'], P[k]['rax']])
    lab = {'conf': 'Five years, %d-day blocks' % blk[0], 'sens': 'Five years, %d-day blocks' % blk14[0], 'fwd': '2019 alone', 'lb': 'Held-back days'}
    l2 = [margin_label]
    texts = C.image_texts(fig)
    plotted['key_labels'] = [lab[r] for r in ROWS] + l2
    plotted['key_drawn'] = KEY_DRAWN
    plotted['row_names_drawn'] = [row_name[r] for r in ROWS]
    plotted['interval_levels_drawn'] = drawn_levels
    plotted['axis_titles_drawn'] = [XLAB_SHORT[k] for k in KS]
    plotted['panel_titles_drawn'] = [TITLE_SHORT[k] for k in KS]
    plotted['y_titles_drawn'] = ['Analysis', right_title]
    plotted['direction_note_drawn'] = note_drawn
    plotted['y_titles_once_for_stack'] = True
    plotted['caption_facts'] = {'holm_levels_conf': lv_conf, 'levels_14': lv_sens, 'level_2019': lv_fwd[0], 'level_lockbox': lv_lb[0], 'block_days': blk[0], 'block_days_sensitivity': blk14[0], 'n_resamples_family': nres[0], 'n_resamples_2019': nres_f[0], 'n_resamples_lockbox': nres_l[0], 'block_2019': blk_f[0], 'block_lockbox': blk_l[0], 'm': m, 'cv_test_days_by_year': dict(zip([d['year'] for d in ff['draws']], cv_days)), 'lockbox_days_by_year': dict(zip(lb_years, lb_days)), 'covers': cv['covers'], 'lb_five_fold_rows_equal_cv': lb_ff_equal}
    plotted['fc_check'] = FC.fc_check(fig)
    plotted['text_check'] = FC.text_check(fig)
    plotted['figure_in'] = [W, round(H, 3)]
    print('figure in: %.2f x %.2f' % (W, H))
    print('fc_check pass', plotted['fc_check']['pass'], 'case_pass', plotted['fc_check']['case_pass'], plotted['fc_check']['case'], plotted['fc_check']['overlaps'], plotted['fc_check']['outside'])
    print('text_check pass', plotted['text_check']['pass'], plotted['text_check']['overlaps'], plotted['text_check']['within_0.08in_of_edge'])
    S.save(fig, STEM, plotted, [cvp, lbp])
    cf = plotted['caption_facts']
    if cf['level_2019'] != cf['level_lockbox'] or cf['n_resamples_2019'] != cf['n_resamples_lockbox']:
        raise SystemExit('the caption joins the 2019 and held-back settings; they differ: reword')
    if cf['block_2019'] != cf['block_days'] or cf['block_lockbox'] != cf['block_days']:
        raise SystemExit('the caption gives one block length for the family, 2019 and the held-back days; reword')
    yy = sorted((int(d['year']) for d in ff['draws']))
    cap = "\\textbf{Four comparisons, fixed in advance, of the residual U-Net with other models of daily mean surface NO$_2$, scored against CMAQ on the contiguous US land cells of the 12 km grid, held-out years %d to %d.} Higher values favor the residual U-Net. Panel (a) gives C-1, its skill score (unitless) against the weekday climatology. Panels (b) to (d) give the RMSE (ppb) of the model in the title minus that of the residual U-Net: C-2 for the gradient-boosted trees, C-3 for the reference U-Net and C-4, the mean-map RMSE, for the weekday climatology. Each interval's level is printed at right. Filled vermilion circles show the mean of the five yearly values with its percentile interval at the level of its Holm step among %s comparisons, from paired resamples of %d-day blocks (confirmatory). Open circles show the same with %d-day blocks (sensitivity), open triangles 2019 alone (%s) and open diamonds the held-back days, evaluated once (%s), the last two with unadjusted 95 percent intervals. In (b) to (d) the solid line marks zero difference; the dashed line marks the margin, 0 in (a), drawn left of the axis break, and %s ppb in (b) to (d). C-4's resampled intervals lay high against their points (Section~\\ref{sec:four}); reflected about the point, the held-back interval is 0.086 to 0.187~ppb, still above the margin. Intervals cover the sampling of test days; Figure~\\ref{fig:F5} shows the spread over seeds." % (yy[0], yy[-1], 'four' if m == 4 else str(m), cf['block_days'], cf['block_days_sensitivity'], LABEL_FWD, LABEL_LB, C.num_plain(mg_b[0], 2))
    import re
    cap = re.sub('\\s+', ' ', cap).strip()
    print('caption words', len(re.sub('\\\\[a-zA-Z]+|[{}$]', ' ', cap).split()))
    alt = 'Four stacked panels of the four comparisons fixed in advance, each with four rows of a point and its interval.'
    rows_txt = {'conf': 'filled vermilion circle', 'sens': 'open circle', 'fwd': 'open triangle', 'lb': 'open diamond'}
    parts = ["A key line above panel (a) names a solid line Zero and a dashed line Margin. In every panel the rows, from the top, are named at the left and each row's interval level is printed at the right; one vertical title, Analysis, at the left and one, %s, at the right are centred on the four stacked panels." % right_title]
    for i, k in enumerate(KS):
        ticks = xt_a if k == 'C-1' else xt_b
        p = ['Panel (%s), titled %s, shows %s. Horizontal axis title: %s; ticks %s.' % ('abcd'[i], TITLE_SHORT[k], NAMES[k], XLAB_SHORT[k], ', '.join((C.tick(t, 2) for t in ticks)))]
        if k == 'C-1':
            p.append('A short broken part of the axis at its left holds the dashed margin line at 0, tick 0.')
        else:
            p.append('A solid vertical line at 0 and a dashed vertical line at the margin, %s (%s).' % (C.num_plain(vals[k]['margin'], 2), margin_label))
        p.append('Rows: ' + '; '.join(('%s, %s, %s%%: %s (%s to %s)' % (row_name[r], rows_txt[r], C.pct(vals[k][r]['level']), C.num_plain(vals[k][r]['point']), C.num_plain(vals[k][r]['lo']), C.num_plain(vals[k][r]['hi'])) for r in ROWS)) + '.')
        parts.append(' '.join(p))
    long_desc = '\n'.join(parts)
    C.check_long_holds(long_desc, texts)
    FC.write_text(STEM, cap, alt, long_desc)
if __name__ == '__main__':
    main()
