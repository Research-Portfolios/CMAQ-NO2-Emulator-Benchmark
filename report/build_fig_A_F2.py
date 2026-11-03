"""Draws figure fig_A_F2: the daily land-mean NO2 of each held-out year for CMAQ, the residual U-Net and the reference U-Net, checking the series against the evaluation's day counts and mean errors."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import build_fig_common as C
import build_style as S
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import build_port_fig_A_F2 as PA
STEM = 'fig_A_F2'
CMAQ_BASE = '#B3B3B3'
CMAQ_LW = 1.4

def main():
    C.setup()
    srcs = []
    arr = {}
    for y in S.YEARS:
        for m in ('M', 'P'):
            p = C.sealed('F5_domain_fold%d_%s.npy' % (y, m))
            srcs.append(p)
            arr[y, m] = np.load(p)
    mets = C.rows('metrics.json')
    srcs.append(C.sealed('metrics.json'))
    mbrow = {(r['fold'], r['model']): r['MB'] for r in mets if r['stratum_kind'] == 'all' and r['model'] in ('M', 'P') and (r['seed'] == 0)}
    days = {}
    for y in S.YEARS:
        doy, dates = C.test_days(y)
        days[y] = (doy, dates)
        a, b = (arr[y, 'M'], arr[y, 'P'])
        assert a.shape == b.shape == (2, len(doy)), y
        assert np.array_equal(a[0], b[0]), y
        for m in ('M', 'P'):
            d = float(np.mean(arr[y, m][1] - arr[y, m][0]))
            assert abs(d - mbrow[y, m]) < 1e-09, (y, m, d, mbrow[y, m])
    vmax = max((float(a.max()) for a in arr.values()))
    vmin = min((float(a.min()) for a in arr.values()))
    D = int(np.ceil(vmax - 1e-12))
    import datetime as DT
    d19 = set((int(v) for v in days[2019][0]))
    miss19 = [d for d in range(1, 366) if d not in d19]
    dec19 = [(DT.date(2019, 12, 1) - DT.date(2019, 1, 1)).days + 1 + i for i in range(31)]
    dec_only = miss19 == dec19
    letters = 'abcde'
    plotted = {'D_ppb': D, 'D_rule': 'smallest whole ppb >= the largest value drawn (all five years, CMAQ and both U-Nets)', 'largest_value_drawn': vmax, 'smallest_value_drawn': vmin, 'row_order': 'row 0 CMAQ, row 1 the model (run_eval.py)', 'panels': {}}
    for k, y in enumerate(S.YEARS):
        doy, dates = days[y]
        a, b = (arr[y, 'M'], arr[y, 'P'])
        plotted['panels'][str(y)] = {'tag': '(%s)' % letters[k], 'day_of_year': doy, 'dates': dates, 'cmaq': a[0], 'residual_unet': a[1], 'reference_unet': b[1], 'metrics_MB_M': mbrow[y, 'M'], 'metrics_MB_P': mbrow[y, 'P']}
    plotted['2019_withheld_day_of_year'] = miss19
    plotted['2019_withheld_is_december'] = dec_only
    fig, Ht, art = PA.draw(arr, days, mbrow, D, plotted)
    chk = C.text_check(fig)
    plotted['text_check'] = chk
    plotted['fc_check'] = C.fc_check(fig)
    month = [t for t in fig.texts if t.get_text().startswith('Month of the')][0]
    plotted['layout'] = {'figure_in': [6.5, Ht], 'gaps_pt': C.gaps_report(fig, [['key to first tag line', art['legend'], art['lev'][0].texts[0]], ['row 1 strip to row 2 tag line', art['stp'][0], art['lev'][1].texts[0]], ['month title to (f) tag line', month, art['ovs'][0].texts[-1]]])}
    S.save(fig, STEM, plotted, srcs)
    nd = ', '.join(('%d in %d' % (len(days[y][0]), y) for y in S.YEARS))
    wd = ', '.join(('%d' % ((366 if y % 4 == 0 else 365) - len(days[y][0])) for y in S.YEARS))
    cap = "\\textbf{Modeled daily mean NO$_2$ (ppb) averaged over the contiguous US land cells of the 12 km grid, CMAQ and the two U-Net emulators, on the test days of each held-out year from 2015 to 2019, with each U-Net's daily difference from CMAQ.} Panels (a) 2015 to (e) 2019 show one held-out year each. The upper axes (scale at left) show the mean over the %s land cells of each test day, CMAQ as a wide light gray line, the residual U-Net as a vermilion solid line and the reference U-Net as a blue dashed line, each at seed 0, trained without that year. The strip below (scale at right) shows each U-Net minus CMAQ, and Table~\\ref{tab:yearly} gives each year's mean error. The test days drawn are %s; the held-back and buffer days (%s days in 2015 to 2019) are left as gaps%s. The level axes run from 0 to %d ppb and the strips from $-$0.5 to 0.5 ppb; ticks mark month starts. (f) The residual and (g) the reference U-Net's daily differences of the five years overlaid by date, earlier years in lighter shades of the U-Net's color, each year named at the right. The black line marks zero difference." % (C.thousands_tex(int(round([r for r in mets if r['stratum_kind'] == 'all' and r['model'] == 'M' and (r['seed'] == 0) and (r['fold'] == 2015)][0]['cells']))), nd, wd, ', and in 2019 they are all of December, shaded gray' if dec_only else '', D)
    print('caption words', len(cap.split()))
    alt = 'Daily land mean NO2 of CMAQ and two U-Nets, five held-out years, and their differences; Table 2 gives the mean errors.'
    longd = "Figure of five year rows and two overlay panels. Key at the top: CMAQ (wide light gray line), Residual U-Net (vermilion solid line), Reference U-Net (blue dashed line), Zero difference (black line). Rows (a) 2015 held out to (e) 2019 held out: each an upper axes with the three daily series, vertical ticks %s at the left, and a strip below with each U-Net minus CMAQ about a black zero line, vertical ticks minus 0.4, 0.0, 0.4 at the right; December 2019 shaded gray. One title at the far left, NO2 (ppb), and one at the far right, U-Net minus CMAQ (ppb), each centered on the five rows; each year's mean error is in Table 2. Month names under row (e); label Month of the held-out year. Panels (f) Residual U-Net minus CMAQ and (g) Reference U-Net minus CMAQ: the five years' daily differences overlaid, one shade per year from light (2015) to dark (2019), each year named at the right, a black zero line; label Month of each held-out year." % ', '.join((str(t) for t in range(0, D + 1, 2)))
    C.write_text(STEM, cap, alt, longd)
    print('D', D, 'vmax', vmax, 'vmin', vmin, 'withheld', wd)
    print(ascii(['text check pass', chk['pass'], 'min', chk['min_size_pt'], 'over', chk['overlaps'][:4], 'edge', chk['within_0.08in_of_edge'], 'near', chk['text_blocks_nearer_than_4pt'][:6]]))
    print(ascii({k: v for k, v in plotted['fc_check'].items() if v}))
    print('gaps', plotted['layout']['gaps_pt'], 'fig 6.5', round(Ht, 3))
if __name__ == '__main__':
    main()
