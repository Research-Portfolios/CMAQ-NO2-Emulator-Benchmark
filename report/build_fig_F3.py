"""Draws figure fig_F3: daily mean NO2 in the ten most populous cities in 2016, CMAQ and the residual U-Net, with withheld days as gaps; the same function draws fig_A_F1."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import build_fig_common as C
import build_style as S
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import build_port_fig_F3 as P3
YEAR = 2016
CMAQ_BASE = S.GREYS6[0]
CMAQ_LW = 1.6
EMU_LW = 0.6

def alnum(s):
    return ''.join((ch for ch in s if ch.isalnum()))

def tick_words(v):
    v = float(v)
    s = '%d' % abs(v) if v == int(v) else '%g' % abs(v)
    return 'minus ' + s if v < 0 else s

def load_all():
    arr, srcs = ({}, [])
    for m in ('M', 'P'):
        for city in S.CITY_ORDER:
            n = 'F5_city_fold%d_%s_%s.npy' % (YEAR, m, city)
            p = C.sealed(n)
            srcs.append(p)
            a = np.load(p)
            assert a.shape == (2, 339), (n, a.shape)
            arr[m, city] = a
    for city in S.CITY_ORDER:
        assert np.array_equal(arr['M', city][0], arr['P', city][0]), city
    cr = C.rows('city_series.json')
    srcs.append(C.sealed('city_series.json'))
    rows = {}
    for r in cr:
        if r.get('fold') == YEAR and r.get('seed') == 0:
            rows[r['model'], alnum(r['city'])] = r
    for (m, city), a in arr.items():
        r = rows[m, city]
        mb = float(np.mean(a[1] - a[0]))
        assert abs(mb - r['MB']) < 1e-09, (m, city, mb, r['MB'])
        assert r['n'] == a.shape[1]
    return (arr, srcs, rows)

def build(model, stem):
    C.setup()
    arr, srcs, rows = load_all()
    doy, dates = C.test_days(YEAR)
    assert len(doy) == 339
    vmax = max((float(a.max()) for a in arr.values()))
    vmin = min((float(a.min()) for a in arr.values()))
    B = int(np.ceil(vmax / 5.0 - 1e-12) * 5)
    W = 6.5
    letters = 'abcdefghij'
    plotted = {'model': model, 'plain': S.PLAIN[model], 'year': YEAR, 'B_ppb': B, 'B_rule': 'smallest multiple of 5 ppb >= the largest value drawn in F3 and A-F1 together (CMAQ, the residual U-Net, the reference U-Net, ten cities)', 'largest_value_drawn_both': vmax, 'smallest_value_drawn_both': vmin, 'row_order': 'row 0 CMAQ, row 1 the model (run_eval.py)', 'x': 'day of year 2016 (1 = 1 Jan) of the 339 TEST_CV days', 'test_day_of_year': doy, 'test_dates': dates, 'panels': {}}
    for k, city in enumerate(S.CITY_ORDER):
        a = arr[model, city]
        r = rows[model, city]
        plotted['panels'][city] = {'tag': '(%s)' % letters[k], 'name': S.CITY_NAME[city], 'cmaq': a[0], 'model': a[1], 'cells': r['cells']}
    if model == 'M':
        fig, Ht, art = P3.draw_f3(arr, rows, doy, B, plotted, YEAR)
    else:
        fig, Ht, art = P3.draw_a3(arr, doy, B, plotted, YEAR)
    chk = C.text_check(fig)
    plotted['text_check'] = chk
    plotted['fc_check'] = C.fc_check(fig)
    month = [t for t in fig.texts if t.get_text().startswith('Month')][0]
    pairs = [['key to first tag line', art['legend'], art['first_tag']], ['row 1 axes to row 2 tag line', art['row1'] if model == 'P' else art['strip1'], art['row2_tag']]]
    if model == 'P':
        pairs.append(['month title to scatter tag line', month, art['scatter_tag']])
    plotted['layout'] = {'figure_in': [W, Ht], 'gaps_pt': C.gaps_report(fig, pairs)}
    S.save(fig, stem, plotted, srcs)
    idx = doy - 1
    br = int(np.sum(np.diff(idx) > 1))
    n_days = len(doy)
    year_days = 366
    cells = sorted(set((int(rows[model, c]['cells']) for c in S.CITY_ORDER)))
    cmin, cmax = (cells[0], cells[-1])
    gaps = ('no', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine')[br]
    names = ', '.join(('(%s) %s' % (letters[k], S.CITY_NAME[c]) for k, c in enumerate(S.CITY_ORDER)))
    block = "the daily mean over the land cells of the 3~$\\times$~3 window of 12 km cells centered on the cell that holds the city's point (%d to %d cells per city)" % (cmin, cmax)
    if model == 'M':
        cap = "\\textbf{Modeled daily mean NO$_2$ (ppb) of CMAQ and the residual U-Net in 10 large US cities, and each U-Net minus CMAQ, on the %d test days of 2016.} Each series is %s; both U-Nets are the networks trained with seed 0 without 2016. In each of (a) to (j), the upper plot shows CMAQ (wide light gray line) and the residual U-Net (vermilion line) on the scale at left. The strip below shows each U-Net minus CMAQ on the scale at right, the reference U-Net as a blue dashed line and the residual U-Net as a vermilion solid line. The black line marks zero difference. Gaps mark the %d held-back and buffer days, and ticks mark month starts. Appendix Table~\\ref{tab:cities} gives each city's $R$, IOA and errors. The difference strips were not specified in advance." % (n_days, block, year_days - n_days)
        alt = 'Ten city panels of 2016 daily NO2, CMAQ and the residual U-Net, with strips of each U-Net minus CMAQ.'
        longd = 'Figure of ten city panel pairs in five rows and two columns. Key above, centered: CMAQ (wide light gray line), Reference U-Net (blue dashed line), Residual U-Net (vermilion solid line), Zero difference (black line). City panels: %s. Each has an upper plot with CMAQ and the residual U-Net, vertical ticks 0, 20, 40 at the left, and a strip below with each U-Net minus CMAQ about a black zero line, vertical ticks minus 10, 0, 10 at the right. Gaps in each series mark the %d held-back and buffer days; ticks mark month starts. One title at the far left, City daily mean NO2 (ppb), and one at the far right, U-Net minus CMAQ (ppb), each centered on the ten panel pairs. Month names under the bottom strips; axis label Month of %d.' % (names, year_days - n_days, YEAR)
    else:
        cap = "\\textbf{Modeled daily mean NO$_2$ (ppb) of CMAQ and the reference U-Net in 10 large US cities, and each U-Net against CMAQ, on the %d test days of 2016.} Each series is %s, as in Figure~\\ref{fig:F3}. Panels (a) to (j) show CMAQ (wide light gray line) and the reference U-Net (blue dashed line), the seed-0 network with 2016 held out; the other %d days, the held-back and buffer days, are left as %s gaps, and ticks mark month starts. All level axes run from 0 to %d ppb, as in Figure~\\ref{fig:F3}. Panels (k) to (t) set each test day's U-Net value against CMAQ's, in the same city order, the reference U-Net as blue squares and the residual U-Net as vermilion discs drawn over them, both translucent, so overlapping marks darken to a gray-brown; CMAQ is the horizontal axis, and the black line is the one-to-one line. The residual U-Net's levels and both U-Nets' daily differences are in Figure~\\ref{fig:F3}." % (n_days, block, year_days - n_days, gaps, B)
        alt = 'Ten panels of 2016 city daily NO2, CMAQ and the reference U-Net, then ten scatters of each U-Net against CMAQ.'
        snames = ', '.join(('(%s) %s' % ('klmnopqrst'[k], S.CITY_NAME[c]) for k, c in enumerate(S.CITY_ORDER)))
        longd = 'Figure of ten line-chart panels in five rows and two columns, then ten square scatter panels in two rows of five. Key above, centered: CMAQ (wide light gray line), Reference U-Net (blue dashed line, square), Residual U-Net (vermilion disc), One-to-one (black line). Line panels: %s; each shows CMAQ and the reference U-Net over %d with gaps at the held-back and buffer days; vertical ticks 0, 20, 40; shared label City daily mean NO2 (ppb); month names under the bottom row; axis label Month of %d. Scatter panels: %s; horizontal axis CMAQ city daily mean NO2 (ppb), vertical axis U-Net city daily mean NO2 (ppb), both 0 to %d with ticks 0, 20, 40; small translucent blue squares and vermilion discs, one per test day, and a black one-to-one line.' % (names, YEAR, YEAR, snames, B)
    print('caption words', len(cap.split()))
    C.write_text(stem, cap, alt, longd)
    print(stem, 'B', B, 'vmax', vmax, 'gaps', br, 'days', n_days, 'cells', cells)
    print(ascii(['text check pass', chk['pass'], 'min', chk['min_size_pt'], 'over', chk['overlaps'][:4], 'edge', chk['within_0.08in_of_edge'], 'near', chk['text_blocks_nearer_than_4pt'][:6]]))
    print(ascii({k: v for k, v in plotted['fc_check'].items() if v}))
    print('gaps', plotted['layout']['gaps_pt'], 'fig', W, round(Ht, 3))

def build_k(stem='fig_F3k'):
    C.setup()
    arr, srcs, rows = load_all()
    doy, dates = C.test_days(YEAR)
    assert len(doy) == 339
    plotted = {'models': ['P', 'M'], 'plain': {m: S.PLAIN[m] for m in ('P', 'M')}, 'year': YEAR, 'row_order': 'row 0 CMAQ, row 1 the model (run_eval.py)', 'test_day_of_year': doy, 'test_dates': dates}
    fig, Ht, art = P3.draw_f3k(arr, rows, plotted, YEAR)
    chk = C.text_check(fig)
    plotted['text_check'] = chk
    plotted['fc_check'] = C.fc_check(fig)
    pairs = [['key to title line', art['legend'], art['title']]]
    plotted['layout'] = {'figure_in': [P3.W, Ht], 'gaps_pt': C.gaps_report(fig, pairs)}
    S.save(fig, stem, plotted, srcs)
    n_days = len(doy)
    cap = "\\textbf{Mean error (ME) of each U-Net against CMAQ in modeled daily mean NO$_2$ (ppb), 10 large US cities, %d test days of 2016.} The city series are those of Figure~\\ref{fig:F3}, and both U-Nets are the networks trained with seed 0 without 2016. In each city row, a square marks the reference U-Net's ME, the mean over the test days of its daily differences from CMAQ (U-Net minus CMAQ), and a disc marks the residual U-Net's; a gray line joins the two. Each mark sits on a pale bar of its color that spans the 10th to the 90th percentile of those daily differences, with a tick at each end, and the black line marks zero difference. Rows run from the city with the highest CMAQ mean NO$_2$ (top) to the lowest; Appendix Table~\\ref{tab:cities} gives each U-Net's mean error by city. These results are descriptive and were not specified in advance." % n_days
    alt = "Ten city rows of each U-Net's 2016 mean error against CMAQ, on pale 10th to 90th percentile bars of the daily differences."
    longd = "Figure of one panel. Key above, centered: Reference U-Net (pale blue bar, square), Residual U-Net (pale vermilion bar, disc), Zero difference (black line). Title: Mean error (U-Net minus CMAQ), %d. One row per city, highest CMAQ mean first; rows, top to bottom: %s; each row has a blue square and a vermilion disc at the two U-Nets' mean errors over the %d test days of %d, joined by a gray line, each on a pale bar from the 10th to the 90th percentile with end ticks, the square's bar above the disc's; a black zero line; vertical axis title City, a tick at each city; horizontal axis U-Net minus CMAQ (ppb), ticks %s." % (YEAR, ', '.join((S.CITY_NAME[c] for c in plotted['k_row_order']['values'])), n_days, YEAR, ' to '.join((tick_words(plotted['k_x_range']['xticks'][i]) for i in (0, -1))))
    print('caption words', len(cap.split()))
    C.write_text(stem, cap, alt, longd)
    print(ascii(['text check pass', chk['pass'], 'min', chk['min_size_pt'], 'over', chk['overlaps'][:4], 'edge', chk['within_0.08in_of_edge'], 'near', chk['text_blocks_nearer_than_4pt'][:6]]))
    print(ascii({k: v for k, v in plotted['fc_check'].items() if v}))
    print('gaps', plotted['layout']['gaps_pt'], 'fig', P3.W, round(Ht, 3))
if __name__ == '__main__':
    build('M', 'fig_F3')
