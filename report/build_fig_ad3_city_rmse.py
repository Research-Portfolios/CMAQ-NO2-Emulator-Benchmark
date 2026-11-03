"""Draws figure fig_ad3_city_rmse: the RMSE of the two U-Nets in each of the ten cities on the 2016 test days, beside the residual U-Net's RMSE over all land cells, each value checked against the table."""
import json
import os
import re
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_style as S
import build_fig_common as C
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
STEM = 'fig_ad3_city_rmse'
NUMBERS = os.path.join(S.ROOT, 'results', 'report', 'numbers.json')
TAB = os.path.join(S.RUN, 'tex', 'tab_A_T1.tex')
JOIN = '#BDBDBD'

def printed_rmse():
    out = {}
    first, rmse_at = (None, [])
    with open(TAB, encoding='ascii') as f:
        for ln in f:
            if 'multicolumn' in ln and first is None:
                first = 'P' if ln.find('Reference') < ln.find('Residual') else 'M'
            parts = [p.strip() for p in ln.replace('\\\\', '').split('&')]
            if parts and parts[0] == 'City':
                rmse_at = [i for i, p in enumerate(parts) if p == 'RMSE']
            if len(parts) == 12 and parts[0] in S.CITY_NAME.values():
                assert first and len(rmse_at) == 2, (first, rmse_at)
                ip, im = rmse_at if first == 'P' else rmse_at[::-1]
                out[parts[0]] = (parts[ip], parts[im], parts[1])
    return out

class _Mk:

    def __init__(self, m, ms):
        self.m, self.ms = (m, ms)

class _MkHandler(HandlerBase):

    def create_artists(self, legend, h, xd, yd, w, ht, fs, trans):
        y = (ht - yd) / 2.0
        return [Line2D([-xd, -xd + w], [y, y], color='none', lw=0, transform=trans), Line2D([-xd + w - h.ms / 2.0 - 0.5], [y], ls='none', marker=S.MODEL_MARKER[h.m], ms=h.ms, mfc=S.MODEL_COLOUR[h.m], mec='white', mew=0.5, transform=trans)]

def main():
    src = C.sealed('city_series.json')
    rows = C.rows('city_series.json')
    with open(NUMBERS, encoding='utf-8') as f:
        nb = json.load(f)
    allcell = nb['y.m.rmse.2016']
    pr = printed_rmse()
    cities = [S.CITY_NAME[k] for k in S.CITY_ORDER]
    val = {}
    for c in cities:
        r = {}
        for m in ('P', 'M'):
            x = [x for x in rows if x['model'] == m and x['city'] == c and (x['fold'] == 2016) and (x['seed'] == 0)]
            assert len(x) == 1, (m, c, len(x))
            r[m] = x[0]
        assert r['P']['cells'] == r['M']['cells']
        assert '%.3f' % r['P']['RMSE'] == pr[c][0], (c, pr[c])
        assert '%.3f' % r['M']['RMSE'] == pr[c][1], (c, pr[c])
        assert str(r['M']['cells']) == pr[c][2]
        val[c] = {'P_rmse': r['P']['RMSE'], 'M_rmse': r['M']['RMSE'], 'cells': r['M']['cells'], 'n_days': r['M']['n']}
    assert abs(float(allcell['text']) - allcell['value']) < 0.0005
    C.setup()
    W = 6.5
    YTITLE = 'City'
    mf = S.plt.figure(figsize=(W, 2.0))
    rr = mf.canvas.get_renderer()
    wlab = max((mf.text(0, 0, s, fontsize=S.FS_TICK).get_window_extent(rr).width for s in cities)) / mf.dpi
    wtit = mf.text(0, 0, YTITLE, fontsize=S.FS_LABEL, rotation=90).get_window_extent(rr).width / mf.dpi
    S.plt.close(mf)
    left = round(0.1 + wtit + (S.AXIS_LABELPAD_PT + S.TICK_PAD_PT + S.TICK_LEN_PT) / 72.0 + wlab, 3)
    right = 0.3
    ph = 3.05 - 0.34 - 0.5
    n = len(cities)
    ys = list(range(n - 1, -1, -1))

    def draw(top, H):
        fig = S.plt.figure(figsize=(W, H))
        ax = fig.add_axes([left / W, (H - top - ph) / H, (W - left - right) / W, ph / H])
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_xlim(0.4, 2.4)
        ax.axvline(allcell['value'], color=S.MODEL_COLOUR['M'], lw=0.7, ls=(0, (1.0, 1.6)), zorder=1)
        for c, y in zip(cities, ys):
            v = val[c]
            ax.plot([v['P_rmse'], v['M_rmse']], [y, y], color=JOIN, lw=1.4, solid_capstyle='butt', zorder=2)
            ax.plot([v['P_rmse']], [y], ls='none', marker=S.MODEL_MARKER['P'], ms=6.0, mfc=S.MODEL_COLOUR['P'], mec='white', mew=0.5, zorder=3)
            ax.plot([v['M_rmse']], [y], ls='none', marker=S.MODEL_MARKER['M'], ms=4.6, mfc=S.MODEL_COLOUR['M'], mec='white', mew=0.5, zorder=4)
        ax.set_yticks(ys)
        ax.set_yticklabels(cities)
        ax.set_ylabel(YTITLE, fontsize=S.FS_LABEL)
        C.house_ticks(ax, 'y')
        ax.set_xticks([0.5, 1.0, 1.5, 2.0])
        ax.set_xticklabels(['0.5', '1.0', '1.5', '2.0'])
        ax.set_xlabel('RMSE against CMAQ (ppb)', fontsize=S.FS_LABEL)
        C.frame(ax)
        hd = [_Mk('M', 4.6), _Mk('P', 6.0), Line2D([0], [0], color=S.MODEL_COLOUR['M'], lw=0.7, ls=(0, (1.0, 1.6)))]
        labs = ['Residual U-Net', 'Reference U-Net', 'Residual U-Net, all land cells']
        legs = C.house_key(fig, [ax], hd, labs, side='above', handler_map={_Mk: _MkHandler()})
        for lg_ in legs:
            for t_ in lg_.get_texts():
                for m_ in ('M', 'P'):
                    if t_.get_text().startswith(S.PLAIN[m_]):
                        t_.set_color(S.MODEL_TEXT[m_])
        return (fig, ax, legs)
    MARG = 0.1
    top0, H0 = (0.8, 4.5)
    fig, ax, legs = draw(top0, H0)
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    S.plt.close(fig)
    top = top0 - (H0 - tb.y1) + MARG
    H = round(tb.y1 - tb.y0 + 2 * MARG, 3)
    fig, ax, legs = draw(top, H)
    labs = [t_.get_text() for lg_ in legs for t_ in lg_.get_texts()]
    plotted = {'label': 'DESC, added after the plan was fixed', 'fold': 2016, 'seed': 0, 'rows_top_to_bottom': [dict(city=c, y=y, **val[c]) for c, y in zip(cities, ys)], 'allcell_rmse_M_2016': {'key': 'y.m.rmse.2016', 'value': allcell['value'], 'text': allcell['text']}, 'x_limits': [0.4, 2.4], 'x_ticks': [0.5, 1.0, 1.5, 2.0], 'key': labs, 'join_colour': JOIN, 'figure_in': [W, H], 'printed_check': 'every RMSE equal to tex\\tab_A_T1.tex at 3 decimals (asserted)'}
    plotted['fc_check'] = C.fc_check(fig)
    plotted['text_check'] = C.text_check(fig)
    fcc, tc = (plotted['fc_check'], plotted['text_check'])
    print('fc_check pass', fcc['pass'], 'case_pass', fcc['case_pass'], fcc['case'], fcc['overlaps'], fcc['outside'], fcc['small'], fcc['open_frames'])
    print('text_check pass', tc['pass'], tc['overlaps'], tc['within_0.08in_of_edge'], 'fig', tc['figure_in'])
    S.save(fig, STEM, plotted, [src, NUMBERS])
    lo = min((val[c]['M_rmse'] for c in cities))
    hi = max((val[c]['M_rmse'] for c in cities))
    cap = "\\textbf{RMSE (ppb) of modeled daily mean surface NO$_2$ against CMAQ in 10 large US cities, reference and residual U-Nets, test days of 2016.} Each row is one city, in the order of Table~\\ref{tab:cities}, which also gives its land cells; each value is the RMSE of the city's daily mean over the land cells of the 3~$\\times$~3 window of 12 km cells centered on the cell that holds its point. Discs show the residual U-Net and squares the reference U-Net, and a gray line joins a city's two values; where they nearly coincide, the disc lies on the square. The dotted vertical line marks the residual U-Net's RMSE over all land cells and test days of 2016 (%s ppb). Both U-Nets are the networks trained with seed 0 without 2016. Mean errors by city are shown in Figure~\\ref{fig:F3k}. These results are descriptive and were not specified in advance." % allcell['text']
    alt = 'Dumbbell chart of the 2016 RMSE of the two U-Nets in ten US cities.'
    longd = "Rows: ten cities. Horizontal axis: RMSE against CMAQ (ppb), 0.5 to 2.0. Blue squares: reference U-Net; vermilion discs: residual U-Net; a gray line joins each pair. A dotted vermilion vertical line marks the residual U-Net's RMSE over all land cells, %s ppb. Values, reference then residual: %s. The land cells per city are in the appendix table of the ten cities." % (allcell['text'], '; '.join(('%s %.3f and %.3f' % (c, val[c]['P_rmse'], val[c]['M_rmse']) for c in cities)))
    C.write_text(STEM, cap, alt, longd)
    print('M range', round(lo, 3), round(hi, 3), 'words', len(cap.split()))
if __name__ == '__main__':
    main()
