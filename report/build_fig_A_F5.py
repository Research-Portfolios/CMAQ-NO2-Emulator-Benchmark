"""Draws figure fig_A_F5: the median autocorrelation of the daily error by lag (1 to 14 days), for the year and each season, for the two U-Nets, the gradient-boosted trees and the weekday climatology."""
import os
import sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_style as S
import build_c2_sealed as SEAL
import build_c2_common as C
import build_check_text2 as CHK
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
STEM = 'fig_A_F5'
MODELS = ['M', 'P', 'GBT', 'CLIM-WD']
SEEDS = {'M': 0, 'P': 0, 'GBT': None, 'CLIM-WD': None}
import build_fig_common as FC
import build_port_fig_A_F5 as PF
LS = {'M': FC.DASH['M'], 'P': '-', 'GBT': (0, (5, 1.5, 1, 1.5)), 'CLIM-WD': (0, (1, 1.3))}
BANDED = ('M', 'P')
LS_WORD = {'M': 'dashed', 'P': 'solid', 'GBT': 'dash-dot', 'CLIM-WD': 'dotted'}
LEVEL_INK, LEVEL_LW = ('#9A9A9A', 0.5)
Y_LABEL = 'Median error autocorrelation'
COLOR_WORD = {'M': 'vermilion', 'P': 'blue', 'GBT': 'green', 'CLIM-WD': 'gray'}
SEASONS = ['year', 'winter', 'spring', 'summer', 'fall']
SEASON_NAME = {'year': 'Whole year', 'winter': 'Winter (Jan, Feb, Dec)', 'spring': 'Spring (Mar to May)', 'summer': 'Summer (Jun to Aug)', 'fall': 'Fall (Sep to Nov)'}
LAGS = list(range(1, 15))
THRESH = None

def main():
    global THRESH
    THRESH, _ = C.code_const('ACF_BELOW')
    src, = SEAL.verify(S.CV, ['residual_acf.json'])
    rows = S.load_json(S.CV, 'residual_acf.json')['rows']
    data = {}
    first = {}
    missing = []
    for m in MODELS:
        data[m] = {}
        first[m] = {}
        for s in SEASONS:
            sel = [r for r in rows if r['model'] == m and r['seed'] == SEEDS[m] and (r['season'] == s)]
            folds = sorted({int(r['fold']) for r in sel})
            mat = np.full((len(folds), len(LAGS)), np.nan)
            for r in sel:
                i = folds.index(int(r['fold']))
                j = LAGS.index(int(r['lag']))
                if not np.isnan(mat[i, j]):
                    raise SystemExit('duplicate row %s %s' % (m, s))
                if r['median'] is None:
                    missing.append([m, s, r['fold'], r['lag']])
                else:
                    mat[i, j] = r['median']
            fb = {}
            for r in sel:
                fb[int(r['fold'])] = r['first_below_0_2']
            first[m][s] = fb
            data[m][s] = {'folds': folds, 'by_fold': mat.tolist(), 'mean': np.nanmean(mat, axis=0).tolist(), 'min': np.nanmin(mat, axis=0).tolist(), 'max': np.nanmax(mat, axis=0).tolist(), 'folds_present_per_lag': np.sum(np.isfinite(mat), axis=0).tolist()}
    if missing:
        print('medians missing (left out of the mean):', len(missing))
    allv = [x for m in MODELS for s in SEASONS for x in data[m][s]['min'] + data[m][s]['max'] if x is not None and np.isfinite(x)]
    ylo = min(0.0, 0.1 * np.floor(min(allv) / 0.1))
    yhi = 0.1 * np.ceil(max(allv) / 0.1)
    print('medians drawn: %.3f to %.3f; axis %.1f to %.1f' % (min(allv), max(allv), ylo, yhi))
    FC.setup()
    plotted = {'models': MODELS, 'seeds': SEEDS, 'seasons': SEASONS, 'lags': LAGS, 'values': data, 'first_below_0_2_by_fold': first, 'medians_missing': missing, 'y_limits': [ylo, yhi], 'source_keys': 'CV residual_acf.json rows: model, seed (0 for the networks, None for the baselines), season, fold, lag -> median; first_below_0_2'}
    fig, H, art = PF.draw(MODELS, SEASONS, LAGS, data, THRESH, ylo, yhi, plotted)
    form = plotted['layout_season_form']['chosen']
    if form == 'code':
        SEASON_CAP = '(JFD, MAM, JJA and SON as in Figure~\\ref{fig:F7}), '
    else:
        SEASON_CAP = "(winter: the held-out year's January, February and December; the December of 2019 was held back), "
    SL = {s: PF.season_label(s, form) for s in SEASONS}

    def A(t):
        return t.replace('–', ' to ')
    chk = FC.text_check(fig)
    plotted['text_check'] = chk
    plotted['fc_check'] = FC.fc_check(fig)
    plotted['layout'] = {'figure_in': [6.5, H]}
    S.save(fig, STEM, plotted, [src])
    folds = sorted({f for m in MODELS for s in SEASONS for f in data[m][s]['folds']})
    nfold = sorted({len(data[m][s]['folds']) for m in MODELS for s in SEASONS})
    if nfold != [5] or folds != S.YEARS:
        raise SystemExit('not five folds everywhere: reword the caption')
    cap = "\\textbf{Day-to-day persistence of the error of modeled daily mean surface NO$_2$ against CMAQ: the median over the contiguous US land cells of the 12 km grid of each cell's error autocorrelation, by lag, held-out years 2015 to 2019 (unitless).} The error is the model minus CMAQ on the test days; at lag $L$ a cell's autocorrelation is the mean product of its centered errors on pairs of test days $L$ calendar days apart over their variance, and the median is taken over cells. Each mark or line is the mean of the five held-out years' medians; each range drawn runs from the lowest to the highest of the five years. (a) Lag 1 by season, with one mark per model (key) on a pale bar spanning its range. (b) The whole held-out year and (c) to (f) its seasons " + SEASON_CAP + "lags 1 to 14, with the range as a vermilion band for the residual U-Net and two thin blue lines for the reference U-Net. (g) to (j) Each model's whole-year line with its four seasons as thin gray lines, from light (winter) to dark (fall), each with its own mark (key). Each model keeps one color, line style and mark in every panel (key); marks at lags 1 to 3. The U-Nets are at seed 0; the two baselines have one fit per held-out year. The black dashed line marks the level of %s set in advance; the lag at which a line falls below it marks how long a day's error carries into the following days." % C.num_plain(THRESH, 1)
    cap = cap.replace('2015 to 2019', '%d to %d' % (folds[0], folds[-1]))
    print('caption words', len(cap.split()))
    alt = 'Median error autocorrelation by lag for four models: lag 1 by season, then by season and by model.'
    yt = [t.get_text() for t in art['b'][0].get_yticklabels() if t.get_text()]
    parts = ['Ten panels in three rows, with one key at the top. Key, models: Residual U-Net (vermilion solid line, disc), Reference U-Net (blue dashed line, square), Gradient-boosted trees (dark gray dash-dot line, triangle), Weekday climatology (light gray dotted line, cross); then the seasons ' + ', '.join((A(SL[q]) for q in SEASONS[1:])) + ' (thin gray lines, light to dark, marked in turn by a thin diamond, a plus, a star and a pentagon), Range over the five years (pale bars and band), and Level %s (black dashed line).' % C.num_plain(THRESH, 1), 'Row 1, panel (a) Lag 1 by season: y axis Season, rows ' + ', '.join((A(SL[q]) for q in SEASONS)) + '; for each, four marks, one per model, on pale bars; x axis Median error autocorrelation at lag 1, ticks 0.0, 0.1, 0.2, 0.3; a black dashed line at 0.2.', 'Row 2, panels ' + ', '.join(('(%s) %s' % ('bcdef'[i], A(SL[q])) for i, q in enumerate(SEASONS))) + ": the four models' lines over lags 1 to 14, the residual U-Net's range as a vermilion band and the reference U-Net's as two thin blue lines.", "Row 3, panels (g) Residual U-Net, (h) Reference U-Net, (i) Gradient-boosted trees, (j) Weekday climatology: the model's whole-year line and four thin gray season lines, each with its own mark.", 'Shared axis titles of rows 2 and 3, each drawn once: Lag (days), under row 3, ticks 1, 7, 14 on both rows; Median error autocorrelation, ticks ' + ', '.join(yt) + '.']
    for k, s in enumerate(SEASONS):
        parts.append('Whole-year and season values, panel (%s): mean median at lag 1, 7 and 14: ' % 'bcdef'[k] + '; '.join(('%s %s, %s, %s' % (S.PLAIN[m], C.num_plain(data[m][s]['mean'][0], 2), C.num_plain(data[m][s]['mean'][6], 2), C.num_plain(data[m][s]['mean'][13], 2)) for m in MODELS)) + '.')
    long_desc = '\n'.join(parts).replace('−', '-')
    FC.write_text(STEM, cap, alt, long_desc)
    print(ascii(['text check', chk['pass'], chk['overlaps'][:4], chk['within_0.08in_of_edge'], chk['text_blocks_nearer_than_4pt'][:6]]))
    print(ascii({k: v for k, v in plotted['fc_check'].items() if v}))
    print('figure 6.5 x', round(H, 3))
if __name__ == '__main__':
    main()
