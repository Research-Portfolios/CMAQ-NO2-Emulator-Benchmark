"""Draws figure fig_A_F4: each network's spatial power over CMAQ's by wavelength band, from the evaluation's band ratios."""
import os
import sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_style as S
import build_c2_sealed as SEAL
import build_c2_common as C
import build_check_text2 as CHK
STEM = 'fig_A_F4'
BANDS = ['24-48', '48-96', '96-192', '192-768']
BAND_NAME = {'24-48': '24 to 48 km', '48-96': '48 to 96 km', '96-192': '96 to 192 km', '192-768': '192 to 768 km'}
BAND_LABEL = {b: BAND_NAME[b].replace(' to ', '–') for b in BANDS}
MODELS = ['P', 'A1', 'A2', 'A3', 'M', 'PIX-R']
ROW_LABEL = {'P': S.PLAIN['P'], 'A1': 'Step 1: loss', 'A2': 'Step 2: network', 'A3': 'Step 3: target and climatology', 'M': S.PLAIN['M'], 'PIX-R': S.PLAIN['PIX-R']}
STEP_INK = S.INK
COLOUR = {'P': S.MODEL_COLOUR['P'], 'M': S.MODEL_COLOUR['M'], 'PIX-R': S.MODEL_COLOUR['PIX-R'], 'A1': STEP_INK, 'A2': STEP_INK, 'A3': STEP_INK}
MARKER = {'P': S.MODEL_MARKER['P'], 'M': S.MODEL_MARKER['M'], 'PIX-R': S.MODEL_MARKER['PIX-R'], 'A1': 'h', 'A2': 'h', 'A3': 'h'}
OPEN = ('A1', 'A2', 'A3')
X_LABEL = 'Spatial power, network over CMAQ'
SEED = 0

def main():
    src, = SEAL.verify(S.CV, ['band_ratios.json'])
    d = S.load_json(S.CV, 'band_ratios.json')
    rows = d['rows']
    expected = sorted({r['expected'] for r in rows})
    data = {}
    windows = set()
    for m in MODELS:
        data[m] = {}
        for b in BANDS:
            sel = [r for r in rows if r['model'] == m and r['seed'] == SEED and (r['band_km'] == b) and (r['ratio_model_over_cmaq'] is not None)]
            if not sel:
                raise SystemExit('no row for %s %s' % (m, b))
            folds = sorted((int(r['fold']) for r in sel))
            if len(set(folds)) != len(folds):
                raise SystemExit('duplicate fold rows for %s %s' % (m, b))
            v = np.array([r['ratio_model_over_cmaq'] for r in sorted(sel, key=lambda r: r['fold'])])
            windows |= {int(r['windows']) for r in sel}
            data[m][b] = {'folds': folds, 'values': v.tolist(), 'mean': float(v.mean()), 'min': float(v.min()), 'max': float(v.max()), 'names': [r['name'] for r in sel]}
    folds_by_model = {m: data[m][BANDS[0]]['folds'] for m in MODELS}
    for m in MODELS:
        for b in BANDS:
            if data[m][b]['folds'] != folds_by_model[m]:
                raise SystemExit('folds differ by band for %s' % m)
    allv = [x for m in MODELS for b in BANDS for x in data[m][b]['values']]
    lo = min(min(allv), 1.0)
    hi = max(max(allv), 1.0)
    xlo = 0.1 * np.floor(lo / 0.1 - 1e-09)
    xhi = 0.1 * np.ceil(hi / 0.1 + 1e-09)
    print('ratio range drawn: %.3f to %.3f; axis %.1f to %.1f' % (min(allv), max(allv), xlo, xhi))
    import build_fig_common as FC
    import build_port_fig_A_F4 as PORT
    span = xhi - xlo
    step = 0.2 if span <= 0.8 else 0.4 if span <= 1.6 else 0.5
    ticks = [round(t, 2) for t in np.arange(0.0, xhi + 1e-09, step) if t >= xlo - 1e-09]
    FC.setup()
    lim = (xlo - 0.02 * span, xhi + 0.02 * span)
    fig, drawn, axes = PORT.draw(data, MODELS, ROW_LABEL, BANDS, BAND_LABEL, COLOUR, MARKER, OPEN, lim)
    texts = C.image_texts(fig)
    plotted = {'rows_top_to_bottom': MODELS, 'row_labels': ROW_LABEL, 'bands': BANDS, 'seed': SEED, 'values': data, 'x_limits': [xlo, xhi], 'x_ticks': ticks, 'windows': sorted(windows), 'expected': expected, 'folds_by_model': folds_by_model, 'source_keys': 'CV band_ratios.json rows: model, seed 0, band_km, fold -> ratio_model_over_cmaq'}
    plotted['drawn_build1029'] = drawn
    plotted['fc_check'] = FC.fc_check(fig)
    plotted['text_check'] = FC.text_check(fig)
    fcc, tc = (plotted['fc_check'], plotted['text_check'])
    print('fc_check pass', fcc['pass'], 'case_pass', fcc['case_pass'], fcc['case'], fcc['overlaps'], fcc['outside'], fcc['small'])
    print('text_check pass', tc['pass'], tc['overlaps'], tc['within_0.08in_of_edge'], 'fig', tc['figure_in'])
    S.save(fig, STEM, plotted, [src])
    nwin = sorted(windows)
    win, _ = C.code_const('WINDOW')
    ckm, _ = C.code_const('CELL_KM')
    if len(nwin) != 1:
        raise SystemExit('window counts differ: reword the caption')
    five = [m for m in MODELS if len(folds_by_model[m]) == 5]
    two = [m for m in MODELS if folds_by_model[m] != folds_by_model[five[0]]]
    two_years = sorted({tuple(folds_by_model[m]) for m in two})
    if len(two_years) != 1 or set(five) != {'P', 'M', 'PIX-R'}:
        raise SystemExit('fold sets not as the caption says: reword')
    yrs2 = ' and '.join((str(y) for y in two_years[0]))
    import json
    with open(os.path.join(S.ROOT, 'results', 'report', 'numbers.json'), encoding='utf-8') as f:
        nb = json.load(f)
    len_m = nb['rec.steps.m']['text'].replace(',', '{,}')
    len_p = nb['rec.steps.p']['text'].replace(',', '{,}')
    word = {2: 'two', 4: 'four', 5: 'five'}
    n2, n5 = (len(two_years[0]), len(folds_by_model['M']))
    cap = "\\textbf{Spatial power of six networks' modeled daily mean surface NO$_2$ fields over that of CMAQ, by wavelength band, in %s %d km windows inside the contiguous US land cells of the 12 km grid, held-out years 2015 to 2019 (unitless).} (a) Profile across the four bands, on a log ratio axis, of the reference U-Net, the residual U-Net and the per-cell network (the residual U-Net's inputs and target with no spatial context), the bands spaced by their order, fine to coarse. Panels (b) to (e) show one band each, every network on its own row, on the same log ratio axis as (a). The power of a band is the sum over its wavenumbers of the radially averaged power spectrum of each test day's field in %d by %d-cell windows wholly inside the land cells, each less its mean and Hann-tapered, averaged over windows and test days of the held-out year. Each marker (an open hexagon for a step) is the mean over held-out years, joined across the bands in (a), and its pale bar the range (lowest to highest); all networks used seed 0. In (b) to (e), the steps' rows are means over the %s held-out years %s (steps 1 to 3 were trained only with those years held out), while the other rows are means over the %s held-out years 2015 to 2019; each range spans the same years. Steps 1 to 3 (rows ``Step 1: loss'', ``Step 2: network'' and ``Step 3: target and climatology'') lie between the two U-Nets and are defined in Appendix G. The residual U-Net was trained for %s steps and the other five networks for %s. The black line marks equal power (ratio 1)." % (word[nwin[0]], int(round(win * ckm)), win, win, word[n2], yrs2, word[n5], len_m, len_p)
    yy = folds_by_model['M']
    import re
    cap = re.sub('\\s+', ' ', cap.replace('2015 to 2019', '%d to %d' % (yy[0], yy[-1]))).strip()
    print('caption words', len(re.sub('\\\\[a-zA-Z]+|[{}$]', ' ', cap).split()))
    alt = "Six networks' spatial power over CMAQ's: a profile across four wavelength bands, then one panel per band."
    prof = drawn['a_key_order']
    parts = ['Panel (a) Profile across the wavelength bands: horizontal axis Wavelength band (km), bands fine to coarse and spaced by their order, ' + ', '.join((BAND_NAME[b] for b in BANDS)) + '; vertical axis Power ratio, network over CMAQ, log scale, ticks ' + ', '.join((t.get_text() for t in axes[0].get_yticklabels())) + '; a black line at 1, named Equal power in the key. Lines with marks for ' + ', '.join((ROW_LABEL[m] for m in prof)) + ' (key in one line above (a), in that order), each mark on a pale bar over the range of years.', 'Panels (b) to (e), one per band, share six rows labeled at the left under the vertical axis title Network, from the top: ' + '; '.join((ROW_LABEL[m] for m in MODELS)) + '; shared horizontal axis title Power ratio, network over CMAQ, log scale as in (a), ticks ' + ', '.join((t.get_text() for t in axes[1].get_xticklabels())) + '; a black vertical line at 1 in each. Steps 1 to 3 are open hexagons.']
    for j, b in enumerate(BANDS):
        parts.append('Panel (%s) %s: ' % ('bcde'[j], BAND_NAME[b]) + '; '.join(('%s mean %s (range %s to %s)' % (ROW_LABEL[m], C.num_plain(data[m][b]['mean'], 2), C.num_plain(data[m][b]['min'], 2), C.num_plain(data[m][b]['max'], 2)) for m in MODELS)) + '.')
    long_desc = '\n'.join(parts)
    FC.write_text(STEM, cap, alt, long_desc)
    _ = texts
if __name__ == '__main__':
    main()
