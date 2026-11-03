"""Draws figure fig_A_F6: RMSE by season and urban class, by coast and edge class and by decile of boundary-layer height, for the two U-Nets, the gradient-boosted trees and the weekday climatology."""
import os
import sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_style as S
import build_c2_sealed as SEAL
import build_c2_common as C
import build_fig_common as FC

def akey(k):
    k = str(k)
    if '|' in k:
        return k.replace('|', ', ')
    return 'boundary-layer decile ' + k if k.isdigit() else k
STEM = 'fig_A_F6'
MODELS = ['M', 'P', 'GBT', 'CLIM-WD']
SEEDS = {'M': 0, 'P': 0, 'GBT': None, 'CLIM-WD': None}
SEASONS = ['winter', 'spring', 'summer', 'fall']
URBAN = ['urban', 'intermediate', 'rural']
COAST = ['edge', 'coast', 'border', 'interior']
DECILES = list(range(1, 11))
EXPECTED = {'D4': "largest relative error in rural winter cells; M's rural winter error below P's", 'D5': 'larger at coast and edge than interior for every model', 'D6': 'largest at the lowest PBLH decile'}

def collect(rows):
    out = {'D4': {}, 'D5': {}, 'D6': {}}
    cats = {'D4': [(s, u) for u in URBAN for s in SEASONS], 'D5': COAST, 'D6': DECILES}
    for t in ('D4', 'D5', 'D6'):
        for c in cats[t]:
            key = '%s|%s' % c if t == 'D4' else str(c)
            out[t][key] = {}
            for m in MODELS:
                sel = [r for r in rows if r['table'] == t and r['model'] == m and (r['seed'] == SEEDS[m])]
                if t == 'D4':
                    sel = [r for r in sel if r['season'] == c[0] and r['urban'] == c[1]]
                elif t == 'D5':
                    sel = [r for r in sel if r['class'] == c]
                else:
                    sel = [r for r in sel if int(r['pblh_decile']) == c]
                sel = sorted(sel, key=lambda r: r['fold'])
                folds = [int(r['fold']) for r in sel]
                if folds != S.YEARS:
                    raise SystemExit('folds %s for %s %s %s' % (folds, t, key, m))
                v = np.array([r['RMSE'] for r in sel], dtype=float)
                if not np.all(np.isfinite(v)):
                    raise SystemExit('RMSE missing %s %s %s' % (t, key, m))
                out[t][key][m] = {'folds': folds, 'values': v.tolist(), 'n': [int(r['n']) for r in sel], 'cells': [r.get('cells') for r in sel], 'mean': float(v.mean()), 'min': float(v.min()), 'max': float(v.max())}
    return out

def main():
    src, = SEAL.verify(S.CV, ['d4_to_d7.json'])
    rows = S.load_json(S.CV, 'd4_to_d7.json')['rows']
    data = collect(rows)
    if '--peek' in sys.argv:
        for t in data:
            for k, d in data[t].items():
                print(t, k, ' '.join(('%s %.3f [%.3f %.3f]' % (m, d[m]['mean'], d[m]['min'], d[m]['max']) for m in MODELS)))
        return
    cats_a = ['%s|%s' % (s, u) for u in URBAN for s in SEASONS]
    top_a = max((data['D4'][k][m]['max'] for k in cats_a for m in MODELS)) * 1.04
    vmax_bc = max([data['D5'][k][m]['max'] for k in COAST for m in MODELS] + [data['D6'][str(k)][m]['max'] for k in DECILES for m in MODELS])
    top_bc = vmax_bc * 1.04
    cells = {c: sorted({x for m in MODELS for x in data['D5'][c][m]['cells']}) for c in COAST}
    for c in COAST:
        if len(cells[c]) != 1:
            raise SystemExit('cells differ by fold/model for %s' % c)
    plotted_cells = {c: cells[c] for c in COAST}
    import build_port_fig_A_F6 as PORT
    FC.setup()
    fig, drawn, axs, gname = PORT.draw(data, cells, top_a, top_bc)
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axs
    texts = C.image_texts(fig)
    plotted = {'models': MODELS, 'seeds': SEEDS, 'values': data, 'order_a': cats_a, 'order_b': COAST, 'order_c': DECILES, 'y_top_a': top_a, 'y_top_bc': top_bc, 'expected_metrics_py': EXPECTED, 'source_keys': 'CV d4_to_d7.json rows: table D4 (season, urban), D5 (class), D6 (pblh_decile); model, seed (0 networks, None baselines), fold -> RMSE'}
    plotted['drawn_build1029'] = drawn
    plotted['fc_check'] = FC.fc_check(fig)
    plotted['text_check'] = FC.text_check(fig)
    fcc, tc = (plotted['fc_check'], plotted['text_check'])
    print('fc_check pass', fcc['pass'], 'case_pass', fcc['case_pass'], fcc['case'], fcc['overlaps'], fcc['outside'], fcc['small'])
    print('text_check pass', tc['pass'], tc['overlaps'], tc['within_0.08in_of_edge'], 'fig', tc['figure_in'])
    print('drawn', {k: v for k, v in drawn.items() if k in ('d_xlim', 'ef_xlim_shared')})
    uhi, _ = C.code_const('URBAN_HI')
    ulo, _ = C.code_const('URBAN_LO')
    edge, _ = C.code_const('EDGE_CELLS')
    bord, _ = C.code_const('BORDER_CELLS')
    ckm, _ = C.code_const('CELL_KM')
    consts = {'URBAN_HI': uhi, 'URBAN_LO': ulo, 'EDGE_CELLS': edge, 'BORDER_CELLS': bord, 'CELL_KM': ckm, 'read_from': 'work/metrics.py'}
    print('constants from metrics.py:', consts)
    plotted['caption_facts'] = {'cells_by_class': plotted_cells, 'constants': consts}
    S.save(fig, STEM, plotted, [src])
    yy = data['D4'][cats_a[0]]['M']['folds']
    cap = '\\textbf{Error of modeled daily mean surface NO$_2$ against CMAQ by stratum: RMSE (ppb) on the contiguous US land cells of the 12 km grid, held-out years %d to %d.} Each mark is the mean of the five yearly RMSEs and its pale bar their range (lowest to highest). U-Nets at seed 0. (a) By season and urban class (urban fraction of the cell: urban at least %s, intermediate %s to under %s, rural under %s; winter (JFD) is January, February and December of each held-out year, the December of 2019 held back, spring (MAM) March to May, summer (JJA) June to August and fall (SON) September to November). (b) By coast and edge class (grid edge, coast, border and interior; cells per class: grid edge %s, coast %s, border %s, interior %s). (c) By decile of daily mean boundary-layer height, set within each held-out year. (d) to (f) repeat (a) to (c) on narrower axes for the U-Nets and trees; there the climatology shows its mean alone, and arrows at right give values beyond the axis. The strata are unequal in size and not independent. Panels (d) to (f) were not specified in advance.' % (yy[0], yy[-1], C.num_plain(uhi, 1), C.num_plain(ulo, 1), C.num_plain(uhi, 1), C.num_plain(ulo, 1), '{:,}'.format(cells['edge'][0]), '{:,}'.format(cells['coast'][0]), '{:,}'.format(cells['border'][0]), '{:,}'.format(cells['interior'][0]))
    import re
    cap = re.sub('\\s+', ' ', cap).strip()
    print('caption words', len(re.sub('\\\\[a-zA-Z]+|[{}$]', ' ', cap).split()))
    alt = 'Dot charts of RMSE for four models by season and urban class, coast class and boundary-layer decile, then magnified.'
    cname = PORT.CNAME
    PL = PORT.NAME

    def means(t, k):
        return ', '.join(('%s %s' % (PL[m], C.num_plain(data[t][k][m]['mean'], 2)) for m in MODELS))
    parts = ['Key in one row: ' + '; '.join((PL[m] for m in MODELS)) + '; each a mark on a pale bar.', 'Panel (a) By season and urban class: vertical axis RMSE (ppb), ticks ' + ', '.join((t.get_text() for t in ax_a.get_yticklabels())) + '; horizontal axis Season and urban class: Winter, Spring, Summer, Fall under each of Urban cells, Intermediate cells, Rural cells. Means: ' + '; '.join(('%s %s: %s' % (gname[k.split('|')[1]], k.split('|')[0], means('D4', k)) for k in cats_a)) + '.', 'Panel (b) By coast and edge class: vertical axis RMSE (ppb), ticks ' + ', '.join((t.get_text() for t in ax_b.get_yticklabels())) + '; horizontal axis Coast and edge class: ' + ', '.join(('%s (%s cells)' % (cname[c], '{:,}'.format(cells[c][0])) for c in COAST)) + '. Means: ' + '; '.join(('%s: %s' % (cname[c], means('D5', c)) for c in COAST)) + '.', 'Panel (c) By boundary-layer height decile: on the vertical scale of panel (b), ticked, not labeled again; horizontal axis Boundary-layer height decile (1 shallowest), deciles ' + ', '.join((str(k) for k in DECILES)) + '. Means: ' + '; '.join(('decile %d: %s' % (k, means('D6', str(k))) for k in DECILES)) + '.', 'Panel (d) As (a), magnified: vertical axis Season and urban class, rows ' + ', '.join((t.get_text() for t in ax_d.get_yticklabels())) + '; horizontal axis RMSE (ppb), ticks ' + ', '.join((t.get_text() for t in ax_d.get_xticklabels())) + '; weekday climatology means beyond the axis printed after an arrow: ' + '; '.join(('%s %s' % (akey(k), C.num_plain(v, 2)) for k, v in drawn['d_climatology_beyond_axis'])) + '.', 'Panel (e) As (b), magnified: vertical axis Coast and edge class, rows ' + ', '.join((t.get_text() for t in ax_e.get_yticklabels())) + '. Panel (f) As (c), magnified: vertical axis Boundary-layer height decile, rows ' + ', '.join((t.get_text() for t in ax_f.get_yticklabels())) + '. Each on the same horizontal axis RMSE (ppb), ticks ' + ', '.join((t.get_text() for t in ax_e.get_xticklabels())) + '; weekday climatology means beyond it printed after an arrow: ' + '; '.join(('%s %s' % (akey(k), C.num_plain(v, 2)) for k, v in drawn['ef_climatology_beyond_axis'])) + '.']
    long_desc = '\n'.join(parts).replace('−', '-')
    FC.write_text(STEM, cap, alt, long_desc)
    print('cells by class:', plotted_cells)
    _ = texts
if __name__ == '__main__':
    main()
