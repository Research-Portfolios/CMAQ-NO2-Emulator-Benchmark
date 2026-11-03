"""Draws figure fig_A_F7: the share of test cell-days with an emission input outside the cell's training range, by held-out year and urban class, with the held-back days apart."""
import json
import os
import re
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_style as S
import build_c2_sealed as SEAL
import build_c2_common as C
import build_fig_common as FC
STEM = 'fig_A_F7'
PARTS = [('TEST_CV', 'CV'), ('LOCKBOX (apart)', 'LB')]
STRATA = ['all', 'urban', 'intermediate', 'rural']
SNAME = {'all': 'All land cells', 'urban': 'Urban cells', 'intermediate': 'Intermediate cells', 'rural': 'Rural cells'}
NOT_USED = {'P18'}

def sig(r):
    return json.dumps([r['share'], r['n_cell_days'], r['n_outside'], r['by_urban'], r.get('channels')], sort_keys=True)

def main():
    cvp, = SEAL.verify(S.CV, ['an02.json'])
    lbp, = SEAL.verify(S.LB, ['an02.json'])
    rows = {'TEST_CV': [r for r in S.load_json(S.CV, 'an02.json')['rows'] if r['part'] == 'TEST_CV'], 'LOCKBOX (apart)': [r for r in S.load_json(S.LB, 'an02.json')['rows'] if r['part'] == 'LOCKBOX (apart)']}
    data, nets, c7, p18 = ({}, {}, {}, {})
    for part, _ in PARTS:
        data[part], nets[part], c7[part], p18[part] = ({}, {}, {}, {})
        for y in S.YEARS:
            main_rows = [r for r in rows[part] if r['fold'] == y and r['source'] == 'main']
            used = [r for r in main_rows if r['model'] not in NOT_USED]
            if not used:
                raise SystemExit('no main row %s %d' % (part, y))
            sigs = {sig(r) for r in main_rows}
            if len(sigs) != 1:
                raise SystemExit('networks differ in %s %d: draw per network instead' % (part, y))
            r0 = used[0]
            d = {'all': {'share': r0['share'], 'n_cell_days': r0['n_cell_days'], 'n_outside': r0['n_outside']}}
            for u in STRATA[1:]:
                d[u] = dict(r0['by_urban'][u])
            for u in STRATA:
                rec = d[u]['n_outside'] / d[u]['n_cell_days']
                if abs(rec - d[u]['share']) > 1e-12:
                    raise SystemExit('share not n_outside / n_cell_days')
            data[part][str(y)] = d
            nets[part][str(y)] = sorted(('%s s%s' % (r['model'], r['seed']) for r in used))
            p18[part][str(y)] = [r['share'] for r in main_rows if r['model'] in NOT_USED]
            c7[part][str(y)] = [r['share'] for r in rows[part] if r['fold'] == y and r['source'] == 'c7']
    for part, _ in PARTS:
        for y, v in c7[part].items():
            if any((x != 0 for x in v)):
                raise SystemExit('M-C7 share not 0 in %s %s' % (part, y))
    allv = [data[p][str(y)][u]['share'] for p, _ in PARTS for y in S.YEARS for u in STRATA]
    vmax = max(allv)
    print('shares drawn: %.4f to %.4f' % (min(allv), vmax))
    top = vmax * 1.06
    names = {'TEST_CV': 'Test days of each held-out year', 'LOCKBOX (apart)': 'Held-back days, evaluated once'}
    import build_port_fig_A_F7 as PORT
    FC.setup()
    fig, drawn, nv6 = PORT.draw(data, [p for p, _ in PARTS], names, top, S.YEARS)
    texts = C.image_texts(fig)
    plotted = {'parts': [p for p, _ in PARTS], 'strata': STRATA, 'values': data, 'networks_sharing_values': nets, 'M-C7_shares': c7, 'P18_not_used_shares': p18, 'y_top': top, 'source_keys': 'CV an02.json rows part TEST_CV and LB an02.json rows part LOCKBOX (apart): fold, source main -> share, n_cell_days, n_outside, by_urban/<class>; source c7 (M-C7) -> share'}
    plotted['drawn_build1029'] = drawn
    plotted['tile_percent_2sf'] = nv6
    plotted['rule'] = '100 x the stored share (values/<part>/<year>/<stratum>/share) to two significant figures, as each tile prints it'
    plotted['fc_check'] = FC.fc_check(fig)
    plotted['text_check'] = FC.text_check(fig)
    fcc, tc = (plotted['fc_check'], plotted['text_check'])
    print('fc_check pass', fcc['pass'], 'case_pass', fcc['case_pass'], fcc['case'], fcc['overlaps'], fcc['outside'], fcc['small'])
    print('text_check pass', tc['pass'], tc['overlaps'], tc['within_0.08in_of_edge'], 'fig', tc['figure_in'])
    S.save(fig, STEM, plotted, [cvp, lbp])
    uhi, _ = C.code_const('URBAN_HI')
    ulo, _ = C.code_const('URBAN_LO')
    nch = sorted({len(r['channels']) for r in rows['TEST_CV']})
    if len(nch) != 1:
        raise SystemExit('channel counts differ: reword')
    words = {5: 'five', 4: 'four', 6: 'six'}
    yy = sorted({int(r['fold']) for r in rows['TEST_CV']})
    cap = "\\textbf{Share of cell-days whose emission inputs to the networks of modeled daily mean surface NO$_2$ fall outside the cell's training range, on the contiguous US land cells of the 12 km grid, held-out years %d to %d (percent).} A cell-day counts when any of the %s emission input channels lies below the cell's minimum or above its maximum over the training days of its held-out year's fold (values before scaling). (a) The test days of each held-out year; (b) the days held back from every year, evaluated once and kept apart. Filled circles show all land cells and open markers the urban (urban fraction at least %s), intermediate (%s to under %s) and rural cells (under %s), set side by side within each year, where the sideways position carries no value. Under each panel, tiles give the same shares in its year columns, in percent to two significant figures; a darker tile is a higher share. Every network trained on the other four years, at every seed, has the same shares, because all read the same five emission channels. The residual U-Net with climatological emissions (2016, 2018 and 2019) has a share of 0 by construction. The reference U-Net with 19 inputs adapted from those of Salman et al.\\ (2024), trained only with 2016 held out, and the networks trained on a single year (Appendix Table~\\ref{tab:alltrain}) are not shown." % (yy[0], yy[-1], words.get(nch[0], str(nch[0])), C.num_plain(uhi, 1), C.num_plain(ulo, 1), C.num_plain(uhi, 1), C.num_plain(ulo, 1))
    cap = re.sub('\\s+', ' ', cap).strip()
    print('caption words', len(re.sub('\\\\[a-zA-Z]+|[{}$]', ' ', cap).split()))
    alt = 'Two dot charts with heat strips of the share of cell-days with emission inputs outside training range, by year and class.'
    parts = ['Vertical axis of the upper panels: Percent of cell-days out of training range, ticks 0 to 100 by 20, shared; under each panel a heat strip of the same shares, one column per year (2015 to 2019, horizontal axis Held-out year) and one row per class, the rows titled Land cells: All (filled circle), Urban (open square), Intermediate (open triangle), Rural (open downward triangle); each tile prints the percent, darker for a higher share.']
    for j, (part, _) in enumerate(PARTS):
        parts.append('Panel (%s) %s: ' % ('ab'[j], names[part]) + '; '.join(('%d: %s' % (y, ', '.join(('%s %s' % (SNAME[u], C.num_plain(100.0 * data[part][str(y)][u]['share'], 1) + ' percent') for u in STRATA))) for y in S.YEARS)) + '.')
    long_desc = '\n'.join(parts)
    FC.write_text(STEM, cap, alt, long_desc)
    _ = texts
if __name__ == '__main__':
    main()
