"""Draws figure fig_ad4_square: the two-by-two of target form (direct or climatology residual) and training length, with the RMSE change along each side and the interaction with its interval."""
import json
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_style as S
import build_fig_common as C
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
STEM = 'fig_ad4_square'
NUMBERS = os.path.join(S.ROOT, 'results', 'report', 'numbers.json')
F8 = os.path.join(S.FIG, 'fig_A_F8_plotted.json')
TAB = os.path.join(S.RUN, 'tex', 'tab_A_T4.tex')
YEARS = [2016, 2019]
CORNER = {'A2': (0, 0), 'A3': (0, 1), 'A4': (1, 0), 'M': (1, 1)}
NAME = {'A2': 'Step 2', 'A3': 'Step 3', 'A4': 'Step 2,\n16,000 steps', 'M': 'Residual U-Net'}
PLAIN_TAB = {'M': 'Residual U-Net', 'A2': 'Step 2: residual blocks, added inputs, log scale', 'A3': 'Step 3: residual target and climatology input', 'A4': 'Step 2, 16,000 steps'}
NET_GREY = '#A8A8A8'
SIDE_INK = '#767676'
INTERVAL = '#4D4D4D'

class _Dot:

    def __init__(self, face, edge):
        self.face, self.edge = (face, edge)

class _DotHandler(HandlerBase):

    def create_artists(self, legend, h, xd, yd, w, ht, fs, trans):
        y = (ht - yd) / 2.0
        return [Line2D([-xd, -xd + w], [y, y], color='none', lw=0, transform=trans), Line2D([-xd + w - 4.0], [y], ls='none', marker='o', ms=8, mfc=h.face, mec=h.edge, mew=0.6, transform=trans)]

class _Int:
    pass

class _IntHandler(HandlerBase):

    def create_artists(self, legend, h, xd, yd, w, ht, fs, trans):
        y = (ht - yd) / 2.0
        return [Line2D([-xd, -xd + w], [y, y], color=INTERVAL, lw=1.1, solid_capstyle='butt', transform=trans), Line2D([-xd + w / 2.0], [y], ls='none', marker='o', ms=3.4, mfc=S.INK, mec=S.INK, transform=trans)]

def tab_rmse():
    out = {}
    with open(TAB, encoding='ascii') as f:
        lines = f.read().splitlines()
    head = [[x.strip() for x in ln.replace('\\\\', '').split('&')] for ln in lines if ln.startswith('Model & Seed & Year')]
    assert head, 'no header line in ' + TAB
    ncol, jr = (len(head[0]), head[0].index('RMSE'))
    for ln in lines:
        p = [x.strip() for x in ln.replace('\\\\', '').split('&')]
        if len(p) == ncol and p[1] == '0' and (p[2] in ('2016', '2019')):
            out[p[0], int(p[2])] = p[jr]
    return out

def ptxt(nb, k):
    return nb[k]['text'].replace('\\textminus{}', S.MINUS)

def main():
    src = C.sealed('metrics.json')
    met = C.rows('metrics.json')
    with open(NUMBERS, encoding='utf-8') as f:
        nb = json.load(f)
    with open(F8, encoding='utf-8') as f:
        f8 = json.load(f)['plotted']['rows_top_to_bottom']
    tr = tab_rmse()
    rmse = {}
    for y in YEARS:
        for m in CORNER:
            x = [x for x in met if x['model'] == m and x['seed'] == 0 and (x['fold'] == y) and (x['stratum_kind'] == 'all')]
            assert len(x) == 1
            assert '%.3f' % x[0]['RMSE'] == tr[PLAIN_TAB[m], y]
            rmse[m, y] = x[0]['RMSE']
    inter = {}
    nv = {}
    for y in YEARS:
        r = [r for r in f8 if r['contrast'] == '2 x 2 interaction' and int(r['fold']) == y]
        assert len(r) == 1
        r = r[0]
        assert '[RMSE(A4) - RMSE(M)] - [RMSE(A2) - RMSE(A3)]' in r['definition'], r['definition']
        assert abs(r['point'] - nb['lad.int.%d' % y]['value']) < 1e-12
        inter[y] = {'point': r['point'], 'lo': r['lo'], 'hi': r['hi'], 'reading': r['reading'], 'level': r['level'], 'printed': nb['lad.int.%d' % y]['text']}
        a = float(tr[PLAIN_TAB['A2'], y]) - float(tr[PLAIN_TAB['A4'], y])
        raw = rmse['A2', y] - rmse['A4', y]
        nv[str(y)] = {'value_from_printed': a, 'value_from_stored': raw, 'text_from_printed': '%.3f' % a, 'text_from_stored': '%.3f' % raw, 'equal_at_3_decimals': '%.3f' % a == '%.3f' % raw, 'drawn': False}
        assert '%.3f' % (rmse['A2', y] - rmse['A3', y]) == nb['lad.a23.%d' % y]['text']
        assert '%.3f' % (rmse['A3', y] - rmse['M', y]) == nb['lad.a3m.%d' % y]['text']
        assert '%.3f' % (rmse['A4', y] - rmse['M', y]) == nb['lad.a4m.%d' % y]['text']
    C.setup()
    W = 6.5
    YLEV = ['Direct', 'Departure']
    YTITLE = 'Target'
    mf = S.plt.figure(figsize=(W, 2.0))
    rr = mf.canvas.get_renderer()
    wlab = max((mf.text(0, 0, s, fontsize=S.FS_TICK).get_window_extent(rr).width for s in YLEV)) / mf.dpi
    wtit = mf.text(0, 0, YTITLE, fontsize=S.FS_LABEL, rotation=90).get_window_extent(rr).width / mf.dpi
    S.plt.close(mf)
    left = round(0.1 + wtit + (S.AXIS_LABELPAD_PT + S.TICK_PAD_PT + S.TICK_LEN_PT) / 72.0 + wlab, 3)
    gut, right = (0.4, 0.2)
    pw = (W - left - gut - right) / 2.0
    ph = pw * 0.92

    def draw(top, H):
        fig = S.plt.figure(figsize=(W, H))
        bottom = H - top - ph
        insets, mains = ([], [])
        for i, y in enumerate(YEARS):
            ax = fig.add_axes([(left + i * (pw + gut)) / W, bottom / H, pw / W, ph / H])
            ax.set_xlim(-0.56, 1.56)
            ax.set_ylim(-0.4, 1.4)
            sides = [('A2', 'A3', ptxt(nb, 'lad.a23.%d' % y)), ('A3', 'M', ptxt(nb, 'lad.a3m.%d' % y)), ('A2', 'A4', None), ('A4', 'M', ptxt(nb, 'lad.a4m.%d' % y))]
            for a, b, t in sides:
                (xa, ya), (xb, yb) = (CORNER[a], CORNER[b])
                sh = 0.16
                p0 = (xa + sh * (xb - xa), ya + sh * (yb - ya))
                p1 = (xb - sh * (xb - xa), yb - sh * (yb - ya))
                if t is None:
                    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=SIDE_INK, lw=0.7)
                    continue
                ax.annotate('', xy=p1, xytext=p0, arrowprops=dict(arrowstyle='-|>,head_width=0.18,head_length=0.35', color=SIDE_INK, lw=0.7, shrinkA=0, shrinkB=0))
                if ya == yb:
                    ax.text((xa + xb) / 2, ya + (0.06 if ya == 1 else -0.06), t, ha='center', va='bottom' if ya == 1 else 'top', fontsize=S.FS_TICK, color=S.INK)
                else:
                    ax.text(xa + (-0.05 if xa == 0 else 0.05), (ya + yb) / 2, t, ha='right' if xa == 0 else 'left', va='center', fontsize=S.FS_TICK, color=S.INK)
            for m, (x, yy) in CORNER.items():
                col = S.MODEL_COLOUR['M'] if m == 'M' else NET_GREY
                ax.plot([x], [yy], ls='none', marker='o', ms=11, mfc=col, mec='white' if m == 'M' else S.INK2, mew=0.6, zorder=4)
                ax.text(x, yy + (0.19 if yy == 1 else -0.15), '%s, %s' % (NAME[m], tr[PLAIN_TAB[m], y]), ha='center', va='bottom' if yy == 1 else 'top', multialignment='center', fontsize=S.FS_TICK, color=S.MODEL_TEXT['M'] if m == 'M' else S.INK)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['%s steps' % nb['rec.steps.p']['text'], '%s steps' % nb['rec.steps.m']['text']])
            ax.set_yticks([0, 1])
            if i == 0:
                ax.set_yticklabels(YLEV)
                ax.set_ylabel(YTITLE, fontsize=S.FS_LABEL)
            else:
                ax.set_yticklabels([])
            C.house_ticks(ax, 'both')
            mains.append(ax)
            C.frame(ax)
            C.tag(ax, 'ab'[i], '%d held out' % y)
            bb = ax.get_position()
            iw, ih = (0.42 * bb.width, 0.11 * bb.height)
            ia = fig.add_axes([bb.x0 + 0.29 * bb.width, bb.y0 + 0.5 * bb.height, iw, ih])
            it = inter[y]
            ia.set_xlim(-0.25, 0.05)
            ia.set_ylim(-1, 1)
            ia.axvline(0.0, color=S.REF_INK, lw=0.6, zorder=1)
            ia.plot([it['lo'], it['hi']], [0, 0], color=INTERVAL, lw=1.1, solid_capstyle='butt', zorder=2)
            ia.plot([it['point']], [0], ls='none', marker='o', ms=3.4, mfc=S.INK, mec=S.INK, zorder=3)
            ia.set_yticks([])
            ia.set_xticks([-0.2, -0.1, 0.0])
            ia.set_xticklabels([S.MINUS + '0.2', S.MINUS + '0.1', '0'])
            C.house_ticks(ia, 'x')
            ia.set_xlabel('Interaction (ppb)', fontsize=S.FS_TICK)
            C.frame(ia)
            insets.append(ia)
        C.block_title(fig, mains, 'Training length', axis='x')
        hd = [_Dot(S.MODEL_COLOUR['M'], 'white'), _Dot(NET_GREY, S.INK2), _Int(), Line2D([0], [0], color=S.REF_INK, lw=0.6)]
        labs = ['Residual U-Net', 'Intermediate network', 'Interaction, 95% interval', 'Zero']
        legs = C.house_key(fig, mains, hd, labs, side='above', handler_map={_Dot: _DotHandler(), _Int: _IntHandler()})
        for lg_ in legs:
            for t_ in lg_.get_texts():
                for m_ in ('M', 'P'):
                    if t_.get_text().startswith(S.PLAIN[m_]):
                        t_.set_color(S.MODEL_TEXT[m_])
        return (fig, mains, insets, legs, labs)
    MARG = 0.1
    top0, H0 = (1.0, 5.0)
    fig, mains, insets, legs, labs = draw(top0, H0)
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    S.plt.close(fig)
    top = top0 - (H0 - tb.y1) + MARG
    H = round(tb.y1 - tb.y0 + 2 * MARG, 3)
    fig, mains, insets, legs, labs = draw(top, H)
    plotted = {'label': 'EXPL, added after the plan was fixed', 'seed': 0, 'rmse': {'%s.%d' % (m, y): rmse[m, y] for y in YEARS for m in CORNER}, 'rmse_printed_tab_A_T4': {'%s.%d' % (m, y): tr[PLAIN_TAB[m], y] for y in YEARS for m in CORNER}, 'sides_printed': {str(y): {'A2-A3': nb['lad.a23.%d' % y]['text'], 'A3-M': nb['lad.a3m.%d' % y]['text'], 'A4-M': nb['lad.a4m.%d' % y]['text']} for y in YEARS}, 'nv_a2_minus_a4_ppb': nv, 'interaction': {str(y): inter[y] for y in YEARS}, 'corners': CORNER, 'key': labs, 'figure_in': [W, H]}
    plotted['fc_check'] = C.fc_check(fig)
    plotted['text_check'] = C.text_check(fig)
    fcc, tc = (plotted['fc_check'], plotted['text_check'])
    print('fc_check pass', fcc['pass'], 'case_pass', fcc['case_pass'], fcc['case'], fcc['overlaps'], fcc['outside'], fcc['small'], fcc['open_frames'])
    print('text_check pass', tc['pass'], tc['overlaps'], tc['within_0.08in_of_edge'], 'fig', tc['figure_in'])
    print('nv', nv)
    S.save(fig, STEM, plotted, [src, NUMBERS, F8, TAB])
    cap = '\\textbf{RMSE (ppb) of modeled daily mean surface NO$_2$ against CMAQ of four networks in a two-by-two design of target (direct, or the departure from the weekday climatology) and training length (%s or %s steps), 2016 (a) and 2019 (b) held out, contiguous US land cells.} The corners are step 2, step 3 (step 2 with the residual target and the climatology input), step 2 trained for 16,000 steps and the residual U-Net (vermilion), each with its RMSE. The number beside each arrow is the RMSE at its tail minus that at its head, from the unrounded values. The center shows the interaction, the gain from the target change at 16,000 steps (step 2 trained for 16,000 steps minus the residual U-Net) minus its gain at 6,000 steps (step 2 minus step 3), as a dot with its unadjusted 95 percent interval. The black line marks zero difference, and an interval narrower than its dot is hidden by the dot. Seed 0; values as in Table~\\ref{tab:alltrain} and Figure~\\ref{fig:AF8}. The figure was not specified in advance; the values it shows were. The interaction was expected in advance to be negative.' % (nb['rec.steps.p']['text'].replace(',', '{,}'), nb['rec.steps.m']['text'].replace(',', '{,}'))
    alt = 'Two-by-two diagrams of target and training length with the RMSE of four networks, 2016 and 2019.'
    parts = ['Two square panels, 2016 and 2019 held out. Horizontal: training length, 6,000 and 16,000 steps; vertical: target, direct and departure from climatology.']
    for y in YEARS:
        parts.append('%d: step 2 %s, step 3 %s, step 2 at 16,000 steps %s, residual U-Net %s ppb; interaction %s ppb (95 percent interval %s to %s).' % (y, tr[PLAIN_TAB['A2'], y], tr[PLAIN_TAB['A3'], y], tr[PLAIN_TAB['A4'], y], tr[PLAIN_TAB['M'], y], inter[y]['printed'].replace('\\textminus{}', 'minus '), nb['lad.int.%d.lo' % y]['text'].replace('\\textminus{}', 'minus '), nb['lad.int.%d.hi' % y]['text'].replace('\\textminus{}', 'minus ')))
    C.write_text(STEM, cap, alt, '\n'.join(parts))
    print('words', len(cap.split()))
if __name__ == '__main__':
    main()
