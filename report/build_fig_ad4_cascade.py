"""Draws figure fig_ad4_cascade: the RMSE at each step between the two U-Nets for 2016 and 2019, with each step's 95 percent interval of the change."""
import json
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_style as S
import build_fig_common as C
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.legend_handler import HandlerBase
STEM = 'fig_ad4_cascade'
NUMBERS = os.path.join(S.ROOT, 'results', 'report', 'numbers.json')
F8 = os.path.join(S.FIG, 'fig_A_F8_plotted.json')
TAB = os.path.join(S.RUN, 'tex', 'tab_A_T4.tex')
YEARS = [2016, 2019]
STAGES = ['P', 'A1', 'A2', 'A3', 'M']
STEPS = [('pa1', 'P', 'A1'), ('a12', 'A1', 'A2'), ('a23', 'A2', 'A3'), ('a3m', 'A3', 'M')]
ROW = {'P': 'Reference U-Net', 'A1': 'Step 1: loss', 'A2': 'Step 2: network', 'A3': 'Step 3: target and climatology', 'M': 'Residual U-Net', 'A4': 'Step 2, 16,000 steps'}
READ_FILL = {'later': '#4F4F4F', 'earlier': '#A8A8A8', 'within': '#E3E3E3', 'inconclusive': '#FFFFFF'}
READ_NAME = {'later': 'Later network closer', 'earlier': 'Earlier network closer', 'within': 'Within the margin', 'inconclusive': 'Across a margin edge'}
SLEEVE = '#BDBDBD'
LINK = '#8C8C8C'

class _Mark:
    pass

class _MarkHandler(HandlerBase):

    def create_artists(self, legend, h, xd, yd, w, ht, fs, trans):
        y = (ht - yd) / 2.0
        return [Line2D([-xd + w - 6.0, -xd + w], [y, y], color='none', lw=0, transform=trans), Line2D([-xd + w - 3.0], [y], ls='none', marker='o', ms=5.0, mfc='white', mec=S.INK, mew=0.9, transform=trans)]

class _Up:

    def __init__(self, color, lw, ls='-'):
        self.color, self.lw, self.ls = (color, lw, ls)

class _UpHandler(HandlerBase):

    def create_artists(self, legend, h, xd, yd, w, ht, fs, trans):
        cx = -xd + w / 2.0
        return [Line2D([cx, cx], [-yd, -yd + ht], color=h.color, lw=h.lw, ls=h.ls, solid_capstyle='butt', transform=trans)]

class _Iv:
    pass

class _IvHandler(HandlerBase):

    def create_artists(self, legend, h, xd, yd, w, ht, fs, trans):
        y = (ht - yd) / 2.0
        return [Line2D([-xd, -xd + w], [y, y], color=S.INK, lw=1.0, solid_capstyle='butt', transform=trans), Line2D([-xd + w / 2.0], [y], ls='none', marker='o', ms=4.6, mfc='white', mec=S.INK2, mew=0.5, transform=trans)]
MARGIN_LS = (0, (2.5, 1.5))

def tab_rmse():
    out = {}
    with open(TAB, encoding='ascii') as f:
        for ln in f:
            p = [x.strip() for x in ln.replace('\\\\', '').split('&')]
            if len(p) == 8 and p[1] == '0' and (p[2] in ('2016', '2019')):
                out[p[0], int(p[2])] = p[3]
    return out

def tnum(nb, k):
    return nb[k]['text'].replace('\\textminus{}', S.MINUS)

def classify(reading, earlier, later):
    if reading.startswith('the interval excludes'):
        return 'within'
    if reading.startswith('inconclusive'):
        return 'inconclusive'
    who = reading.split(' closer')[0]
    if who == later:
        return 'later'
    if who == earlier:
        return 'earlier'
    raise SystemExit('reading not mapped: %r' % reading)

def main():
    src = C.sealed('metrics.json')
    met = C.rows('metrics.json')
    with open(NUMBERS, encoding='utf-8') as f:
        nb = json.load(f)
    with open(F8, encoding='utf-8') as f:
        f8 = json.load(f)['plotted']['rows_top_to_bottom']
    tr = tab_rmse()
    plain_tab = {'P': 'Reference U-Net', 'M': 'Residual U-Net', 'A1': 'Step 1: squared-error loss', 'A2': 'Step 2: residual blocks, added inputs, log scale', 'A3': 'Step 3: residual target and climatology input', 'A4': 'Step 2, 16,000 steps'}
    rmse = {}
    for y in YEARS:
        for m in STAGES + ['A4']:
            x = [x for x in met if x['model'] == m and x['seed'] == 0 and (x['fold'] == y) and (x['stratum_kind'] == 'all')]
            assert len(x) == 1, (m, y, len(x))
            assert '%.3f' % x[0]['RMSE'] == tr[plain_tab[m], y], (m, y)
            rmse[m, y] = x[0]['RMSE']
    step = {}
    nv = {}
    for y in YEARS:
        for key, a, b in STEPS:
            r = [r for r in f8 if r['contrast'] == 'ladder %s -> %s' % (a, b) and int(r['fold']) == y]
            assert len(r) == 1
            r = r[0]
            assert r['definition'].startswith('RMSE(%s) - RMSE(%s)' % (a, b)), r['definition']
            assert abs(r['point'] - nb['lad.%s.%d' % (key, y)]['value']) < 1e-12
            assert abs(rmse[a, y] - rmse[b, y] - r['point']) < 1e-09
            e3 = float(tr[plain_tab[a], y])
            nv['%s.%d' % (key, y)] = [e3 - r['hi'], e3 - r['lo']]
            step['%s.%d' % (key, y)] = {'earlier': a, 'later': b, 'point': r['point'], 'lo': r['lo'], 'hi': r['hi'], 'reading': r['reading'], 'class': classify(r['reading'], a, b), 'printed': nb['lad.%s.%d' % (key, y)]['text']}
    C.setup()
    W = 6.5
    gut, right = (0.2, 0.12)
    rows = STAGES + ['A4']
    ypos = {m: -(i if m != 'A4' else i + 0.35) for i, m in enumerate(rows)}
    ph = 0.245 * (len(rows) + 0.35) + 0.15
    xlim = (0.4, 0.88)
    bh = 0.46
    YTITLE = 'Network'
    mf = S.plt.figure(figsize=(W, 2.0))
    rr = mf.canvas.get_renderer()
    wmax = max((mf.text(0, 0, ROW[m], fontsize=S.FS_TICK).get_window_extent(rr).width for m in rows)) / mf.dpi
    wtit = mf.text(0, 0, YTITLE, fontsize=S.FS_LABEL, rotation=90).get_window_extent(rr).width / mf.dpi
    S.plt.close(mf)
    lab_w = round(0.1 + wtit + (S.AXIS_LABELPAD_PT + S.TICK_PAD_PT + S.TICK_LEN_PT) / 72.0 + wmax, 3)
    sw, sg = (0.62, S.GUTTER_PT / 72.0)
    pw = (W - lab_w - gut - right - 2 * (sg + sw)) / 2.0
    MRG = 0.05
    s_lo = min([v['lo'] for v in step.values()] + [-MRG])
    s_hi = max([v['hi'] for v in step.values()] + [MRG])
    s_pad = 0.08 * (s_hi - s_lo)
    s_lim = (s_lo - s_pad, s_hi + s_pad)
    s_ticks = [t for t in (-0.2, 0.0, 0.2) if s_lim[0] <= t <= s_lim[1]]
    strip_drawn = {}

    def draw(top, H):
        fig = S.plt.figure(figsize=(W, H))
        bottom = H - top - ph
        axes = []
        strips = []
        for i, y in enumerate(YEARS):
            x0 = lab_w + i * (pw + sg + sw + gut)
            ax = fig.add_axes([x0 / W, bottom / H, pw / W, ph / H])
            sx = fig.add_axes([(x0 + pw + sg) / W, bottom / H, sw / W, ph / H])
            strips.append(sx)
            sx.set_xlim(*s_lim)
            sx.set_ylim(ypos['A4'] - 0.6, 0.6)
            sx.axvline(0.0, color=S.REF_INK, lw=0.6, zorder=1)
            for mv in (-MRG, MRG):
                sx.axvline(mv, color=LINK, lw=0.6, ls=MARGIN_LS, zorder=1)
            for key, a, b in STEPS:
                s = step['%s.%d' % (key, y)]
                yb = ypos[b]
                sc = sx.scatter([s['point']], [yb], s=3.6 ** 2, marker='o', facecolor=READ_FILL[s['class']], edgecolor=S.INK2, linewidths=0.5, hatch='//////' if s['class'] == 'inconclusive' else None, zorder=3)
                strip_drawn['%s.%d' % (key, y)] = {'point': float(sc.get_offsets()[0][0]), 'class': s['class'], 'row': b}
            sx.set_yticks([ypos[m] for m in rows])
            sx.set_yticklabels([])
            C.house_ticks(sx, 'y')
            sx.set_xticks(s_ticks)
            sx.set_xticklabels([('%.1f' % t).replace('-', S.MINUS) if t else '0' for t in s_ticks])
            sx.set_xlabel('Drop (ppb)', fontsize=S.FS_LABEL)
            C.frame(sx)
            ax.set_xlim(*xlim)
            ax.set_ylim(ypos['A4'] - 0.6, 0.6)
            ax.axhline((ypos['M'] + ypos['A4']) / 2.0, color=S.GRID, lw=0.5, zorder=0)
            x0 = rmse['P', y]
            ax.plot([x0, x0], [ypos['P'] - bh / 2, ypos['P'] + bh / 2], color=S.INK, lw=1.2, solid_capstyle='butt', zorder=4)
            for key, a, b in STEPS:
                s = step['%s.%d' % (key, y)]
                xa, xb = (rmse[a, y], rmse[b, y])
                yb = ypos[b]
                ax.plot([xa, xa], [ypos[a] - bh / 2, yb + bh / 2], color=LINK, lw=0.6, zorder=1)
                cls = s['class']
                ax.add_patch(Rectangle((min(xa, xb), yb - bh / 2), abs(xb - xa), bh, facecolor=READ_FILL[cls], edgecolor=S.INK2, lw=0.5, hatch='//////' if cls == 'inconclusive' else None, zorder=2))
                ax.plot([xb, xb], [yb - bh / 2, yb + bh / 2], color=S.INK, lw=1.2, solid_capstyle='butt', zorder=4)
                right_side = max(xa, xb)
                an = ax.annotate(s['printed'].replace('\\textminus{}', S.MINUS), xy=(right_side, yb), xycoords='data', xytext=(2.5, 0.0), textcoords='offset points', ha='left', va='center', fontsize=S.FS_TICK, color=S.INK, zorder=5)
                fig.canvas.draw()
                rr_ = fig.canvas.get_renderer()
                if an.get_window_extent(rr_).x1 > ax.get_window_extent(rr_).x1 - 2.0 * fig.dpi / 72.0:
                    an.xy = (min(xa, xb), yb)
                    an.set_position((-2.5, 0.0))
                    an.set_ha('right')
            xa4 = rmse['A4', y]
            ax.plot([xa4], [ypos['A4']], ls='none', marker='o', ms=5.0, mfc='white', mec=S.INK, mew=0.9, zorder=4)
            ax.set_yticks([ypos[m] for m in rows])
            if i == 0:
                ax.set_yticklabels([ROW[m] for m in rows])
                ax.set_ylabel(YTITLE, fontsize=S.FS_LABEL)
            else:
                ax.set_yticklabels([])
            C.house_ticks(ax, 'y')
            ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8])
            ax.set_xticklabels(['0.4', '0.5', '0.6', '0.7', '0.8'])
            C.frame(ax)
            C.tag(ax, 'ab'[i], '%d held out' % y)
            axes.append(ax)
            ax.set_xlabel('RMSE against CMAQ (ppb)', fontsize=S.FS_LABEL)
        hd = [Patch(facecolor=READ_FILL['later'], edgecolor=S.INK2, lw=0.5), Patch(facecolor=READ_FILL['earlier'], edgecolor=S.INK2, lw=0.5), Patch(facecolor=READ_FILL['within'], edgecolor=S.INK2, lw=0.5), Patch(facecolor='white', edgecolor=S.INK2, lw=0.5, hatch='//////'), _Up(S.INK, 1.2), _Up(LINK, 0.6), _Mark(), _Up(LINK, 0.6, MARGIN_LS), _Up(S.REF_INK, 0.6)]
        labs = [READ_NAME['later'], READ_NAME['earlier'], READ_NAME['within'], READ_NAME['inconclusive'], 'RMSE of the network', "Earlier network's RMSE", 'Step 2, 16,000 steps', 'Margin, 0.05 ppb', 'Zero difference']
        order = [0, 2, 4, 1, 3, 5, 6, 8, 7]
        hd, labs = ([hd[j] for j in order], [labs[j] for j in order])
        hmap = {_Mark: _MarkHandler(), _Up: _UpHandler(), _Iv: _IvHandler()}
        last = C.house_key(fig, axes + strips, hd[6:], labs[6:], ncol=3, side='above', handleheight=0.9, handler_map=hmap, handlelength=6.0 / S.KEY_FS)
        fig.canvas.draw()
        r_ = fig.canvas.get_renderer()
        fr_ = [a_.get_window_extent(r_) for a_ in axes + strips]
        cx_ = 0.5 * (min((f_.x0 for f_ in fr_)) + max((f_.x1 for f_ in fr_)))
        ring_ = [ln_ for ln_ in last[0].findobj(Line2D) if ln_.get_marker() == 'o']
        ink0_ = min((ln_.get_window_extent(r_).x0 for ln_ in ring_)) - 0.45 * fig.dpi / 72.0
        ink1_ = last[0].get_texts()[-1].get_window_extent(r_).x1
        off_ = (0.5 * (ink0_ + ink1_) - cx_) * 72.0 / fig.dpi
        assert abs(off_) <= S.CENTRE_TOL_PT, off_
        print('last key row: ink centre %.2f pt off' % off_)
        tight_ = max((a_.get_tightbbox(r_).y1 for a_ in axes + strips))
        up_ = last[0].get_window_extent(r_).y1
        legs = C.house_key(fig, axes + strips, hd[:6], labs[:6], ncol=3, side='above', handleheight=0.9, handler_map=hmap, gap_pt=(up_ - tight_) * 72.0 / fig.dpi + S.KEY_ROWSPACE_EM * S.KEY_FS) + last
        return (fig, axes, legs, labs)
    MARG = 0.1
    top0, H0 = (1.2, 5.0)
    fig, axes, legs, labs = draw(top0, H0)
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    S.plt.close(fig)
    top = top0 - (H0 - tb.y1) + MARG
    H = round(tb.y1 - tb.y0 + 2 * MARG, 3)
    fig, axes, legs, labs = draw(top, H)
    plotted = {'label': 'EXPL, added after the plan was fixed', 'seed': 0, 'rmse_by_stage': {'%s.%d' % (m, y): rmse[m, y] for y in YEARS for m in STAGES + ['A4']}, 'rmse_printed_tab_A_T4': {'%s.%d' % (m, y): tr[plain_tab[m], y] for y in YEARS for m in STAGES + ['A4']}, 'steps': step, 'nv_sleeve_ends_ppb': nv, 'reading_fill': READ_FILL, 'x_limits': list(xlim), 'key': labs, 'figure_in': [W, H]}
    for k_, v_ in strip_drawn.items():
        for f_ in ('point',):
            assert v_[f_] == step[k_][f_], (k_, f_)
    plotted['strip_drawn'] = strip_drawn
    plotted['strip_x_limits'] = list(s_lim)
    plotted['strip_x_ticks'] = s_ticks
    plotted['strip_margin_ppb'] = MRG
    plotted['fc_check'] = C.fc_check(fig)
    plotted['text_check'] = C.text_check(fig)
    fcc, tc = (plotted['fc_check'], plotted['text_check'])
    print('fc_check pass', fcc['pass'], 'case_pass', fcc['case_pass'], fcc['case'], fcc['overlaps'], fcc['outside'], fcc['small'], fcc['open_frames'])
    print('text_check pass', tc['pass'], tc['overlaps'], tc['within_0.08in_of_edge'], 'fig', tc['figure_in'])
    print('nv', {k: [round(a, 6) for a in v] for k, v in nv.items()})
    print('classes', {k: v['class'] for k, v in step.items()})
    S.save(fig, STEM, plotted, [src, NUMBERS, F8, TAB])
    cap = "\\textbf{RMSE (ppb) of modeled daily mean surface NO$_2$ against CMAQ for each network on the steps from the reference to the residual U-Net, with 2016 (a) and 2019 (b) held out, contiguous US land cells.} Rows run from the reference U-Net through steps 1 to 3, which are defined in Appendix~G, to the residual U-Net, which then trained longer. Black ticks mark the RMSE of each network, and thin gray lines carry the earlier network's RMSE down to the next row. Each bar spans the change at one step and is filled by where that step's unadjusted 95 percent interval (Figure~\\ref{fig:AF8}) lies against the 0.05~ppb margin, as the key names. In the strip at each panel's right, a dot filled as its bar marks each step's drop in RMSE; the black line marks zero difference and dashed gray lines the margin, $-$0.05 and 0.05~ppb. The number beside each bar, and the dot in the strip, is the earlier network's RMSE minus the later one's, so a negative value is a rise in RMSE. The open circle below the thin gray rule is step 2 trained for %s steps, as the residual U-Net; the reference U-Net and steps 1 to 3 trained for %s. At %s steps the step-2 network's RMSE was close to the reference U-Net's, so most of step 2's rise at %s steps went with the shorter training. Seed 0; values as in Table~\\ref{tab:alltrain}. The figure was not specified in advance; the values it shows were." % (nb['rec.steps.m']['text'].replace(',', '{,}'), nb['rec.steps.p']['text'].replace(',', '{,}'), nb['rec.steps.m']['text'].replace(',', '{,}'), nb['rec.steps.p']['text'].replace(',', '{,}'))
    alt = 'Sideways cascade of RMSE across four steps between the two U-Nets, 2016 and 2019.'
    parts = ['Two panels, 2016 and 2019 held out. Rows: ' + '; '.join((ROW[m] for m in rows)) + ". Vertical axis title: Network. A key above the panels names the bar fills Later network closer (dark gray), Earlier network closer (mid gray), Within the margin (light gray) and Across a margin edge (hatched), a black tick (RMSE of the network), a thin gray line (Earlier network's RMSE), an open circle (Step 2, 16,000 steps), a black line (Zero difference) and a dashed gray line (Margin, 0.05 ppb). Horizontal axis: RMSE against CMAQ (ppb). At the right of each panel, a strip with the horizontal axis Drop (ppb) shows each step's change as a dot, filled as its bar, with a black line at 0 and dashed gray lines at -0.05 and 0.05 (the margin). RMSE by network and the change at each step (earlier minus later, ppb):"]
    for y in YEARS:
        parts.append('%d: %s.' % (y, '; '.join(('%s %s' % (ROW[m], tr[plain_tab[m], y]) for m in rows))))
        parts.append('%d steps: %s.' % (y, '; '.join(('%s to %s %s (%s)' % (ROW[a], ROW[b], step['%s.%d' % (k, y)]['printed'].replace('\\textminus{}', '-'), (lambda s_: s_[:1].lower() + s_[1:])(READ_NAME[step['%s.%d' % (k, y)]['class']])) for k, a, b in STEPS))))
    C.write_text(STEM, cap, alt, '\n'.join(parts))
    print('words', len(cap.split()))
if __name__ == '__main__':
    main()
