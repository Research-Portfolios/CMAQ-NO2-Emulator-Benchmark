"""The drawing part of figure fig_A_F5: the autocorrelation panels by season and lag, with one shared key and one line style per series."""
import numpy as np
from matplotlib.legend_handler import HandlerBase, HandlerTuple
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import build_fig_common as FC
import build_style as S
import matplotlib.pyplot as plt
W = 6.5
NAME = {'M': 'Residual U-Net', 'P': 'Reference U-Net', 'GBT': 'Gradient-boosted trees', 'CLIM-WD': 'Weekday climatology'}
LS = {'M': S.MODEL_LS['M'], 'P': S.MODEL_LS['P'], 'GBT': (0, (4, 1.2, 1, 1.2)), 'CLIM-WD': (0, (1, 1.2))}
SNAME = {'year': 'Whole year', 'winter': 'Winter', 'spring': 'Spring', 'summer': 'Summer', 'fall': 'Fall'}
EN = '–'
SLONG = {'winter': 'Jan, Feb, Dec', 'spring': 'Mar' + EN + 'May', 'summer': 'Jun' + EN + 'Aug', 'fall': 'Sep' + EN + 'Nov'}
SCODE = {'winter': 'JFD', 'spring': 'MAM', 'summer': 'JJA', 'fall': 'SON'}
SEAS4 = ['winter', 'spring', 'summer', 'fall']
SC = ['#C9C9C9', '#A3A3A3', '#7E7E7E', '#5A5A5A']
SMK = {'winter': 'd', 'spring': '+', 'summer': '*', 'fall': 'p'}
SMS = {'winter': 4.4, 'spring': 4.4, 'summer': 4.8, 'fall': 3.8}
SMW = {'winter': 0.0, 'spring': 1.0, 'summer': 0.0, 'fall': 0.0}
SLAGS = {'winter': [4, 8, 12], 'spring': [5, 9, 13], 'summer': [6, 10, 14], 'fall': [7, 11]}
LEVEL_LS = (0, (3.0, 2.0))
PT = 1.0 / 72.0

class SplitPatch(object):

    def __init__(self, colours):
        self.colours = colours

class HandlerSplit(HandlerBase):

    def create_artists(self, legend, orig, xd, yd, w, h, fs, trans):
        out = [Rectangle((-xd, -yd), w, h, facecolor='none', edgecolor='none', lw=0, transform=trans)]
        n = len(orig.colours)
        for i, c in enumerate(orig.colours):
            out.append(Rectangle((-xd + i * w / n, -yd), w / n, h, facecolor=c, edgecolor='none', lw=0, transform=trans))
        return out

def rect(H, x, ytop, w, h):
    return [x / W, (H - ytop - h) / H, w / W, h / H]

def over_white(hexc, a):
    c = a * FC.hex2rgb(hexc) + (1 - a) * 1.0
    return FC.rgb2hex(c)

def season_label(s, form):
    if s not in SLONG:
        return SNAME[s]
    if form == 'long':
        return '%s (%s)' % (SNAME[s], SLONG[s])
    return '%s (%s)' % (SNAME[s], SCODE[s])

def season_form(frame_w_in):
    fig = plt.figure(figsize=(W, 2.0))
    r = fig.canvas.get_renderer()
    rec = {'frame_width_pt': round(frame_w_in * 72.0, 2)}
    for form in ('long', 'code'):
        widths = {}
        for i, s in enumerate(SEAS4):
            t1 = fig.text(0, 0, '(%s)' % 'cdef'[i], fontsize=8, fontweight='bold')
            t2 = fig.text(0, 0, season_label(s, form), fontsize=8)
            w = (t1.get_window_extent(r).width + t2.get_window_extent(r).width) * 72.0 / fig.dpi + 2.5
            widths['(%s) %s' % ('cdef'[i], season_label(s, form))] = round(w, 2)
            t1.remove()
            t2.remove()
        rec[form] = widths
        rec[form + '_fits'] = bool(max(widths.values()) <= frame_w_in * 72.0)
    plt.close(fig)
    form = 'long' if rec['long_fits'] else 'code'
    rec['chosen'] = form
    return (form, rec)

def draw(models, seasons, lags, data, level, ylo, yhi, plotted):
    adj = {'top': 0.0, 'ab': 0.0, 'bc': 0.0, 'bot': 0.0, 'gb': 0.0, 'gc': 0.0}
    for _ in range(4):
        fig, H, art, meas = _draw(models, seasons, lags, data, level, ylo, yhi, adj)
        if all((abs(v) < 0.2 * PT for v in meas['err'].values())):
            break
        for k_, v in meas['err'].items():
            adj[k_] += v if k_ in ('gb', 'gc') else v
        plt.close(fig)
    plotted['layout_season_form'] = meas['season_form']
    plotted['layout_gaps_pt'] = meas['gaps_pt']
    return (fig, H, art)

def _draw(models, seasons, lags, data, level, ylo, yhi, adj):
    lags = np.asarray(lags)

    def arr(m, s, part):
        return np.array(data[m][s][part], float)
    X0 = 0.62
    GB = S.GUTTER_PT * PT + adj['gb']
    GC = S.GUTTER_PT * PT + adj['gc']
    WB = (W - X0 - 0.12 - 4 * GB) / 5.0
    WC = (W - X0 - 0.12 - 3 * GC) / 4.0
    form, frec = season_form(WB)
    TOP_A = 0.8 - adj['top']
    H_A = 1.48
    TOP_B = TOP_A + H_A + 0.62 - adj['ab']
    H_B = 1.04
    TOP_C = TOP_B + H_B + 0.34 - adj['bc']
    H_C = 1.04
    H = TOP_C + H_C + 0.36 - adj['bot']
    fig = plt.figure(figsize=(W, H))
    AX0 = 1.2
    ax = fig.add_axes(rect(H, AX0, TOP_A, W - AX0 - 0.1, H_A))
    off = {'M': 0.27, 'P': 0.09, 'GBT': -0.09, 'CLIM-WD': -0.27}
    n = len(seasons)
    for r, s in enumerate(seasons):
        yb = n - 1 - r
        for m in models:
            lo, hi = (arr(m, s, 'min')[0], arr(m, s, 'max')[0])
            mu = arr(m, s, 'mean')[0]
            ax.plot([lo, hi], [yb + off[m]] * 2, color=over_white(S.MODEL_COLOUR[m], 0.3), lw=4.0, solid_capstyle='butt', zorder=2)
            ax.plot(mu, yb + off[m], S.MODEL_MARKER[m], ms=3.8, color=S.MODEL_COLOUR[m], mec='white', mew=0.4, zorder=3)
    ax.axvline(level, color=S.REF_INK, lw=0.6, ls=LEVEL_LS, zorder=1)
    ax.set_xlim(0, 0.36)
    ax.set_ylim(-0.55, n - 0.45)
    ax.set_yticks(np.arange(n)[::-1])
    ax.set_yticklabels([season_label(s, form) for s in seasons])
    FC.house_ticks(ax, 'both')
    ax.set_xticks([0, 0.1, 0.2, 0.3])
    ax.set_xticklabels(['0', '0.1', '0.2', '0.3'])
    ax.set_xlabel('Median error autocorrelation at lag 1', fontsize=S.FS_LABEL)
    ax.set_ylabel('Season', fontsize=S.FS_LABEL)
    FC.frame(ax)
    FC.tag(ax, 'a', 'Lag 1 by season')
    yt = [v for v in np.round(np.arange(0.0, yhi + 1e-09, 0.1), 1)]

    def lag_axis(a, ylab):
        a.set_xlim(0.4, 14.6)
        a.set_xticks([1, 7, 14])
        a.set_ylim(ylo - 0.02, yhi + 0.02)
        a.set_yticks(yt)
        a.set_yticklabels(['%g' % v for v in yt])
        a.tick_params(labelleft=ylab)
        FC.house_ticks(a, 'both')
        a.axhline(level, color=S.REF_INK, lw=0.6, ls=LEVEL_LS, zorder=1)
        FC.frame(a)
    bax = []
    for i, s in enumerate(seasons):
        x = X0 + i * (WB + GB)
        a = fig.add_axes(rect(H, x, TOP_B, WB, H_B))
        bax.append(a)
        lag_axis(a, i == 0)
        a.fill_between(lags, arr('M', s, 'min'), arr('M', s, 'max'), color=S.MODEL_COLOUR['M'], alpha=0.18, lw=0, zorder=2)
        for part in ('min', 'max'):
            a.plot(lags, arr('P', s, part), color=S.MODEL_COLOUR['P'], lw=0.4, zorder=2)
        for m in models[::-1]:
            a.plot(lags, arr(m, s, 'mean'), color=S.MODEL_COLOUR[m], lw=0.9, ls=LS[m], marker=S.MODEL_MARKER[m], ms=2.4, mew=0, markevery=[0, 1, 2], zorder=4 if m in ('M', 'P') else 3)
        FC.tag(a, 'bcdef'[i], season_label(s, form))
    cax = []
    for j, m in enumerate(models):
        x = X0 + j * (WC + GC)
        a = fig.add_axes(rect(H, x, TOP_C, WC, H_C))
        cax.append(a)
        lag_axis(a, j == 0)
        for k, s in enumerate(SEAS4):
            a.plot(lags, arr(m, s, 'mean'), color=SC[k], lw=0.7, ls='-', marker=SMK[s], ms=SMS[s], mew=SMW[s], mec=SC[k], mfc=SC[k], markevery=[int(np.nonzero(lags == q)[0][0]) for q in SLAGS[s]], zorder=2)
        a.plot(lags, arr(m, 'year', 'mean'), color=S.MODEL_COLOUR[m], lw=1.2, ls=LS[m], marker=S.MODEL_MARKER[m], ms=2.6, mew=0, markevery=[0, 1, 2], zorder=4)
        FC.tag(a, 'ghij'[j], NAME[m])
    xt = FC.block_title(fig, cax, 'Lag (days)', axis='x')
    FC.block_title(fig, [bax[0], cax[0]], 'Median error autocorrelation', axis='y')
    S.register_block(fig, 'lag 1 by season', [ax])
    S.register_block(fig, 'lags', bax + cax)
    hm = [(Line2D([], [], color=S.MODEL_COLOUR[m], lw=1.1, ls=LS[m]), Line2D([], [], color=S.MODEL_COLOUR[m], ls='none', marker=S.MODEL_MARKER[m], ms=3.6, mew=0)) for m in models]
    hs = [(Line2D([], [], color=c, lw=0.9, ls='-'), Line2D([], [], color=c, ls='none', marker=SMK[s], ms=SMS[s] + 0.4, mew=SMW[s], mec=c, mfc=c)) for c, s in zip(SC, SEAS4)]
    hr = SplitPatch([over_white(S.MODEL_COLOUR['M'], 0.3), over_white(S.MODEL_COLOUR['P'], 0.3)])
    hl = Line2D([], [], color=S.REF_INK, lw=0.6, ls=LEVEL_LS)
    rows = [[(h, NAME[m]) for h, m in zip(hm, models)], [(h, season_label(s, form)) for h, s in zip(hs, SEAS4)], [(hr, 'Range over the five years'), (hl, 'Level %g' % level)]]
    legs = FC.house_key(fig, [ax] + bax + cax, None, None, rows=rows, side='above', handler_map={tuple: HandlerTuple(ndivide=1), SplitPatch: HandlerSplit()})
    txtc = {'M': S.MODEL_TEXT['M'], 'P': S.MODEL_TEXT['P'], 'GBT': S.MODEL_COLOUR['GBT'], 'CLIM-WD': '#6B6B6B'}
    names = [NAME[m] for m in models]
    for lg in legs:
        for t in lg.get_texts():
            if t.get_text() in names:
                t.set_color(txtc[models[names.index(t.get_text())]])
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    k = 72.0 / fig.dpi
    Hp = H * fig.dpi
    key_top = max((lg.get_window_extent(r).y1 for lg in legs))
    a_bot = ax.get_tightbbox(r).y0
    b_top = max((a.get_tightbbox(r).y1 for a in bax))
    b_bot = min((a.get_tightbbox(r).y0 for a in bax))
    c_top = max((a.get_tightbbox(r).y1 for a in cax))
    x_bot = xt.get_window_extent(r).y0
    tb = [a.get_tightbbox(r) for a in bax]
    tc = [a.get_tightbbox(r) for a in cax]
    gb = np.mean([(q.x0 - p.x1) * k for p, q in zip(tb, tb[1:])])
    gc = np.mean([(q.x0 - p.x1) * k for p, q in zip(tc, tc[1:])])
    gaps = {'figure_top_to_key_pt': (Hp - key_top) * k, 'gutter_row_b_pt': gb, 'gutter_row_c_pt': gc, 'a_to_lag_rows_pt': (a_bot - b_top) * k, 'row_b_to_row_c_pt': (b_bot - c_top) * k, 'lag_title_to_bottom_pt': x_bot * k}
    err = {'top': (gaps['figure_top_to_key_pt'] - 0.1 * 72) * PT, 'gb': (S.GUTTER_PT - gb) * PT, 'gc': (S.GUTTER_PT - gc) * PT, 'ab': (gaps['a_to_lag_rows_pt'] - S.BLOCK_GAP_PT) * PT, 'bc': (gaps['row_b_to_row_c_pt'] - S.GUTTER_PT) * PT, 'bot': (gaps['lag_title_to_bottom_pt'] - 0.1 * 72) * PT}
    meas = {'err': err, 'season_form': frec, 'gaps_pt': {q: round(v, 2) for q, v in gaps.items()}}
    return (fig, H, {'legs': legs, 'a': ax, 'b': bax, 'c': cax}, meas)
