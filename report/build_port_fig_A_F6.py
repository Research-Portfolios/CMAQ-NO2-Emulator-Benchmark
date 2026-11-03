"""The drawing part of figure fig_A_F6: the four models on the full scale, a dot at the five-year mean on a bar over the range of the five years, and the same values magnified."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
import build_style as S
import build_fig_common as FC
MODELS = ['M', 'P', 'GBT', 'CLIM-WD']
NAME = {'M': 'Residual U-Net', 'P': 'Reference U-Net', 'GBT': 'Gradient-boosted trees', 'CLIM-WD': 'Weekday climatology'}
SEASONS = ['winter', 'spring', 'summer', 'fall']
URBAN = ['urban', 'intermediate', 'rural']
COAST = ['edge', 'coast', 'border', 'interior']
CNAME = {'edge': 'Grid edge', 'coast': 'Coast', 'border': 'Border', 'interior': 'Interior'}
DECILES = list(range(1, 11))
BAR_A = 0.3
MS = {'M': 4.4, 'P': 4.2, 'GBT': 4.4, 'CLIM-WD': 4.2}

def tint(hx, a):
    c = FC.hex2rgb(hx) * a + (1 - a)
    return '#%02X%02X%02X' % tuple((int(round(v * 255)) for v in c))

def tk(v, nd=1):
    s = ('%.' + str(nd) + 'f') % v
    return '0' if float(s) == 0.0 else s.replace('-', S.MINUS)

def draw(data, cells, y_top_a, y_top_bc):
    W = 6.5
    EDGE = 0.09
    HA, HBC, HE, HF = (0.75, 0.72, 0.86, 1.3)
    H0 = 9.0
    fig = plt.figure(figsize=(W, H0))
    drawn = {}

    def box(x0, y0, w, h):
        return fig.add_axes([x0 / W, y0 / H0, w / W, h / H0])

    def vbars(ax, xs, keys, table, off):
        for x, key in zip(xs, keys):
            for m in MODELS:
                d = data[table][key][m]
                xx = x + off[m]
                ax.plot([xx, xx], [d['min'], d['max']], color=tint(S.MODEL_COLOUR[m], BAR_A), lw=3.2, solid_capstyle='butt', zorder=2)
                ax.plot([xx], [d['mean']], ls='none', marker=S.MODEL_MARKER[m], ms=MS[m], color=S.MODEL_COLOUR[m], mec='white', mew=0.4, zorder=3)

    def hbars(ax, ys, keys, table, lim, models, beyond=None):
        dy = {m: (k - (len(models) - 1) / 2.0) * 0.2 for k, m in enumerate(models)}
        for y, key in zip(ys, keys):
            for m in models:
                d = data[table][key][m]
                yy = y - dy[m]
                if d['mean'] > lim[1]:
                    if m != 'CLIM-WD':
                        raise SystemExit('STOP: a learned model beyond the magnified axis')
                    beyond.append((key, y, d['mean']))
                    continue
                if m != 'CLIM-WD':
                    if d['min'] < lim[0] or d['max'] > lim[1]:
                        raise SystemExit('STOP: a range beyond the axis')
                    ax.plot([d['min'], d['max']], [yy, yy], color=tint(S.MODEL_COLOUR[m], BAR_A), lw=3.0, solid_capstyle='butt', zorder=2)
                ax.plot([d['mean']], [yy], ls='none', marker=S.MODEL_MARKER[m], ms=MS[m] - 0.4, color=S.MODEL_COLOUR[m], mec='white', mew=0.4, zorder=3)
    off = {m: (k - 1.5) * 0.19 for k, m in enumerate(MODELS)}
    ax_a = box(0.6, 7.0, 5.7, HA)
    gsp = 0.7
    xs_a, cats_a, centres = ([], [], {})
    for gi, u in enumerate(URBAN):
        base = gi * (4 + gsp)
        for si, s in enumerate(SEASONS):
            xs_a.append(base + si)
            cats_a.append('%s|%s' % (s, u))
        centres[u] = base + 1.5
    vbars(ax_a, xs_a, cats_a, 'D4', off)
    ax_a.set_ylim(0.0, y_top_a)
    yt = np.arange(0.0, y_top_a + 1e-09, 2.0)
    ax_a.set_yticks(yt)
    ax_a.set_yticklabels([tk(v, 0) for v in yt])
    ax_a.set_xlim(-0.6, xs_a[-1] + 0.6)
    ax_a.set_xticks(xs_a)
    ax_a.set_xticklabels([s.capitalize() for s in SEASONS] * 3)
    FC.house_ticks(ax_a, 'x')
    ax_a.set_ylabel('RMSE (ppb)')
    FC.frame(ax_a)
    gname = {'urban': 'Urban cells', 'intermediate': 'Intermediate cells', 'rural': 'Rural cells'}
    fig.canvas.draw()
    rr = fig.canvas.get_renderer()
    hs_pt = max((t.get_window_extent(rr).height for t in ax_a.get_xticklabels())) * 72.0 / fig.dpi
    ax_a.set_xticks([centres[u] for u in URBAN], minor=True)
    ax_a.set_xticklabels([gname[u] for u in URBAN], minor=True, fontsize=S.FS_TICK)
    ax_a.tick_params(axis='x', which='minor', length=0, pad=S.TICK_LEN_PT + S.TICK_PAD_PT + hs_pt + 4.5)
    ax_a.set_xlabel('Season and urban class')
    FC.tag(ax_a, 'a', 'By season and urban class')
    ax_b = box(0.6, 5.0, 1.8, HBC)
    ax_c = box(2.8, 5.0, 3.5, HBC)
    vbars(ax_b, range(4), COAST, 'D5', off)
    vbars(ax_c, range(10), [str(k) for k in DECILES], 'D6', off)
    ytb = np.arange(0.0, y_top_bc + 1e-09, 0.5)
    for ax in (ax_b, ax_c):
        ax.set_ylim(0.0, y_top_bc)
        ax.set_yticks(ytb)
        ax.set_yticklabels([tk(v, 1) for v in ytb])
        FC.house_ticks(ax, 'x')
        FC.frame(ax)
    ax_c.tick_params(axis='y', labelleft=False)
    ax_b.set_xlim(-0.6, 3.6)
    ax_b.set_xticks(range(4))
    ax_b.set_xticklabels([CNAME[c] for c in COAST])
    ax_b.set_xlabel('Coast and edge class')
    ax_b.set_ylabel('RMSE (ppb)')
    ax_c.set_xlim(-0.6, 9.6)
    ax_c.set_xticks(range(10))
    ax_c.set_xticklabels([str(k) for k in DECILES])
    ax_c.set_xlabel('Boundary-layer height decile (1 shallowest)')
    FC.tag(ax_b, 'b', 'By coast and edge class')
    FC.tag(ax_c, 'c', 'By boundary-layer height decile')
    learned = ['M', 'P', 'GBT']
    rows_d, keys_d, y = ([], [], 0.0)
    for u in URBAN:
        for s in SEASONS:
            keys_d.append('%s|%s' % (s, u))
            rows_d.append(y)
            y -= 1.0
        y -= 0.5
    n_d = -rows_d[-1] + 1.0
    ax_d = box(1.3, 0.6, 1.9, 3.5)
    lim_d = (0.0, float(np.ceil(max((data['D4'][k][m]['max'] for k in keys_d for m in learned)) / 0.1) * 0.1))
    beyond = []
    hbars(ax_d, rows_d, keys_d, 'D4', lim_d, MODELS, beyond)
    ax_d.set_xlim(*lim_d)
    ax_d.set_ylim(rows_d[-1] - 0.6, 0.6)
    ax_d.set_yticks(rows_d)
    ax_d.set_yticklabels(['%s, %s' % (k.split('|')[0].capitalize(), k.split('|')[1]) for k in keys_d])
    FC.house_ticks(ax_d, 'y')
    xtd = np.arange(0.0, lim_d[1] + 1e-09, 0.5)
    ax_d.set_xticks(xtd)
    ax_d.set_xticklabels([tk(v, 1) for v in xtd])
    ax_d.set_xlabel('RMSE (ppb)')
    ax_d.set_ylabel('Season and urban class')
    for key, yy, v in beyond:
        ax_d.annotate('→ %.2f' % v, xy=(1.0, yy), xycoords=('axes fraction', 'data'), xytext=(4, 0), textcoords='offset points', ha='left', va='center', fontsize=S.FS_TICK, color=S.INK, annotation_clip=False)
    FC.frame(ax_d)
    FC.tag(ax_d, 'd', 'As (a), magnified')
    drawn['d_xlim'] = list(lim_d)
    drawn['d_climatology_beyond_axis'] = [[k, v] for k, _, v in beyond]
    drawn['d_rows'] = n_d
    vals_ef = [data[t][k][m][f] for t, ks in (('D5', COAST), ('D6', [str(k) for k in DECILES])) for k in ks for m in learned for f in ('min', 'max')]
    lim_ef = (float(np.floor(min(vals_ef) / 0.1) * 0.1), float(np.ceil(max(vals_ef) / 0.1) * 0.1))
    clim_ef = [data[t][k]['CLIM-WD']['mean'] for t, ks in (('D5', COAST), ('D6', [str(k) for k in DECILES])) for k in ks]
    while any((lim_ef[1] - 0.05 < v <= lim_ef[1] for v in clim_ef)):
        lim_ef = (lim_ef[0], round(lim_ef[1] + 0.1, 10))
    ax_e = box(4.5, 3.0, 1.4, HE)
    ys_e = [-i for i in range(4)]
    beyond_e = []
    hbars(ax_e, ys_e, COAST, 'D5', lim_ef, MODELS, beyond_e)
    ax_e.set_yticks(ys_e)
    ax_e.set_yticklabels([CNAME[c] for c in COAST])
    ax_f = box(4.5, 0.6, 1.4, HF)
    ys_f = [-i for i in range(10)]
    beyond_f = []
    hbars(ax_f, ys_f, [str(k) for k in DECILES], 'D6', lim_ef, MODELS, beyond_f)
    ax_f.set_yticks(ys_f)
    ax_f.set_yticklabels(['1 (shallowest)'] + [str(k) for k in DECILES[1:-1]] + ['10 (deepest)'])
    for ax, bb in ((ax_e, beyond_e), (ax_f, beyond_f)):
        for key, yy, v in bb:
            ax.annotate('→ %.2f' % v, xy=(1.0, yy), xycoords=('axes fraction', 'data'), xytext=(4, 0), textcoords='offset points', ha='left', va='center', fontsize=S.FS_TICK, color=S.INK, annotation_clip=False)
    drawn['ef_climatology_beyond_axis'] = [[k, v] for k, _, v in beyond_e + beyond_f]
    xte = np.round(np.arange(np.ceil(lim_ef[0] / 0.4 - 1e-09) * 0.4, lim_ef[1] + 1e-09, 0.4), 1)
    for ax, ys in ((ax_e, ys_e), (ax_f, ys_f)):
        ax.set_xlim(*lim_ef)
        ax.set_ylim(ys[-1] - 0.6, 0.6)
        FC.house_ticks(ax, 'y')
        ax.set_xticks(xte)
        ax.set_xticklabels([tk(v, 1) for v in xte])
        ax.set_xlabel('RMSE (ppb)')
        FC.frame(ax)
    ax_e.set_ylabel('Coast and edge class')
    ax_f.set_ylabel('Boundary-layer height decile')
    FC.tag(ax_e, 'e', 'As (b), magnified')
    FC.tag(ax_f, 'f', 'As (c), magnified')
    hd = [(Line2D([0], [0], ls='none', marker='s', ms=7.0, color=tint(S.MODEL_COLOUR[m], BAR_A)), Line2D([0], [0], ls='none', marker=S.MODEL_MARKER[m], ms=MS[m], color=S.MODEL_COLOUR[m], mec='white', mew=0.4)) for m in MODELS]
    hmap = {tuple: HandlerTuple(ndivide=1)}
    probe = fig.legend(hd, [NAME[m] for m in MODELS], ncol=4, fontsize=S.KEY_FS, frameon=False, borderpad=0.0, handlelength=S.KEY_HANDLELENGTH_EM, handler_map=hmap)
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    rr = fig.canvas.get_renderer()
    key_h = probe.get_window_extent(rr).height / fig.dpi
    fig.legends.remove(probe)
    E = {}
    for a in (ax_a, ax_b, ax_c, ax_d, ax_e, ax_f):
        f = a.get_window_extent(rr)
        t = a.get_tightbbox(rr)
        E[id(a)] = {'l': (f.x0 - t.x0) / fig.dpi, 'r': (t.x1 - f.x1) / fig.dpi, 't': (t.y1 - f.y1) / fig.dpi, 'b': (f.y0 - t.y0) / fig.dpi}
    ea, eb, ec, ed, ee, ef = [E[id(a)] for a in (ax_a, ax_b, ax_c, ax_d, ax_e, ax_f)]
    gut = S.GUTTER_PT / 72.0
    gut_def = (S.GUTTER_PT + 0.5) / 72.0
    bgap = S.BLOCK_GAP_PT / 72.0
    L = EDGE + max(ea['l'], eb['l'])
    Ld = EDGE + ed['l']
    R = W - EDGE - max(ea['r'], ec['r'])
    wbc = R - L - eb['r'] - gut - ec['l']
    wb = wbc * 0.33
    xc0 = L + wb + eb['r'] + gut + ec['l']
    lef = max(ee['l'], ef['l'])
    ref = max(ee['r'], ef['r'])
    wde = W - EDGE - ref - Ld - ed['r'] - gut_def - lef
    wd = wde * 0.56
    we = wde - wd
    xe0 = Ld + wd + ed['r'] + gut_def + lef
    top = EDGE + key_h + S.KEY_GAP_PT / 72.0
    ta = top + ea['t']
    tbc = ta + HA + ea['b'] + bgap + max(eb['t'], ec['t'])
    tdef = tbc + HBC + max(eb['b'], ec['b']) + bgap + max(ed['t'], ee['t'])
    tf = tdef + HE + ee['b'] + gut_def + ef['t']
    bdef = tf + HF
    H = bdef + max(ed['b'], ef['b']) + EDGE
    fig.set_size_inches(W, H)

    def put(a, x0, t_from_top, w, h):
        a.set_position([x0 / W, (H - t_from_top - h) / H, w / W, h / H])
    put(ax_a, L, ta, R - L, HA)
    put(ax_b, L, tbc, wb, HBC)
    put(ax_c, xc0, tbc, R - xc0, HBC)
    put(ax_d, Ld, tdef, wd, bdef - tdef)
    put(ax_e, xe0, tdef, we, HE)
    put(ax_f, xe0, tf, we, HF)
    axs = (ax_a, ax_b, ax_c, ax_d, ax_e, ax_f)
    S.register_block(fig, 'a', [ax_a])
    S.register_block(fig, 'b and c', [ax_b, ax_c])
    S.register_block(fig, 'd to f', [ax_d, ax_f, ax_e])
    leg = FC.house_key(fig, list(axs), hd, [NAME[m] for m in MODELS], side='above', handler_map=hmap)
    for lg in leg:
        for t in lg.get_texts():
            m = {NAME[k]: k for k in MODELS}.get(t.get_text())
            t.set_color(S.MODEL_TEXT.get(m, S.INK))
    drawn['ef_xlim_shared'] = list(lim_ef)
    drawn['bc_ylim'] = [0.0, float(y_top_bc)]
    drawn['figure_in'] = [W, float(round(H, 3))]
    return (fig, drawn, axs, gname)
