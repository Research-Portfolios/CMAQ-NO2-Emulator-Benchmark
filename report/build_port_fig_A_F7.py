"""The drawing part of figure fig_A_F7: the shares as markers for all land cells and each urban class, with a strip of gray tiles that gives each share in percent."""
import math
import numpy as np
import matplotlib.pyplot as plt
import build_style as S
import build_fig_common as FC
STRATA = ['all', 'urban', 'intermediate', 'rural']
RNAME = {'all': 'All', 'urban': 'Urban', 'intermediate': 'Intermediate', 'rural': 'Rural'}
STYLE = {'all': dict(marker='o', ms=5.0, mfc=S.INK, mec=S.INK, mew=0.8), 'urban': dict(marker='s', ms=4.6, mfc='white', mec=S.INK, mew=0.9), 'intermediate': dict(marker='^', ms=5.0, mfc='white', mec=S.INK, mew=0.9), 'rural': dict(marker='v', ms=5.0, mfc='white', mec=S.INK, mew=0.9)}
STEP = 0.18
LIGHT, DARK = ('#E6E6E6', '#222222')

def sig2(p):
    if p == 0:
        return ('0', 0.0)
    v = float('%.2g' % p)
    nd = max(0, 1 - int(math.floor(math.log10(abs(v)))))
    s = ('%.' + str(nd) + 'f') % v
    return (s, float(s))

def grey(t):
    a, b = (FC.hex2rgb(LIGHT), FC.hex2rgb(DARK))
    c = a + (b - a) * t
    return '#%02X%02X%02X' % tuple((int(round(v * 255)) for v in c))

def draw(data, parts, names, y_top, years):
    W = 6.5
    L, R, hg = (1.1, 0.1, 0.2)
    pw = (W - L - R - hg) / 2.0
    top_pad, tagl, ah, sgap, sh, below = (0.08, 0.22, 2.05, 0.08, 1.0, 0.42)
    H = top_pad + tagl + ah + sgap + sh + below + 0.04
    fig = plt.figure(figsize=(W, H))
    ys = below + 0.04
    ya = ys + sh + sgap
    n = len(years)
    off = {u: (k - 1.5) * STEP for k, u in enumerate(STRATA)}
    allv = [data[p][str(y)][u]['share'] for p in parts for y in years for u in STRATA]
    vmax = max(allv)
    drawn = {'figure_in': [W, float(H)], 'y_axis': 'percent (share x 100)'}
    nv6 = {}
    tile_ink = {}
    top = 100.0 * y_top
    yt = np.arange(0.0, top + 1e-09, 20.0)
    sxs = []
    for j, part in enumerate(parts):
        x0 = L + j * (pw + hg)
        ax = fig.add_axes([x0 / W, ya / H, pw / W, ah / H])
        sx = fig.add_axes([x0 / W, ys / H, pw / W, sh / H])
        for i, y in enumerate(years):
            for u in STRATA:
                ax.plot([i + off[u]], [100.0 * data[part][str(y)][u]['share']], ls='none', zorder=3, **STYLE[u])
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_xticks(range(n))
        ax.tick_params(axis='x', labelbottom=False, length=2.5)
        ax.set_ylim(0.0, top)
        ax.set_yticks(yt)
        ax.set_yticklabels(['%d' % v for v in yt])
        if j == 0:
            ax.set_ylabel('Percent of cell-days out of training range')
        else:
            ax.tick_params(axis='y', labelleft=False)
        FC.frame(ax)
        FC.tag(ax, 'ab'[j], names[part])
        nv6[part] = {}
        tile_ink[part] = {}
        for i, y in enumerate(years):
            nv6[part][str(y)] = {}
            for r, u in enumerate(STRATA):
                v = data[part][str(y)][u]['share']
                t = v / vmax
                col = grey(t)
                sx.add_patch(plt.Rectangle((i - 0.5, len(STRATA) - 1 - r - 0.5), 1.0, 1.0, facecolor=col, edgecolor='white', lw=1.2))
                txt, num = sig2(100.0 * v)
                nv6[part][str(y)][u] = num
                ink = 'white' if FC.lab_of(col)[0] < 55 else S.INK
                tile_ink[part]['%s %s' % (y, u)] = [col, ink]
                sx.text(i, len(STRATA) - 1 - r, txt, ha='center', va='center', fontsize=S.FS_TICK, color=ink)
        sx.set_xlim(-0.5, n - 0.5)
        sx.set_ylim(-0.5, len(STRATA) - 0.5)
        sx.set_xticks(range(n))
        sx.set_xticklabels([str(y) for y in years])
        FC.house_ticks(sx, 'x')
        sx.set_yticks(range(len(STRATA)))
        if j == 0:
            sx.set_yticklabels([RNAME[u] for u in STRATA[::-1]])
            sx.set_ylabel('Land cells')
            sx.tick_params(axis='y', direction='out', length=S.TICK_LEN_PT, width=S.TICK_W_PT, pad=17)
            for r, u in enumerate(STRATA[::-1]):
                sx.plot([-0.5 - 0.11 * n / pw * 1.0], [r], ls='none', clip_on=False, zorder=4, **STYLE[u])
        else:
            sx.tick_params(axis='y', left=False, labelleft=False)
        sxs.append(sx)
        FC.frame(sx)
    FC.block_title(fig, sxs, 'Held-out year', axis='x')
    drawn['tile_scale'] = {'light': LIGHT, 'dark': DARK, 'rule': 'linear in share / largest share drawn', 'largest_share': vmax}
    drawn['tile_inks'] = tile_ink
    return (fig, drawn, nv6)
