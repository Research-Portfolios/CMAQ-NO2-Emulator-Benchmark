"""The drawing part of figure fig_A_F4: the profile across wavelength bands of the two U-Nets and the per-cell network, then one panel per band with every network on its own row, on one log ratio axis."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator
import build_style as S
import build_fig_common as FC
BAR_A = 0.32
TICKS = [0.5, 0.7, 1.0]

def tint(hx, a):
    c = FC.hex2rgb(hx) * a + (1 - a)
    return '#%02X%02X%02X' % tuple((int(round(v * 255)) for v in c))

def tl(v):
    return '%.1f' % v

def ratio_axis(ax, which, lim):
    (ax.xaxis if which == 'x' else ax.yaxis).keep_tick_text = True
    if which == 'x':
        ax.set_xscale('log')
        ax.set_xlim(*lim)
        ax.xaxis.set_major_locator(FixedLocator(TICKS))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([tl(t) for t in TICKS])
    else:
        ax.set_yscale('log')
        ax.set_ylim(*lim)
        ax.yaxis.set_major_locator(FixedLocator(TICKS))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_yticklabels([tl(t) for t in TICKS])

def draw(data, models, row_label, bands, band_name, colour, marker, open_, lim):
    W = 6.5
    drawn = {'ratio_axis': 'log', 'ratio_limits': list(lim), 'ratio_ticks': TICKS}
    rowlab = {m: row_label[m] for m in models}
    prof = [m for m in models if m not in open_]
    kord = list(prof)
    L0 = 1.8
    R0 = 0.14
    gut = (S.GUTTER_PT + 3.1) / 72.0
    pw = (W - L0 - R0 - 3 * gut) / 4.0
    hb, ha = (2.12, 2.01)

    def build(H, ya, yb0):
        fig = plt.figure(figsize=(W, H))
        ypos = {m: len(models) - 1 - i for i, m in enumerate(models)}
        axes = []
        for j, b in enumerate(bands):
            bx = fig.add_axes([(L0 + j * (pw + gut)) / W, yb0 / H, pw / W, hb / H])
            axes.append(bx)
            bx.axvline(1.0, color=S.REF_INK, lw=0.6, zorder=1)
            for m in models:
                v = data[m][b]
                y = ypos[m]
                bx.plot([v['min'], v['max']], [y, y], color=tint(colour[m], BAR_A), lw=4.0, solid_capstyle='butt', zorder=2)
                if m in open_:
                    bx.plot([v['mean']], [y], ls='none', marker=marker[m], ms=4.8 * 0.7, mfc='none', mec=S.REF_INK, mew=0.8, zorder=3)
                else:
                    bx.plot([v['mean']], [y], ls='none', marker=marker[m], ms=4.6, color=colour[m], mec='white', mew=0.4, zorder=3)
            ratio_axis(bx, 'x', lim)
            bx.set_ylim(-0.6, len(models) - 0.4)
            bx.set_yticks([ypos[m] for m in models])
            if j == 0:
                bx.set_yticklabels([rowlab[m] for m in models])
                for t, m in zip(bx.get_yticklabels(), models):
                    t.set_color(S.MODEL_TEXT.get(m, S.INK))
                bx.set_ylabel('Network')
            else:
                bx.set_yticklabels([])
            FC.frame(bx)
            FC.house_ticks(bx, 'both')
            FC.tag(bx, 'bcde'[j], band_name[b])
        x0a = L0 / W
        wa = (4 * pw + 3 * gut) / W
        ax = fig.add_axes([x0a, ya / H, wa, ha / H])
        xs = np.arange(len(bands))
        dx = {m: (k - (len(prof) - 1) / 2.0) * 0.1 for k, m in enumerate(prof)}
        for m in prof:
            mean = np.array([data[m][b]['mean'] for b in bands])
            x = xs + dx[m]
            for xi, b in zip(x, bands):
                ax.plot([xi, xi], [data[m][b]['min'], data[m][b]['max']], color=tint(colour[m], BAR_A), lw=4.0, solid_capstyle='butt', zorder=2)
            ls = S.MODEL_LS.get(m, (0, (1.0, 1.4)))
            ax.plot(x, mean, color=colour[m], lw=1.2 if m in ('M', 'P') else 0.9, ls=ls, zorder=3)
            ax.plot(x, mean, ls='none', marker=marker[m], ms=4.6, color=colour[m], mec='white', mew=0.4, zorder=4)
        ax.axhline(1.0, color=S.REF_INK, lw=0.6, zorder=1)
        ratio_axis(ax, 'y', lim)
        ax.set_xlim(-0.5, len(bands) - 0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels([band_name[b].replace(' km', '') for b in bands])
        ax.set_xlabel('Wavelength band (km)')
        ax.set_ylabel('Power ratio, network over CMAQ')
        FC.frame(ax)
        FC.house_ticks(ax, 'both')
        FC.tag(ax, 'a', 'Profile across the wavelength bands')
        bt = FC.block_title(fig, axes, 'Power ratio, network over CMAQ', axis='x')
        S.register_block(fig, 'profile', [ax])
        S.register_block(fig, 'bands', axes)
        hk = {m: Line2D([], [], color=colour[m], lw=1.2 if m in ('M', 'P') else 0.9, ls=S.MODEL_LS.get(m, (0, (1.0, 1.4))), marker=marker[m], ms=4.6, mec='white', mew=0.4) for m in prof}
        legs = FC.house_key(fig, [ax], [hk[m] for m in kord] + [Line2D([], [], color=S.REF_INK, lw=0.6, marker='o', ms=4.6, mfc='none', mec='none', mew=0.4)], [row_label[m] for m in kord] + ['Equal power'], side='above')
        for t, m in zip(legs[0].get_texts(), kord):
            t.set_color(S.MODEL_TEXT.get(m, S.INK))
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        top = max((lg.get_window_extent(r).y1 for lg in legs)) / fig.dpi
        atight = ax.get_tightbbox(r).y0 / fig.dpi
        btight = max((a.get_tightbbox(r).y1 for a in axes)) / fig.dpi
        bot = min([a.get_tightbbox(r).y0 for a in axes] + [bt.get_window_extent(r).y0]) / fig.dpi
        return (fig, ax, axes, legs, top, atight, btight, bot)
    H1, ya1 = (7.0, 4.0)
    fig, ax, axes, legs, top, atight, btight, bot = build(H1, ya1, 0.5)
    plt.close(fig)
    ya = ya1 + (btight + S.BLOCK_GAP_PT / 72.0) - atight
    shift = 0.06 - bot
    top2 = top + (ya - ya1) + shift
    H = top2 + 0.06
    fig, ax, axes, legs, top, atight, btight, bot = build(H, ya + shift, 0.5 + shift)
    drawn['figure_in'] = [W, round(H, 3)]
    drawn['a_key_order'] = kord
    return (fig, drawn, [ax] + axes)
