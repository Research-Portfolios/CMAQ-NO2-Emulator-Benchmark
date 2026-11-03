"""The drawing part of figure fig_A_F2: the daily series panels, the daily differences of each U-Net from CMAQ, the month axis and the key."""
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import build_fig_common as C
import build_style as S
import matplotlib.pyplot as plt
W = 6.5
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
LSTEPS = [75, 61, 47, 33, 19]

def rect(H, x, ytop, w, h):
    return [x / W, (H - ytop - h) / H, w / W, h / H]

def lab(v, fmt='%g'):
    return (fmt % v).replace('-', S.MINUS)

def shade(base, dark, lstar):
    w = np.ones(3)
    b = C._lin(C.hex2rgb(base))
    d = C._lin(C.hex2rgb(dark))

    def at(s):
        return (1 - s) * w + s * b if s < 1 else (2 - s) * b + (s - 1) * d
    lo, hi = (0.0, 2.0)
    for _ in range(50):
        s = 0.5 * (lo + hi)
        if C._lin2lab(at(s))[0] > lstar:
            lo = s
        else:
            hi = s
    return C.rgb2hex(C._delin(at(0.5 * (lo + hi))))

def month_axis(ax, year, labels, every=1):
    st = np.array(C.month_starts(year) + [367 if year % 4 == 0 else 366], dtype=float)
    mid = 0.5 * (st[:-1] + st[1:])
    ax.set_xlim(0.5, 366.5)
    ax.set_xticks(st[:-1])
    ax.tick_params(axis='x', length=S.TICK_LEN_PT, width=S.TICK_W_PT, labelbottom=False)
    if labels:
        ax.set_xticks(mid[::every], minor=True)
        ax.set_xticklabels(MONTHS[::every], minor=True)
        ax.tick_params(axis='x', which='minor', length=0, labelbottom=True, pad=2.5)

def draw(arr, days, mbrow, D, plotted, ov_top=None):
    years = S.YEARS
    diff = {y: {m: arr[y, m][1] - arr[y, m][0] for m in ('M', 'P')} for y in years}
    dmax = max((float(np.abs(d).max()) for y in years for d in diff[y].values()))
    slim = 0.5
    assert dmax < slim, dmax
    plotted['daily_diff_unet_minus_cmaq'] = {'rule': 'row 1 minus row 0 of F5_domain_fold<y>_<M|P>.npy (the U-Net minus CMAQ, land-cell mean) on each test day', 'values': {str(y): diff[y] for y in years}, 'largest_abs_ppb': dmax}
    shades = {'M': [shade(S.MODEL_COLOUR['M'], '#3A1A00', v) for v in LSTEPS], 'P': [shade(S.MODEL_COLOUR['P'], '#001A33', v) for v in LSTEPS]}
    plotted['year_shades_colours'] = shades
    X0, XRB = (0.62, 0.6)
    AXW = W - X0 - XRB
    H_LEV, H_GAP, H_STR, H_ROW = (0.24, 0.08, 0.3, 0.27)
    TOP_AX = 0.5
    blk = 5 * (H_LEV + H_GAP + H_STR) + 4 * H_ROW
    OV_TOP = TOP_AX + blk + 0.36 + 0.24 if ov_top is None else ov_top
    OV_H = 0.66
    GX = 0.66
    OV_W = (AXW - GX) / 2.0
    H = OV_TOP + OV_H + 0.44
    fig = plt.figure(figsize=(W, H))
    lev, stp = ([], [])
    for i, y in enumerate(years):
        doy = days[y][0]
        ytop = TOP_AX + i * (H_LEV + H_GAP + H_STR + H_ROW)
        ax = fig.add_axes(rect(H, X0, ytop, AXW, H_LEV))
        sx = fig.add_axes(rect(H, X0, ytop + H_LEV + H_GAP, AXW, H_STR))
        lev.append(ax)
        stp.append(sx)
        month_axis(ax, y, labels=False)
        month_axis(sx, y, labels=i == 4)
        a, b = (arr[y, 'M'], arr[y, 'P'])
        x, (cm, em, ep) = C.with_gaps(doy, a[0], a[1], b[1])
        ax.plot(x, cm, color=S.CMAQ_LEVEL, lw=2.0, solid_capstyle='butt', zorder=2)
        ax.plot(x, ep, color=S.MODEL_COLOUR['P'], lw=0.6, ls=S.MODEL_LS['P'], zorder=3)
        ax.plot(x, em, color=S.MODEL_COLOUR['M'], lw=0.6, ls=S.MODEL_LS['M'], zorder=4)
        ax.set_ylim(0, D)
        ax.set_yticks(np.arange(0, D + 1, 2))
        x, (dm, dp) = C.with_gaps(doy, diff[y]['M'], diff[y]['P'])
        sx.plot(x, dp, color=S.MODEL_COLOUR['P'], lw=0.55, ls=S.MODEL_LS['P'], zorder=3)
        sx.plot(x, dm, color=S.MODEL_COLOUR['M'], lw=0.55, ls=S.MODEL_LS['M'], zorder=4)
        sx.axhline(0, color=S.REF_INK, lw=0.6, zorder=5)
        sx.set_ylim(-slim, slim)
        sx.set_yticks([-0.4, 0, 0.4])
        sx.set_yticklabels([lab(v) for v in (-0.4, 0, 0.4)])
        sx.yaxis.tick_right()
        for a_ in (ax, sx):
            C.frame(a_)
        if y == 2019 and plotted.get('2019_withheld_is_december'):
            d0 = min(plotted['2019_withheld_day_of_year']) - 0.5
            d1 = max(plotted['2019_withheld_day_of_year']) + 0.5
            for a_ in (ax, sx):
                a_.add_patch(Rectangle((d0, a_.get_ylim()[0]), d1 - d0, a_.get_ylim()[1] - a_.get_ylim()[0], facecolor='#E6E6E6', lw=0, zorder=0))
        C.tag(ax, 'abcde'[i], '%d held out' % y)
    C.block_title(fig, lev + stp, 'NO$_2$ (ppb)', side='left')
    rt_ = C.block_title(fig, lev + stp, 'U-Net minus CMAQ (ppb)', side='right')
    fig.canvas.draw()
    rr = fig.canvas.get_renderer()
    l2 = max((tk.label2.get_window_extent(rr).x1 for sx_ in stp for tk in sx_.yaxis.get_major_ticks() if tk.label2.get_visible() and tk.label2.get_text()))
    got = (rt_.get_window_extent(rr).x0 - l2) * 72.0 / fig.dpi
    x_, y_ = rt_.get_position()
    rt_.set_position((x_ + (S.AXIS_LABELPAD_PT - got) / 72.0 / W, y_))
    C.block_title(fig, [stp[-1]], 'Month of the held-out year', axis='x')
    ovs = []
    for j, m in enumerate(('M', 'P')):
        x = X0 + j * (OV_W + GX)
        ax = fig.add_axes(rect(H, x, OV_TOP, OV_W, OV_H))
        ovs.append(ax)
        month_axis(ax, 2015, labels=True, every=3)
        ax.axhline(0, color=S.REF_INK, lw=0.6, zorder=7)
        for k, y in enumerate(years):
            xx, (v,) = C.with_gaps(days[y][0], diff[y][m])
            ax.plot(xx, v, color=shades[m][k], lw=0.8, ls='-', zorder=1 + k)
        ax.set_ylim(-slim, slim)
        ax.set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
        ax.set_yticklabels([lab(v) for v in (-0.4, -0.2, 0, 0.2, 0.4)])
        if j == 1:
            ax.tick_params(labelleft=False)
        for k, y in enumerate(years):
            yy = slim * (0.8 - 0.4 * k)
            ax.plot([372, 388], [yy] * 2, color=shades[m][k], lw=2.0, clip_on=False, solid_capstyle='butt')
            ax.text(393, yy, str(y), color=S.INK, fontsize=S.FS_TICK, va='center', ha='left', clip_on=False)
        C.frame(ax)
        C.tag(ax, 'fg'[j], '%s minus CMAQ' % S.PLAIN[m])
    ovs[0].set_ylabel('U-Net minus CMAQ (ppb)', fontsize=S.FS_LABEL)
    C.block_title(fig, ovs, 'Month of each held-out year', axis='x')
    hs = [Line2D([], [], color=S.CMAQ_LEVEL, lw=3.0, solid_capstyle='butt'), Line2D([], [], color=S.MODEL_COLOUR['M'], lw=1.0, ls=S.MODEL_LS['M'], solid_capstyle='butt', dash_capstyle='butt'), Line2D([], [], color=S.MODEL_COLOUR['P'], lw=1.0, ls=S.MODEL_LS['P'], solid_capstyle='butt', dash_capstyle='butt'), Line2D([], [], color=S.REF_INK, lw=0.6)]
    leg = C.house_key(fig, lev + stp, hs, ['CMAQ', S.PLAIN['M'], S.PLAIN['P'], 'Zero difference'], side='above')[0]
    for t, c in zip(leg.get_texts(), (S.INK, S.MODEL_TEXT['M'], S.MODEL_TEXT['P'], S.INK)):
        t.set_color(c)
    if ov_top is None:
        fig.canvas.draw()
        rr = fig.canvas.get_renderer()
        mt = [t for t in fig.texts if t.get_text().startswith('Month of the')][0]
        mb = mt.get_window_extent(rr).y0
        kt = max((t.get_window_extent(rr).y1 for a in ovs for t in a.texts if (t.get_gid() or '').startswith('tagline:')))
        d_in = (mb - kt) / fig.dpi - S.BLOCK_GAP_PT / 72.0
        plt.close(fig)
        return draw(arr, days, mbrow, D, plotted, ov_top=OV_TOP - d_in)
    return (fig, H, {'legend': leg, 'lev': lev, 'stp': stp, 'ovs': ovs})
