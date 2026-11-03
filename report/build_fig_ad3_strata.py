"""Draws figure fig_ad3_strata: the mean error and RMSE of four models by season and by urban class, each value checked against the table."""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_style as S
import build_fig_common as C
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
STEM = 'fig_ad3_strata'
TAB = os.path.join(S.RUN, 'tex', 'tab_A_T5.tex')
MODELS = ['M', 'P', 'GBT', 'CLIM-WD']
HEAD = {'M': 'Residual U-Net', 'P': 'Reference U-Net', 'GBT': 'Gradient-boosted trees', 'CLIM-WD': 'Weekday climatology'}
SEASONS = ['winter', 'spring', 'summer', 'fall']
SEASON_LABEL = {'winter': 'Winter (JFD)', 'spring': 'Spring (MAM)', 'summer': 'Summer (JJA)', 'fall': 'Fall (SON)'}
URBAN = ['urban', 'intermediate', 'rural']
BOUNDS = [-0.2, -0.05, 0.05, 0.2]
CLASSES = [S.K4_NEG[0], S.K4_NEG[2], S.K4_ANCHOR, S.K4_POS[1], S.K4_POS[3]]
CLASS_NAMES = ['Below −0.2', '−0.2 to −0.05', '−0.05 to 0.05', '0.05 to 0.2', '0.2 and above']
RMAX = 3.5
SQ = 0.3

class _Sq:

    def __init__(self, side_pt):
        self.side = side_pt

class _SqHandler(HandlerBase):

    def create_artists(self, legend, h, xd, yd, w, ht, fs, trans):
        y = (ht - yd) / 2.0
        x1 = -xd + w
        return [Line2D([-xd, x1], [y, y], color='none', lw=0, transform=trans), Rectangle((x1 - h.side, y - h.side / 2.0), h.side, h.side, facecolor='none', edgecolor=S.INK, lw=0.8, transform=trans)]

def me_class(v):
    k = 0
    while k < len(BOUNDS) and v >= BOUNDS[k]:
        k += 1
    return k

def tnum(v):
    s = '%.3f' % v
    if float(s) == 0.0 and s.startswith('-'):
        s = s[1:]
    return s

def printed():
    out = {}
    with open(TAB, encoding='ascii') as f:
        for ln in f:
            if not ln.strip().startswith('\\quad'):
                continue
            p = [x.strip().replace('\\textminus{}', '-') for x in ln.replace('\\\\', '').split('&')]
            name = p[0].replace('\\quad', '').strip().lower().split(' (')[0]
            for k, m in enumerate(MODELS):
                out[name, m] = (p[1 + 2 * k], p[2 + 2 * k])
    return out

def main():
    src = C.sealed('metrics.json')
    met = C.rows('metrics.json')
    pr = printed()
    val = {}
    for kind, strata in (('season', SEASONS), ('urban', URBAN)):
        for st in strata:
            for m in MODELS:
                s = 0 if m in ('M', 'P') else None
                rr = []
                for y in S.YEARS:
                    x = [x for x in met if x['model'] == m and x['seed'] == s and (x['fold'] == y) and (x['stratum_kind'] == kind) and (x['stratum'] == st)]
                    assert len(x) == 1, (m, y, st, len(x))
                    rr.append(x[0])
                e = float(np.mean([x['RMSE'] for x in rr]))
                b = float(np.mean([x['MB'] for x in rr]))
                assert tnum(e) == pr[st, m][0], (st, m, e, pr[st, m])
                assert tnum(b) == pr[st, m][1], (st, m, b, pr[st, m])
                val['%s|%s|%s' % (kind, st, m)] = {'RMSE': e, 'ME': b, 'ME_text': tnum(b), 'class': me_class(b), 'years_RMSE': [x['RMSE'] for x in rr], 'years_ME': [x['MB'] for x in rr]}
    ramp = C.ramp_measure(CLASSES)
    C.setup()
    W = 6.5
    tw, th, gx, gy = (1.16, 0.46, 0.06, 0.06)
    lab_w = 0.8
    gridw = 4 * tw + 3 * gx
    left = (W - (lab_w + gridw + 2 * gx)) / 2.0 + lab_w
    ha = 4 * th + 3 * gy + 2 * gy
    hb = 3 * th + 2 * gy + 2 * gy
    sizes = (0.5, 1.0, 2.0, 3.0)
    STEP_IN, BAR_H = (0.4, 0.11)
    PAIR_GAP_PT = 3.0 * S.KEY_COLSPACE_EM * S.KEY_FS

    def draw(top, H, layout=None):
        fig = S.plt.figure(figsize=(W, H))
        axes = []
        ytop = H - top
        for blk, (kind, strata, hh) in enumerate((('season', SEASONS, ha), ('urban', URBAN, hb))):
            if blk == 1:
                ytop -= ha + 0.6
            ax = fig.add_axes([left / W, (ytop - hh) / H, (gridw + 2 * gx) / W, hh / H])
            ax.set_xlim(0, gridw + 2 * gx)
            ax.set_ylim(0, hh)
            ycent = []
            for i, st in enumerate(strata):
                y0 = hh - gy - (i + 1) * th - i * gy
                ycent.append(y0 + th / 2)
                for j, m in enumerate(MODELS):
                    v = val['%s|%s|%s' % (kind, st, m)]
                    x0 = gx + j * (tw + gx)
                    k = v['class']
                    ax.add_patch(Rectangle((x0, y0), tw, th, facecolor=CLASSES[k], edgecolor='none', zorder=1))
                    side = SQ * np.sqrt(v['RMSE'] / RMAX)
                    ax.add_patch(Rectangle((x0 + tw / 2 - 0.23 - side / 2, y0 + th / 2 - side / 2), side, side, facecolor='none', edgecolor='white' if k in (0, 4) else S.INK, lw=0.8, zorder=2))
                    dark = k in (0, 4)
                    ax.text(x0 + tw / 2 - 0.01, y0 + th / 2, v['ME_text'].replace('-', S.MINUS), ha='left', va='center', fontsize=S.FS_TICK, color='white' if dark else S.INK, zorder=3)
            ax.set_yticks(ycent)
            ax.set_yticklabels([SEASON_LABEL.get(s, s.capitalize()) for s in strata])
            ax.set_ylabel('Season' if kind == 'season' else 'Urban class', fontsize=S.FS_LABEL)
            C.house_ticks(ax, 'y')
            if blk == 0:
                ax.xaxis.tick_top()
                ax.xaxis.set_label_position('top')
                ax.xaxis.label.set_va('bottom')
                ax.set_xticks([gx + j * (tw + gx) + tw / 2 for j in range(4)])
                ax.set_xticklabels([HEAD[m] for m in MODELS])
                ax.set_xlabel('Model', fontsize=S.FS_LABEL)
                C.house_ticks(ax, 'x')
            else:
                ax.set_xticks([])
            C.frame(ax)
            axes.append(ax)
        C.tag(axes[0], 'a', 'By season', gap_pt=C.title_gap_over(axes[0]))
        C.tag(axes[1], 'b', 'By urban class')
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        ga = (axes[0].get_tightbbox(r).y0 - axes[1].get_tightbbox(r).y1) * 72.0 / fig.dpi
        p = axes[1].get_position()
        dy = (ga - S.BLOCK_GAP_PT) / 72.0 / H
        axes[1].set_position([p.x0, p.y0 + dy, p.width, p.height])
        S.register_block(fig, 'season', [axes[0]])
        S.register_block(fig, 'urban', [axes[1]])
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        fr = axes[1].get_window_extent(r)
        tb_ = axes[1].get_tightbbox(r)
        bx0, bx1 = (fr.x0 / fig.dpi, fr.x1 / fig.dpi)
        key_top = tb_.y0 / fig.dpi - S.KEY_GAP_PT / 72.0
        t = fig.text(0, 0, 'Hg0', fontsize=S.KEY_FS)
        th_in = t.get_window_extent(r).height / fig.dpi
        t.remove()
        bw = 5 * STEP_IN
        side_by_side = layout['side_by_side'] if layout else True
        if layout:
            xb = layout['xb']
            xl = layout['xl']
        else:
            xb = 0.5 * (bx0 + bx1) - bw / 2.0
            xl = None
        y_bar = key_top - th_in - S.BAR_TITLE_GAP_PT / 72.0 - BAR_H
        sk = C.stepped_bar_key(fig, [xb / W, y_bar / H, bw / W, BAR_H / H], CLASSES, BOUNDS, title='Mean error (ppb)')
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        sides = [SQ * np.sqrt(v / RMAX) for v in sizes]
        labs = ['%.1f' % v for v in sizes]
        tmp = [fig.text(0, 0, s_, fontsize=S.FS_TICK) for s_ in labs]
        lw_ = [t_.get_window_extent(r).width / fig.dpi for t_ in tmp]
        for t_ in tmp:
            t_.remove()
        g1 = S.KEY_TEXTPAD_EM * S.KEY_FS / 72.0
        g2 = S.KEY_COLSPACE_EM * S.KEY_FS / 72.0
        rk_w = sum((s_ + g1 + w_ for s_, w_ in zip(sides, lw_))) + g2 * (len(sizes) - 1)
        if side_by_side:
            rx0 = xl if xl is not None else xb + bw + 0.5
            rtop = key_top
        else:
            yb = min(sk['cax'].get_tightbbox(r).y0 / fig.dpi, y_bar)
            rx0 = 0.5 * (bx0 + bx1) - rk_w / 2.0
            rtop = yb - S.KEY_GAP_PT / 72.0
        kax = fig.add_axes([0, 0, 1, 1])
        kax.set_gid('key')
        kax.set_xlim(0, W)
        kax.set_ylim(0, H)
        kax.axis('off')
        kax.patch.set_visible(False)
        rt = kax.text(rx0 + rk_w / 2.0, rtop, 'RMSE (ppb)', ha='center', va='top', fontsize=S.KEY_FS, color=S.INK)
        th_r = rt.get_window_extent(r).height / fig.dpi
        yc = rtop - th_r - S.BAR_TITLE_GAP_PT / 72.0 - max(sides) / 2.0
        x_ = rx0
        for s_, w_, lab_ in zip(sides, lw_, labs):
            kax.add_patch(Rectangle((x_, yc - s_ / 2.0), s_, s_, facecolor='none', edgecolor=S.INK, lw=0.8))
            kax.text(x_ + s_ + g1, yc, lab_, ha='left', va='center', fontsize=S.FS_TICK, color=S.INK)
            x_ += s_ + g1 + w_ + g2
        fig.canvas.draw()
        cb_ = S._key_content_box(kax, fig.canvas.get_renderer())
        xa_, xe_ = (cb_.x0 / fig.dpi, cb_.x1 / fig.dpi)
        ya_, ye_ = (cb_.y0 / fig.dpi, cb_.y1 / fig.dpi)
        kax.set_position([xa_ / W, ya_ / H, (xe_ - xa_) / W, (ye_ - ya_) / H])
        kax.set_xlim(xa_, xe_)
        kax.set_ylim(ya_, ye_)
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        sk['cax'].set_gid('noframe')
        bar_box = type(fr).union([sk['cax'].get_tightbbox(r), sk['title'].get_window_extent(r)])
        lg_box = S._key_content_box(kax, r)
        k = 72.0 / fig.dpi
        meas = {'bar_key_w_pt': round(float(bar_box.width * k), 2), 'rmse_key_w_pt': round(float(lg_box.width * k), 2), 'pair_gap_pt': PAIR_GAP_PT, 'inner_gap_max_pt': S.KEY_COLSPACE_EM * S.KEY_FS, 'room_pt': round((W - 0.2) * 72.0, 2), 'side_by_side': bool(side_by_side)}
        meas['pair_w_pt'] = round(meas['bar_key_w_pt'] + PAIR_GAP_PT + meas['rmse_key_w_pt'], 2)
        meas['fits'] = bool(meas['pair_w_pt'] <= meas['room_pt'])
        if side_by_side:
            meas['title_tops_pt'] = [round(float(sk['title'].get_window_extent(r).y1 * k), 2), round(float(rt.get_window_extent(r).y1 * k), 2)]
            u = type(fr).union([bar_box, lg_box])
            meas['pair_centre_offset_pt'] = round(float((0.5 * (u.x0 + u.x1) - 0.5 * (fr.x0 + fr.x1)) * k), 2)
            meas['clear_gap_pt'] = round(float((lg_box.x0 - bar_box.x1) * k), 2)
        return (fig, axes, sk, kax, meas, bar_box, lg_box)
    MARG = 0.1
    top0, H0 = (1.0, 9.0)
    fig, axes, sk, lg, meas, bb_, lb_ = draw(top0, H0)
    dpi = fig.dpi
    S.plt.close(fig)
    if meas['fits']:
        wb_in = bb_.width / dpi
        pair = wb_in + PAIR_GAP_PT / 72.0 + lb_.width / dpi
        mid = left + (gridw + 2 * gx) / 2.0
        bar_off = bb_.x0 / dpi - (mid - 2.5 * STEP_IN)
        xb = mid - pair / 2.0 - bar_off
        xl = mid - pair / 2.0 + wb_in + PAIR_GAP_PT / 72.0
        lay = {'side_by_side': True, 'xb': xb, 'xl': xl}
    else:
        lay = {'side_by_side': False, 'xb': None, 'xl': None}
    if not lay['side_by_side']:
        lay = None
    fig, axes, sk, lg, meas, bb_, lb_ = draw(top0, H0, lay or {'side_by_side': False, 'xb': left + (gridw + 2 * gx) / 2.0 - 2.5 * STEP_IN, 'xl': None})
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    S.plt.close(fig)
    top = top0 - (H0 - tb.y1) + MARG
    H = round(tb.y1 - tb.y0 + 2 * MARG, 3)
    fig, axes, sk, lg, meas, bb_, lb_ = draw(top, H, lay or {'side_by_side': False, 'xb': left + (gridw + 2 * gx) / 2.0 - 2.5 * STEP_IN, 'xl': None})
    fig.canvas.draw()
    r_ = fig.canvas.get_renderer()
    tg = [t_ for t_ in axes[0].texts if (t_.get_gid() or '').startswith('tagline:')]
    tb0 = min((t_.get_window_extent(r_).y0 for t_ in tg))
    meas['title_a_gap_to_model_title_pt'] = round(float((tb0 - axes[0].xaxis.label.get_window_extent(r_).y1) * 72.0 / fig.dpi), 2)
    print('keys', meas)
    plotted = {'label': 'DESC, added after the plan was fixed', 'values': val, 'me_class_bounds_ppb': BOUNDS, 'me_class_colours': CLASSES, 'ramp_measure': ramp, 'rmse_square_side_in_at': {'RMAX_ppb': RMAX, 'side_in': SQ, 'area_proportional': True}, 'key_rmse_sizes_ppb': list(sizes), 'printed_check': 'every RMSE and ME equal to tex\\tab_A_T5.tex at 3 decimals (asserted)', 'figure_in': [W, H], 'layout_keys_pair': meas}
    plotted['fc_check'] = C.fc_check(fig)
    plotted['text_check'] = C.text_check(fig)
    fcc, tc = (plotted['fc_check'], plotted['text_check'])
    print('ramp', ramp)
    print('fc_check pass', fcc['pass'], 'case_pass', fcc['case_pass'], fcc['case'], fcc['overlaps'], fcc['outside'], fcc['small'], fcc['open_frames'])
    print('text_check pass', tc['pass'], tc['overlaps'], tc['within_0.08in_of_edge'], 'fig', tc['figure_in'])
    S.save(fig, STEM, plotted, [src, TAB])
    cap = '\\textbf{Mean error and RMSE (ppb) of four models against modeled CMAQ daily mean surface NO$_2$ by season (a) and by urban class (b), test days of the five held-out years, 2015 to 2019, contiguous US land cells.} Each tile is one model in one stratum and shows the mean of the five yearly values: its fill and printed value give the mean error (model minus CMAQ; purple below zero, green above, pale gray within 0.05~ppb of zero), and the area of the open square gives the RMSE. The values are those of Table~\\ref{tab:strata}. Seasons are those of the held-out year (JFD, its own January, February and December, the December of 2019 held back or kept as buffer days; MAM, March to May; JJA, June to August; SON, September to November); urban classes by urban fraction (urban at least 0.5, intermediate 0.1 to under 0.5, rural under 0.1). Networks at seed 0. The figure was not specified in advance; the values it shows were.'
    alt = 'Tiles of mean error and RMSE by season and urban class for four models.'
    parts = ["Two blocks of tiles: (a) by season, rows winter (JFD, the held-out year's own January, February and December, the December of 2019 held back or kept as buffer days), spring (MAM, March to May), summer (JJA, June to August), fall (SON, September to November); (b) by urban class, rows urban, intermediate, rural. Columns: residual U-Net, reference U-Net, gradient-boosted trees, weekday climatology. Each tile's color and printed number give the mean error in ppb; an open square's area gives the RMSE. Values, RMSE then mean error:"]
    for kind, strata in (('season', SEASONS), ('urban', URBAN)):
        for st in strata:
            parts.append('%s: %s.' % (st.capitalize(), '; '.join(('%s %.3f and %s' % (HEAD[m], val['%s|%s|%s' % (kind, st, m)]['RMSE'], val['%s|%s|%s' % (kind, st, m)]['ME_text']) for m in MODELS))))
    C.write_text(STEM, cap, alt, '\n'.join(parts))
    print('words', len(cap.split()))
if __name__ == '__main__':
    main()
