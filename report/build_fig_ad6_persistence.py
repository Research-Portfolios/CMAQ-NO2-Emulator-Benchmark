"""Draws figure fig_ad6_persistence: for each land cell, the number of held-out years in which the residual U-Net's test-day mean lay below CMAQ's, on the map of fig_F2."""
import json
import math
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_fig_maps as FM
import build_style as S
import build_fig_common as C
import numpy as np
from matplotlib.collections import QuadMesh
from matplotlib.colors import ListedColormap
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
TintPatch = FM.SwatchSet
HandlerTint = FM.HandlerSwatches
MAPKEY_GAP_PT = FM.MAPKEY_GAP_PT
STEM = 'fig_ad6_persistence'
F2 = os.path.join(S.FIG, 'fig_F2_plotted.json')
AF3 = os.path.join(S.FIG, 'fig_A_F3_plotted.json')
RAMP = [S.K4_POS[3], S.K4_POS[2], S.K4_POS[0], S.K4_NEG[3], S.K4_NEG[1], S.K4_NEG[0]]
SCREEN = '\\\\\\\\\\\\'
SCREEN_INK = '#FFFFFF'
SCREEN_LW = 0.6
DOT_S_AD6 = 0.8

def _oriented_cells_path(rr_, kk_):
    import shapely
    from shapely.geometry.polygon import orient
    from matplotlib.path import Path
    AX, AY = (FM.AX, FM.AY)
    q = np.stack([np.column_stack([AX[rr_, kk_], AY[rr_, kk_]]), np.column_stack([AX[rr_, kk_ + 1], AY[rr_, kk_ + 1]]), np.column_stack([AX[rr_ + 1, kk_ + 1], AY[rr_ + 1, kk_ + 1]]), np.column_stack([AX[rr_ + 1, kk_], AY[rr_ + 1, kk_]]), np.column_stack([AX[rr_, kk_], AY[rr_, kk_]])], axis=1)
    u = shapely.union_all(shapely.polygons(q))
    verts, codes = ([], [])
    for p in shapely.get_parts(u):
        if p.geom_type != 'Polygon':
            continue
        p = orient(p, sign=1.0)
        for ring in [p.exterior] + list(p.interiors):
            a = np.asarray(ring.coords)
            verts.extend(a.tolist())
            codes.extend([Path.MOVETO] + [Path.LINETO] * (len(a) - 2) + [Path.CLOSEPOLY])
    return Path(verts, codes)

class ScreenSwatch(object):

    def __init__(self, colours):
        self.colours = list(colours)

class HandlerScreen(FM.HandlerSwatches):

    def create_artists(self, legend, orig, xd, yd, w, h, fs, trans):
        out = FM.HandlerSwatches.create_artists(self, legend, FM.SwatchSet(orig.colours, frame=False), xd, yd, w, h, fs, trans)
        p = Rectangle((-xd, -yd), w, h, facecolor='none', edgecolor=S.FRAME_INK, lw=0.4, hatch=SCREEN, hatchcolor=SCREEN_INK, transform=trans)
        p.set_hatch_linewidth(SCREEN_LW)
        out.append(p)
        return out
TINT = ['#8EA08B', '#B7BDB7', '#E7F1E5', '#E3DCF6', '#C2C0C7', '#9C9AA1']

def tint_measure():

    def dd(c1, c2, v):
        if v == 'greyscale':
            return abs(C.lab_of(c1)[0] - C.lab_of(c2)[0])
        return float(C.de2000(C.lab_of(c1, v), C.lab_of(c2, v)))
    out = {}
    for v in C.VIEWS:
        out[v] = {'tint_to_neighbour_tint': round(min((dd(TINT[j], TINT[j + 1], v) for j in range(5))), 2), 'tint_to_any_class': round(min((dd(t, r, v) for t in TINT for r in RAMP)), 2), 'tint_to_own_class': round(min((dd(TINT[j], RAMP[j], v) for j in range(6))), 2), 'tint_to_white': round(min((dd(t, '#FFFFFF', v) for t in TINT)), 2)}
    n = out['normal']
    out['pass_normal'] = bool(n['tint_to_neighbour_tint'] >= 10.0 and n['tint_to_any_class'] >= 10.0)
    return out

def main():
    with open(F2, encoding='utf-8') as f:
        f2 = json.load(f)['plotted']
    with open(AF3, encoding='utf-8') as f:
        a3 = json.load(f)['plotted']
    s = float(a3['diff_step_s'])
    assert s == float(f2['diff_step_s'])
    d = {2016: np.asarray(f2['values_in_mask_order']['d_M_minus_cmaq'], float)}
    for y in (2015, 2017, 2018, 2019):
        d[y] = np.asarray(a3['values_in_mask_order'][str(y)], float)
    D = np.vstack([d[y] for y in S.YEARS])
    assert D.shape == (5, FM.FLAT.size) and np.isfinite(D).all()
    k = np.signbit(D).sum(axis=0)
    n_neg_zero = int((np.signbit(D) & (D == 0)).sum())
    near = (np.abs(D) < s).all(axis=0)
    N = k.size
    counts = [int((k == j).sum()) for j in range(6)]
    shares = [c / N for c in counts]
    coin = [math.comb(5, j) / 32.0 for j in range(6)]
    ramp_m = C.ramp_measure(RAMP, skip_grey_pairs=(2,))
    assert ramp_m['pass'], ramp_m
    cls12 = k.copy()
    n_below_most = int((k >= 3).sum())
    dot_below = n_below_most < N - n_below_most
    sel = k >= 3 if dot_below else k <= 2
    r_all, k_all = np.divmod(FM.FLAT, FM.NC)
    h_ = FM.DOT_EVERY // 2
    lat = (r_all % FM.DOT_EVERY == 0) & (k_all % FM.DOT_EVERY == 0) | (r_all % FM.DOT_EVERY == h_) & (k_all % FM.DOT_EVERY == h_)
    pick = sel & lat
    screen_path = _oriented_cells_path(r_all[near], k_all[near])
    order = [5, 4, 3, 2, 1, 0]
    near_pct = 100 * near.mean()
    lab_tint = 'Within %s ppb of CMAQ in all five years, %.1f%%' % ('%g' % s, near_pct)
    lab_dot = 'Below CMAQ in three or more years' if dot_below else 'Below CMAQ in two or fewer years'

    def draw(key_h):
        W = 6.5
        M = 0.1
        mw = W - 2 * M
        mh = mw * FM.ASPECT
        H = M + key_h + mh + M
        fig = S.plt.figure(figsize=(W, H))
        ax = fig.add_axes([M / W, (M + key_h) / H, mw / W, mh / H])
        cmap = ListedColormap(RAMP)
        bounds = [-1000000000.0] + [j + 0.5 for j in range(5)] + [1000000000.0]
        FM.draw_map(ax, cls12.astype(float), cmap, bounds)
        from matplotlib.patches import PathPatch
        sp = PathPatch(screen_path, facecolor='none', edgecolor=SCREEN_INK, lw=0, hatch=SCREEN, hatchcolor=SCREEN_INK, zorder=2.6, gid='screen')
        sp.set_hatch_linewidth(SCREEN_LW)
        ax.add_patch(sp)
        ax.scatter(FM.GXC[r_all, k_all][pick], FM.GYC[r_all, k_all][pick], s=DOT_S_AD6, marker='o', color=FM.DOT_GREY, linewidths=0, zorder=3, rasterized=False)
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        k = 72.0 / fig.dpi
        PT = 1.0 / 72.0
        t_ = fig.text(0, 0, 'Hg0', fontsize=S.KEY_FS)
        th = t_.get_window_extent(r).height / fig.dpi
        t_.set_fontsize(S.FS_TICK)
        lh = t_.get_window_extent(r).height / fig.dpi
        t_.remove()
        kw, bh = (4.4, 0.14)
        lab_gap = (S.TICK_LEN_PT + S.TICK_PAD_PT) * PT
        map_bot = M + key_h
        title_top = map_bot - MAPKEY_GAP_PT * PT
        num_bot = title_top - th - S.BAR_TITLE_GAP_PT * PT - lh
        by = num_bot - lab_gap - bh
        kx0 = M + (mw - kw) / 2.0
        kax = fig.add_axes([kx0 / W, by / H, kw / W, bh / H])
        kax.set_xlim(0, 1)
        kax.set_ylim(0, 1)
        kax.set_yticks([])
        kax.set_xticks([])
        x = 0.0
        seg = {}
        for j in order:
            kax.add_patch(Rectangle((x, 0), shares[j], 1, facecolor=RAMP[j], edgecolor='none'))
            seg[j] = (x, x + shares[j])
            x += shares[j]
        C.frame(kax)
        cum, coin_x = (0.0, [])
        over = S.TICK_LEN_PT * PT / bh
        for j in order[:-1]:
            cum += coin[j]
            coin_x.append(cum)
            kax.plot([cum, cum], [-over, 1 + over], color=S.INK, lw=1.0, solid_capstyle='butt', clip_on=False, zorder=5)
        nums, shs = ([], [])
        for j in order:
            a, b = seg[j]
            xm = kx0 + kw * (a + b) / 2.0
            nums.append(fig.text(xm / W, num_bot / H, str(j), ha='center', va='bottom', fontsize=S.FS_TICK, color=S.INK))
            shs.append(fig.text(xm / W, (by - lab_gap) / H, '%.1f%%' % (100 * shares[j]), ha='center', va='top', fontsize=S.FS_TICK, color=S.INK))
        fig.canvas.draw()
        top = max((t.get_window_extent(r).y1 for t in nums)) / fig.dpi
        bot = min((t.get_window_extent(r).y0 for t in shs)) / fig.dpi
        xc = (kx0 + kw / 2.0) / W
        t_up = fig.text(xc, (top + S.BAR_TITLE_GAP_PT * PT) / H, 'Years below CMAQ', ha='center', va='bottom', fontsize=S.KEY_FS, color=S.INK, gid='bartitle')
        t_lo = fig.text(xc, (bot - S.BAR_TITLE_GAP_PT * PT) / H, 'Share of land cells', ha='center', va='top', fontsize=S.KEY_FS, color=S.INK, gid='bartitle:lower')
        C._reg(fig, '_house_bars', {'cax': kax, 'bar': None, 'title': t_up, 'lower_title': t_lo})
        fig.canvas.draw()
        lo_bot = t_lo.get_window_extent(r).y0
        frame_bot = ax.get_window_extent(r).y0
        gap = (frame_bot - lo_bot) * k + S.KEY_GAP_PT
        h_tint = ScreenSwatch([RAMP[j] for j in order])
        h_dot = FM.DotSwatch(RAMP[3] if dot_below else RAMP[2])
        h_hat = Patch(facecolor='white', edgecolor=S.FRAME_INK, lw=0.4, hatch=FM.HATCH, hatchcolor=FM.HATCH_INK)
        h_odd = (Line2D([], [], color='none', lw=0.6), Line2D([], [], ls='none', marker='|', ms=S.KEY_FS, mew=1.0, color=S.INK))

        class HandlerTickOnBar(HandlerTuple):

            def create_artists(self, legend, orig_handle, xd, yd, w, h, fs, trans):
                box = Rectangle((-xd, -yd), w, h, facecolor='white', edgecolor=S.FRAME_INK, lw=0.4, transform=trans)
                ov = 0.25 * h
                tk = Line2D([-xd + w / 2.0] * 2, [-yd - ov, -yd + h + ov], color=S.INK, lw=1.0, solid_capstyle='butt', transform=trans)
                return [box, tk]
        legs = C.house_key(fig, [ax], None, None, side='below', gap_pt=gap, rows=[[(h_tint, lab_tint), (h_dot, lab_dot)], [(h_hat, 'Land not evaluated'), (h_odd, 'Even odds')]], handler_map={ScreenSwatch: HandlerScreen(), FM.DotSwatch: FM.HandlerDots(), tuple: HandlerTickOnBar()})
        fig.canvas.draw()
        key_bot = min((lg.get_window_extent(r).y0 for lg in legs)) / fig.dpi
        tb_up = t_up.get_window_extent(r)
        gaps = {'map_frame_to_top_title': (frame_bot - tb_up.y1) * k, 'top_title_to_numbers': (tb_up.y0 - top * fig.dpi) * k, 'numbers_to_bar': (num_bot - by - bh) * 72.0, 'bar_to_shares': (by - bot) * 72.0, 'shares_to_lower_title': (bot * fig.dpi - t_lo.get_window_extent(r).y1) * k, 'lower_title_to_bottom_row': (lo_bot - max((lg.get_window_extent(r).y1 for lg in legs))) * k, 'bottom_row_entry_gap': S.KEY_COLSPACE_EM * S.KEY_FS, 'key_bottom_to_figure_edge': key_bot * 72.0}
        bar_off = (kx0 + kw / 2.0 - (M + mw / 2.0)) * 72.0
        return (fig, ax, H, key_bot, coin_x, {'gaps_pt': {q: round(v, 2) for q, v in gaps.items()}, 'bar_centre_to_map_centre_pt': round(bar_off, 2), 'bar_width_in': kw, 'mapkey_gap_pt': MAPKEY_GAP_PT})
    key_h = 1.4
    for _ in range(3):
        fig, ax, H, key_bot, coin_x, key_rec = draw(key_h)
        if abs(key_bot - 0.1) < 0.2 / 72.0:
            break
        key_h -= key_bot - 0.1
        S.plt.close(fig)
    W = 6.5
    qm = [c for c in ax.collections if isinstance(c, QuadMesh)]
    assert len(qm) == 1
    arr = np.ma.filled(qm[0].get_array(), np.nan).reshape(FM.NR, FM.NC)
    drawn = arr.ravel()[FM.FLAT]
    drawn_counts = [int((drawn == j).sum()) for j in range(6)]
    drawn_hex = [S.plt.matplotlib.colors.to_hex(qm[0].get_cmap()(qm[0].norm(j))).upper() for j in range(6)]
    sp_ = [p for p in ax.patches if p.get_gid() == 'screen']
    assert len(sp_) == 1
    from matplotlib.path import Path as _P
    _pts = np.column_stack([FM.GXC[r_all, k_all], FM.GYC[r_all, k_all]])
    _sp = sp_[0].get_path()
    _wind = np.zeros(len(_pts), dtype=int)
    _v, _c = (_sp.vertices, _sp.codes)
    _starts = list(np.nonzero(_c == _P.MOVETO)[0]) + [len(_c)]
    for _a, _b in zip(_starts[:-1], _starts[1:]):
        _ring = _v[_a:_b]
        _x, _y = (_ring[:, 0], _ring[:, 1])
        _area = 0.5 * float(np.sum(_x[:-1] * _y[1:] - _x[1:] * _y[:-1]))
        _in = _P(_ring, closed=True).contains_points(_pts)
        _wind += np.where(_in, 1 if _area > 0 else -1, 0)
    in_screen = _wind != 0
    colour_ok = drawn_counts == counts and np.array_equal(drawn, cls12) and np.array_equal(drawn, k) and (drawn_hex == [c.upper() for c in RAMP])
    screen_ok = bool(np.array_equal(in_screen, near))
    print('screen cells drawn', int(in_screen.sum()), 'vector', int(near.sum()), 'equal', screen_ok)
    assert colour_ok, (drawn_counts, counts, drawn_hex, int(in_screen.sum()), int(near.sum()))
    plotted = {'label': 'DESC, added after the plan was fixed', 'source': 'figures\\fig_F2_plotted.json values_in_mask_order d_M_minus_cmaq (2016); figures\\fig_A_F3_plotted.json values_in_mask_order 2015, 2017, 2018, 2019; diff_step_s; geometry of report/build_fig_maps.py', 'mask_sha256': FM.MS.SHA, 'diff_step_s': s, 'years_below_per_cell_mask_order': k, 'within_s_all_years_mask_order': near.astype(int), 'class_ramp_0_to_5': RAMP, 'class_ramp_measure': ramp_m, 'colour_check': {'cells_per_class_drawn': drawn_counts, 'cells_per_class_vector': counts, 'class_colours_drawn': drawn_hex, 'cells_screened_drawn': int(in_screen.sum()), 'cells_within_s_vector': int(near.sum()), 'screen_equals_vector': screen_ok, 'pass': bool(colour_ok)}, 'screen': {'pattern': SCREEN, 'colour': SCREEN_INK, 'linewidth_pt': SCREEN_LW, 'cells': int(near.sum())}, 'dots': {'lattice': 'as report/build_fig_maps.py dot_cells', 'side': 'below CMAQ in 3 to 5 years' if dot_below else 'below CMAQ in 0 to 2 years', 'cells_dotted': int(pick.sum()), 'size_pt2': DOT_S_AD6, 'colour': FM.DOT_GREY}, 'hatch': {'cells_us_land_not_evaluated': int(len(FM.NOTEV_R)), 'cells_row_col': [[int(a_), int(b_)] for a_, b_ in zip(FM.NOTEV_R, FM.NOTEV_K)]}, 'values_stored_as_negative_zero': n_neg_zero, 'nv_cells_per_years_below_0_to_5': counts, 'nv_shares_per_years_below_0_to_5': shares, 'nv_within_s_all_years': {'cells': int(near.sum()), 'share': float(near.mean())}, 'nv_fair_coin_shares_0_to_5': coin, 'fair_coin_tick_positions_5_to_1': coin_x, 'figure_in': [W, H], 'layout': {'key_block': key_rec}}
    plotted['fc_check'] = C.fc_check(fig)
    plotted['text_check'] = C.text_check(fig)
    fcc, tc = (plotted['fc_check'], plotted['text_check'])
    print('fc_check pass', fcc['pass'], 'case_pass', fcc['case_pass'], ascii(fcc['case']), ascii(fcc['overlaps']), ascii(fcc['outside']), fcc['small'], fcc['open_frames'])
    print('text_check pass', tc['pass'], ascii(tc['overlaps']), ascii(tc['within_0.08in_of_edge']), 'fig', tc['figure_in'])
    print('counts', counts, 'near', int(near.sum()), 'colour_ok', colour_ok, 'ramp', ramp_m)
    S.save(fig, STEM, plotted, [F2, AF3])
    cap = "\\textbf{Number of held-out years (2015 to 2019) in which the residual U-Net's modeled test-day mean surface NO$_2$ lay below CMAQ's, per land cell of the contiguous US (12 km grid).} Each year's value comes from the network that did not see that year, as in Figure~\\ref{fig:F2}~(d) and Appendix Figure~\\ref{fig:AF3}. Purple cells lay below CMAQ in three or more of the five years, darker for more years; green cells lay below in two or fewer, darker for fewer. Dots mark the cells %s. A light screen of white lines marks the %s cells within %s ppb of CMAQ (the step of the difference maps) in all five years, and the key gives their share. The bar below the map gives the share of land cells in each class, and the black ticks the shares expected if each year's sign were independent, with even odds; neighboring cells are not independent, so the ticks are a reference only. %s These results are descriptive and were not specified in advance." % ('below CMAQ in three or more years' if dot_below else 'below CMAQ in two or fewer years', C.thousands(int(near.sum())), '%g' % s, FM.tail_tex(len(plotted['hatch']['cells_row_col'])))
    alt = "Map of the contiguous US: in how many of five years the residual U-Net's mean lay below CMAQ's, per cell."
    longd = "A map of the contiguous US on the 12 km grid, each land cell colored by the number of held-out years, 0 to 5, in which the residual U-Net's test-day mean NO2 lay below CMAQ's: purple for 3 to 5 years (darker for more), green for 0 to 2 (darker for fewer). Cells within %s ppb of CMAQ in all five years lie under a light screen of white lines (%s cells, %.1f percent). Dots mark the cells %s. Hatching marks US land not evaluated; state boundaries are drawn. Below, a bar divided by the share of cells in each class, from 5 years down to 0: %s; black ticks mark the shares expected if the years were independent, with even odds. Above the bar its title, Years below CMAQ, over the class numbers; under it the shares and the title Share of land cells; two rows under the key name the screen, the dots, the hatch and the even-odds tick." % ('%g' % s, C.thousands(int(near.sum())), 100 * near.mean(), 'below CMAQ in three or more years' if dot_below else 'below CMAQ in two or fewer years', ', '.join(('%d year%s %.1f percent' % (j, '' if j == 1 else 's', 100 * shares[j]) for j in order)))
    C.write_text(STEM, cap, alt, longd)
    print('words', len(cap.split()), 'H', round(H, 3))
if __name__ == '__main__':
    main()
