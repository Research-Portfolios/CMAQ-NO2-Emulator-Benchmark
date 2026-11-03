"""Draws figure fig_ad5_performance: performance diagrams of exceedance skill above 13.30 and 26.60 ppb, each model a box spanning its yearly ranges of probability of detection and success ratio."""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_style as S
import build_fig_common as C
import numpy as np
from matplotlib import patheffects as PE
STEM = 'fig_ad5_performance'
TAB = os.path.join(S.RUN, 'tex', 'tab_A_T2.tex')
LEVELS = [13.3, 26.6]
MODELS = ['M', 'P', 'GBT', 'CLIM-WD']
MS = {'M': 5.2, 'P': 5.2, 'GBT': 4.6, 'CLIM-WD': 4.6}
MEW = 0.45
ZORD = {'M': 4, 'P': 4, 'GBT': 3, 'CLIM-WD': 3}
NEST_FRAC = 1.0 / 3.0
NEST_SCALE = 1.6
CASE = 0.35
ISO = '#BDBDBD'
RAY = '#8C8C8C'
CSI_LEV = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
BIAS = [0.5, 1.0, 1.5, 2.0]

def r2(v):
    return '%.2f' % v

def rng_txt(vals):
    lo, hi = (r2(min(vals)), r2(max(vals)))
    return lo if lo == hi else '%s to %s' % (lo, hi)

def printed():
    out = {}
    lev = None
    with open(TAB, encoding='ascii') as f:
        for ln in f:
            if 'textit{Above 13.30' in ln:
                lev = 13.3
                continue
            if 'textit{Above 26.60' in ln:
                lev = 26.6
                continue
            if 'textit{Above 63.83' in ln:
                lev = None
                continue
            if lev and ln.strip().startswith('\\quad'):
                p = [x.strip() for x in ln.replace('\\\\', '').split('&')]
                name = p[0].replace('\\quad', '').strip()
                out[lev, name] = {'events': p[1], 'POD': p[2], 'FAR': p[3], 'CSI': p[4]}
    return out

def main():
    src = C.sealed('exceedance.json')
    exc = C.rows('exceedance.json')
    pr = printed()
    val = {}
    for lev in LEVELS:
        for m in MODELS:
            s = 0 if m in ('M', 'P') else None
            rr = [x for x in exc if abs(x['level_ppb'] - lev) < 1e-06 and x['model'] == m and (x['seed'] == s)]
            sc = [x for x in rr if x['scored']]
            p = pr[lev, S.PLAIN[m]]
            pod = [x['POD'] for x in sc if x['POD'] is not None]
            far = [x['FAR'] for x in sc if x['FAR'] is not None]
            csi = [x['CSI'] for x in sc if x['CSI'] is not None]
            assert rng_txt(pod) == p['POD'], (lev, m, pod, p)
            assert rng_txt(far) == p['FAR'], (lev, m, far, p)
            assert rng_txt(csi) == p['CSI'], (lev, m, csi, p)
            evs = [x['cmaq_events'] for x in rr]
            assert '{:,} to {:,}'.format(min(evs), max(evs)) == p['events']
            val['%.2f|%s' % (lev, m)] = {'POD_range': [min(pod), max(pod)], 'FAR_range': [min(far), max(far)], 'CSI_range': [min(csi), max(csi)], 'years_scored': [x['fold'] for x in sc], 'years_FAR_defined': [x['fold'] for x in sc if x['FAR'] is not None], 'events_text': p['events'], 'points': [{'model': m, 'level_ppb': lev, 'year': x['fold'], 'POD': x['POD'], 'FAR': x['FAR'], 'CSI': x['CSI'], 'success_ratio': 1.0 - x['FAR']} for x in sorted(sc, key=lambda q: q['fold']) if x['FAR'] is not None and x['POD'] is not None]}
    nv = {}
    for k, v in val.items():
        nv[k] = [[q['year'], q['success_ratio']] for q in v['points']]
    skip = {'26.60|CLIM-WD'}
    assert max(val['26.60|CLIM-WD']['POD_range']) < 0.005
    C.setup()
    W = 6.5
    left, right, gut = (0.58, 0.34, 0.4)
    pw = (W - left - right - gut) / 2.0
    g = np.linspace(0.004, 1.0, 500)
    SRg, PODg = np.meshgrid(g, g)
    CSIg = 1.0 / (1.0 / SRg + 1.0 / PODg - 1.0)
    samp = []
    for b in BIAS:
        xe, ye = (1.0, b) if b <= 1.0 else (1.0 / b, 1.0)
        t_ = np.linspace(0.0, 1.0, 1500)
        samp.append(np.column_stack([t_ * xe, t_ * ye]))
    for c in CSI_LEV:
        x_ = np.linspace(0.004, 1.0, 4000)
        with np.errstate(divide='ignore', invalid='ignore'):
            p_ = 1.0 / (1.0 / c - 1.0 / x_ + 1.0)
        ok = np.isfinite(p_) & (p_ > 0) & (p_ <= 1.0)
        samp.append(np.column_stack([x_[ok], p_[ok]]))
    samp = np.vstack(samp)
    TOP_C = sorted([(dx, dy, 'right' if dx < 0 else 'left', 'top') for dx in (-2.0, -4.0, -6.0, -8.0, -10.0, 2.0, 4.0, 6.0, 8.0, 10.0) for dy in (-2.0, -5.0, -8.0)], key=lambda c: abs(c[0]) + abs(c[1]))
    SIDE_C = sorted([(dx, dy, 'right', 'bottom' if dy > 0 else 'top') for dx in (-2.0, -5.0, -8.0) for dy in (2.0, 4.0, 6.0, 8.0, 10.0, -2.0, -4.0, -6.0, -8.0, -10.0)], key=lambda c: abs(c[0]) + abs(c[1]))

    def place_bias(fig, ax, others, marks=()):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        fr = ax.get_window_extent(r)
        pad = fig.dpi / 72.0
        pts = ax.transData.transform(samp)
        obst = [o.get_window_extent(r) for o in others]
        obst += [p.get_window_extent(r) for p in ax.patches]
        from matplotlib.transforms import Bbox
        for x_, y_, s_ in marks:
            cx_, cy_ = ax.transData.transform((x_, y_))
            h_ = 0.5 * (s_ + MEW) * fig.dpi / 72.0
            obst.append(Bbox.from_extents(cx_ - h_, cy_ - h_, cx_ + h_, cy_ + h_))
        placed, rec = ([], [])
        for b in (2.0, 1.5, 1.0, 0.5):
            xe, ye = (1.0, b) if b <= 1.0 else (1.0 / b, 1.0)
            cands = SIDE_C if b <= 0.5 else TOP_C
            got = None
            for dx, dy, ha, va in cands:
                t = ax.annotate('%g' % b, xy=(xe, ye), xycoords='data', xytext=(dx, dy), textcoords='offset points', ha=ha, va=va, fontsize=S.FS_TICK, color=RAY, zorder=5)
                bb = t.get_window_extent(r)
                ex = bb.expanded(1.0, 1.0)
                ex = type(bb).from_extents(bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad)
                inside = ex.x0 >= fr.x0 and ex.x1 <= fr.x1 and (ex.y0 >= fr.y0) and (ex.y1 <= fr.y1)
                hit = np.any((pts[:, 0] >= ex.x0) & (pts[:, 0] <= ex.x1) & (pts[:, 1] >= ex.y0) & (pts[:, 1] <= ex.y1))
                over = any((ex.overlaps(o) for o in obst + placed))
                if inside and (not hit) and (not over):
                    got = (t, bb, (dx, dy, ha, va))
                    break
                t.remove()
            if got is None:
                for t in [q for q in ax.texts if q.get_gid() == 'biasval']:
                    t.remove()
                rec.append({'bias': b, 'placed': False})
                return (None, rec)
            got[0].set_gid('biasval')
            placed.append(got[1])
            rec.append({'bias': b, 'placed': True, 'offset_pt': [got[2][0], got[2][1]], 'ha': got[2][2], 'va': got[2][3]})
        return ([q for q in ax.texts if q.get_gid() == 'biasval'], rec)

    def draw(top, H):
        fig = S.plt.figure(figsize=(W, H))
        bottom = H - top - pw
        axes = []
        drawn = {}
        lay = {}
        for i, lev in enumerate(LEVELS):
            ax = fig.add_axes([(left + i * (pw + gut)) / W, bottom / H, pw / W, pw / H])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            cs = ax.contour(SRg, PODg, CSIg, levels=CSI_LEV, colors=ISO, linewidths=0.5, zorder=1)
            pos, lab_clear = ([], [])
            for c in CSI_LEV:
                best = None
                for pod_lab in np.linspace(0.96, 0.8, 33):
                    x = 1.0 / (1.0 / c - 1.0 / pod_lab + 1.0)
                    if not 0.03 < x < 0.97:
                        continue
                    dmin = min((abs(pod_lab - b * x) / np.hypot(1.0, b) for b in BIAS))
                    if best is None or dmin > best[0] + 1e-09:
                        best = (dmin, x, pod_lab)
                pos.append((best[1], best[2]))
                lab_clear.append(round(float(best[0]) * pw * 72.0, 1))
            drawn['csi_label_layout_%d' % i] = {'pos': [[round(float(a), 4), round(float(b), 4)] for a, b in pos], 'centre_to_nearest_ray_pt': lab_clear}
            lab = ax.clabel(cs, CSI_LEV, inline=True, fmt='%.1f', fontsize=S.FS_TICK, manual=pos, inline_spacing=2, colors=S.MUTED)
            for t in lab:
                t.set_rotation(0)
                t.set_bbox(dict(facecolor='white', edgecolor='none', pad=1.6))
                t.set_zorder(2)
                if t.get_text() == '0.5':
                    t.set_text('CSI 0.5')
            for b in BIAS:
                xe, ye = (1.0, b) if b <= 1.0 else (1.0 / b, 1.0)
                tt_ = np.linspace(0.0, 1.0, 600)
                xr_, yr_ = (tt_ * xe, tt_ * ye)
                fig.canvas.draw()
                rr_ = fig.canvas.get_renderer()
                q_ = ax.transData.transform(np.column_stack([xr_, yr_]))
                cut_ = np.zeros(len(q_), bool)
                for t in lab:
                    b_ = t.get_window_extent(rr_).padded(4.0 * fig.dpi / 72.0)
                    cut_ |= (q_[:, 0] > b_.x0) & (q_[:, 0] < b_.x1) & (q_[:, 1] > b_.y0) & (q_[:, 1] < b_.y1)
                yr_ = yr_.copy()
                yr_[cut_] = np.nan
                ax.plot(xr_, yr_, color=RAY, lw=0.6, ls=(0, (1.0, 1.6)), zorder=1)
            marks = []
            for m in ('CLIM-WD', 'GBT', 'P', 'M'):
                k = '%.2f|%s' % (lev, m)
                if k in skip:
                    continue
                col = S.MODEL_COLOUR[m]
                pts = val[k]['points']
                ctr = ax.transData.transform([[q['success_ratio'], q['POD']] for q in pts]) * (72.0 / fig.dpi)
                rec_ = []
                for a_, q in enumerate(pts):
                    near = [pts[b_]['year'] for b_ in range(a_ + 1, len(pts)) if np.hypot(*ctr[b_] - ctr[a_]) < NEST_FRAC * MS[m]]
                    ms_ = MS[m] * (NEST_SCALE if near else 1.0)
                    z_ = ZORD[m] + (0.01 * (1.0 - q['POD']) if m == 'GBT' else 0.0)
                    ln_, = ax.plot([q['success_ratio']], [q['POD']], ls='none', marker=S.MODEL_MARKER[m], ms=ms_, mfc=col, mec=S.INK, mew=MEW, zorder=z_)
                    ln_.set_gid('mark|%s|%d' % (k, q['year']))
                    ln_.set_path_effects([PE.Stroke(linewidth=MEW + 2.0 * CASE, foreground='white'), PE.Normal()])
                    marks.append((q['success_ratio'], q['POD'], ms_, m, ctr[a_]))
                    rec_.append({'year': q['year'], 'x_success_ratio': q['success_ratio'], 'y_POD': q['POD'], 'ms_pt': ms_, 'behind_nearly_coinciding': near, 'zorder': z_})
                drawn[k] = {'marker': S.MODEL_MARKER[m], 'face': col, 'edge': S.INK, 'edge_pt': MEW, 'marks': rec_}
            for a_ in range(len(marks)):
                for b_ in range(a_ + 1, len(marks)):
                    if marks[a_][3] != marks[b_][3]:
                        d_ = np.hypot(*marks[a_][4] - marks[b_][4])
                        assert d_ >= NEST_FRAC * max(MS.values()), (lev, marks[a_][:4], marks[b_][:4], d_)
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            if i == 0:
                ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
                ax.set_ylabel('Probability of detection (POD)', fontsize=S.FS_LABEL)
            else:
                ax.set_yticklabels([])
            C.frame(ax)
            bt, rec = place_bias(fig, ax, lab, [mk_[:3] for mk_ in marks])
            lay['bias_values_%d' % i] = rec
            if bt is None:
                for b in (2.0, 1.5, 1.0):
                    ax.annotate('%g' % b, xy=(1.0 / b, 1.0), xycoords='data', xytext=(0.0, 2.0), textcoords='offset points', ha='center', va='bottom', fontsize=S.FS_TICK, color=S.MUTED, annotation_clip=False)
                ax.text(1.012, 0.5, '0.5', ha='left', va='center', fontsize=S.FS_TICK, color=S.MUTED, clip_on=False)
                ax.annotate('Frequency bias', xy=(0.5, 1.0), xycoords='data', xytext=(-5.0, 2.0), textcoords='offset points', ha='right', va='bottom', fontsize=S.FS_TICK, color=S.MUTED, annotation_clip=False)
                C.tag(ax, 'ab'[i], 'Above %.2f ppb' % lev, gap_pt=C.title_gap_over(ax))
            else:
                C.tag(ax, 'ab'[i], 'Above %.2f ppb' % lev)
            axes.append(ax)
        C.block_title(fig, axes, 'Success ratio (1 ' + S.MINUS + ' FAR)', axis='x')
        from matplotlib.lines import Line2D
        from matplotlib.legend_handler import HandlerBase

        class _Mk:

            def __init__(self, m, ms):
                self.m, self.ms = (m, ms)

        class _MkHandler(HandlerBase):

            def create_artists(self, legend, h, xd, yd, w, ht, fs, trans):
                y = (ht - yd) / 2.0
                return [Line2D([-xd, -xd + w], [y, y], color='none', lw=0, transform=trans), Line2D([-xd + w - h.ms / 2.0 - 0.5], [y], ls='none', marker=S.MODEL_MARKER[h.m], ms=h.ms, mfc=S.MODEL_COLOUR[h.m], mec=S.INK, mew=MEW, transform=trans)]
        hd = [_Mk(m, MS[m]) for m in MODELS]
        legs = C.house_key(fig, axes, hd, [S.PLAIN[m] for m in MODELS], side='above', handleheight=0.9, handlelength=0.75, handler_map={_Mk: _MkHandler()})
        for lg_ in legs:
            for t_ in lg_.get_texts():
                for m_ in ('M', 'P'):
                    if t_.get_text().startswith(S.PLAIN[m_]):
                        t_.set_color(S.MODEL_TEXT[m_])
        return (fig, axes, drawn, lay)
    MARG = 0.1
    top0, H0 = (1.0, 5.0)
    fig, axes, drawn, lay = draw(top0, H0)
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    S.plt.close(fig)
    top = top0 - (H0 - tb.y1) + MARG
    H = round(tb.y1 - tb.y0 + 2 * MARG, 3)
    fig, axes, drawn, lay = draw(top, H)
    plotted = {'label': 'DESC, added after the plan was fixed', 'values': val, 'nv_success_ratio_points': nv, 'not_drawn': {'26.60|CLIM-WD': 'POD 0.00 in every year (tab_A_T2.tex); no marker drawn'}, 'marks_drawn': drawn, 'marks_rule': {'ms_pt': MS, 'edge': S.INK, 'edge_pt': MEW, 'zorder': ZORD, 'nest_frac': NEST_FRAC, 'nest_scale': NEST_SCALE, 'casing_pt': CASE}, 'csi_isolines': CSI_LEV, 'bias_rays': BIAS, 'printed_check': 'POD, FAR and CSI ranges and CMAQ events equal to tex\\tab_A_T2.tex at printed precision (asserted)', 'figure_in': [W, H], 'layout_bias_values': lay}
    plotted['fc_check'] = C.fc_check(fig)
    plotted['text_check'] = C.text_check(fig)
    fcc, tc = (plotted['fc_check'], plotted['text_check'])
    print('fc_check pass', fcc['pass'], 'case_pass', fcc['case_pass'], fcc['case'], fcc['overlaps'], fcc['outside'], fcc['small'], fcc['open_frames'])
    print('text_check pass', tc['pass'], tc['overlaps'], tc['within_0.08in_of_edge'], 'fig', tc['figure_in'])
    print('nv', {k: [[a, round(b, 4)] for a, b in v] for k, v in nv.items()})
    S.save(fig, STEM, plotted, [src, TAB])
    cap = "\\textbf{Exceedance skill of four models against modeled CMAQ daily mean surface NO$_2$ above 13.30~ppb (a) and 26.60~ppb (b), test days of the five held-out years, 2015 to 2019, contiguous US land cells.} Each marker is one model's probability of detection (POD, vertical) and success ratio (1 minus the false-alarm ratio, FAR; horizontal) in one held-out year. The residual U-Net is drawn as vermilion discs, the reference U-Net as blue squares, the gradient-boosted trees as dark gray triangles and the weekday climatology as light gray crosses. A model's years share one marker. Where two years of one model nearly coincide, in panel (a) the reference U-Net and in panel (b) the residual U-Net, the earlier year's marker is drawn larger behind the later one, so that both stay visible. Above 26.60~ppb the weekday climatology's POD rounded to 0.00 in every year and it is not drawn. Gray curves join points of equal critical success index (CSI, labeled 0.1 to 0.9), which is higher toward the upper right, the corner of full detection with no false alarms. Dotted rays from the origin mark frequency bias 0.5, 1, 1.5 and 2 (labeled at the frame); above the ray of 1 a model predicted more events than CMAQ, below it fewer. Table~\\ref{tab:exceed} gives each model's POD, FAR and CSI ranges over the years and the CMAQ events per level; above 63.83~ppb too few events were found to score. Networks at seed 0. The figure was not specified in advance; the values it shows were."
    alt = "Two performance diagrams of four models' detection and success ratio per held-out year, above 13.30 and 26.60 ppb."
    parts = ['Two square panels, (a) above 13.30 ppb and (b) above 26.60 ppb; horizontal axis success ratio (1 minus the false-alarm ratio, FAR), vertical axis probability of detection (POD), both 0 to 1. Gray curves of equal critical success index (CSI), labeled 0.1 to 0.9 and higher toward the upper right; dotted rays from the origin of frequency bias 0.5, 1, 1.5 and 2, labeled at the frame. Markers, one per model and held-out year at its success ratio and POD: residual U-Net, vermilion discs; reference U-Net, blue squares; gradient-boosted trees, dark gray triangles; weekday climatology, light gray crosses. Their POD, FAR and CSI ranges over the years, per model:']
    for lev in LEVELS:
        for m in MODELS:
            k = '%.2f|%s' % (lev, m)
            if k in skip:
                parts.append('Above %.2f ppb, %s: POD rounded to 0.00, not drawn.' % (lev, S.PLAIN[m]))
                continue
            parts.append('Above %.2f ppb, %s: POD %s; FAR %s; CSI %s.' % (lev, S.PLAIN[m], rng_txt(val[k]['POD_range']), rng_txt(val[k]['FAR_range']), rng_txt(val[k]['CSI_range'])))
            for q in drawn[k]['marks']:
                for y_ in q['behind_nearly_coinciding']:
                    parts.append('Above %.2f ppb, %s: the %d and %d markers nearly coincide; the %d marker is drawn larger behind.' % (lev, S.PLAIN[m], q['year'], y_, q['year']))
    C.write_text(STEM, cap, alt, '\n'.join(parts))
    print('words', len(cap.split()))
if __name__ == '__main__':
    main()
