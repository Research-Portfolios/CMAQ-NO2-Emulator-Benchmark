"""Draws figure fig_ad1_design: the study design as a calendar of 2015 to 2019 with one row per held-out year, marking test, validation, held-back and buffer days, and checks the day counts against the report's."""
import datetime as DT
import json
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_style as S
import build_fig_common as C
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory, offset_copy
STEM = 'fig_ad1_design'
NUMBERS = os.path.join(S.ROOT, 'results', 'report', 'numbers.json')
TESTDAYS = os.path.join(S.ROOT, 'results', 'grid', 'testdays_cv.json')
COL = {'test': '#24374B', 'val': '#7F93A8', 'train': '#D5DEE8'}
SEAM_EDGE = S.FRAME_INK
SEAM_LW = 0.45
WEEK_TICK_PT = 2.5

def num(nb, key):
    return (nb[key]['value'], nb[key]['text'])

def main():
    with open(NUMBERS, encoding='utf-8') as f:
        nb = json.load(f)
    d0 = DT.date(2015, 1, 1)
    days = [d0 + DT.timedelta(days=k) for k in range((DT.date(2019, 12, 31) - d0).days + 1)]
    n_total, t_total = num(nb, 'f.days.total')
    assert len(days) == n_total, (len(days), n_total)
    test = {}
    counts = {}
    for y in S.YEARS:
        _doy, dates = C.test_days(y)
        test[y] = set(dates)
        v, t = num(nb, 'days.%d' % y)
        assert len(dates) == v, (y, len(dates), v)
        counts[y] = t
    key = [d.strftime('%Y%m%d') for d in days]
    is_test = np.array([key[i] in test[d.year] for i, d in enumerate(days)])
    withheld = ~is_test
    wk = [d.isocalendar()[:2] for d in days]
    by_week = {}
    for i, w in enumerate(wk):
        by_week.setdefault(w, []).append(i)
    held = np.zeros(len(days), bool)
    for w, idx in by_week.items():
        if len(idx) == 7 and all((withheld[i] for i in idx)):
            held[idx] = True
    buffer_ = withheld & ~held
    lb_v, lb_t = num(nb, 'lbdays.all')
    bf_v, bf_t = num(nb, 'rec.buffer')
    out_v, out_t = num(nb, 'f.days.out')
    assert int(held.sum()) == lb_v, (int(held.sum()), lb_v)
    assert int(buffer_.sum()) == bf_v, (int(buffer_.sum()), bf_v)
    assert int(withheld.sum()) == out_v
    blocks = []
    for i in range(len(days)):
        if held[i] and (i == 0 or not held[i - 1]):
            j = i
            while j + 1 < len(days) and held[j + 1]:
                j += 1
            blocks.append([i, j])
    nblk_v, _ = num(nb, 'rec.lbblocks')
    n_weeks = sum(((j - i + 1) // 7 for i, j in blocks))
    assert n_weeks == nblk_v, (n_weeks, nblk_v)
    per_year = [sum((1 for i, _j in blocks if days[i].year == y)) for y in S.YEARS]
    assert per_year[:4] == [3, 3, 3, 3], per_year
    val_day = np.array([d.isocalendar()[1] % 5 == 0 for d in days])
    role = {}
    for y in S.YEARS:
        r = np.full(len(days), '', dtype=object)
        for i, d in enumerate(days):
            if withheld[i]:
                r[i] = 'out'
            elif d.year == y:
                r[i] = 'test'
            elif val_day[i]:
                r[i] = 'val'
            else:
                r[i] = 'train'
        role[y] = r
    C.setup()
    W = 6.5
    left, right = (0.78, 0.62)
    ph = 2.32 - 0.36 - 0.52

    def draw(top, H):
        fig = S.plt.figure(figsize=(W, H))
        ax = fig.add_axes([left / W, (H - top - ph) / H, (W - left - right) / W, ph / H])
        N = len(days)
        rh, gap = (1.0, 0.32)
        ytop = len(S.YEARS) * rh + (len(S.YEARS) - 1) * gap
        pad = 0.16
        ax.set_xlim(0, N)
        ax.set_ylim(-pad, ytop + pad)
        centres = {}
        runs_drawn = {}
        for k, y in enumerate(S.YEARS):
            y1 = ytop - k * (rh + gap)
            y0 = y1 - rh
            centres[y] = (y0 + y1) / 2.0
            r = role[y]
            runs = []
            i = 0
            while i < N:
                j = i
                while j + 1 < N and r[j + 1] == r[i]:
                    j += 1
                runs.append([r[i], i, j + 1])
                i = j + 1
            for rl, a, b in runs:
                if rl == 'out':
                    continue
                ax.add_patch(Rectangle((a, y0), b - a, rh, facecolor=COL[rl], edgecolor='none', lw=0, zorder=1))
            runs_drawn[str(y)] = runs
        seams = []
        i = 0
        while i < N:
            if withheld[i]:
                j = i
                while j + 1 < N and withheld[j + 1]:
                    j += 1
                seams.append([i, j + 1])
                i = j + 1
            else:
                i += 1
        for a, b in seams:
            ax.add_patch(Rectangle((a, -pad), b - a, ytop + 2 * pad, facecolor='white', edgecolor=SEAM_EDGE, lw=SEAM_LW, zorder=3, clip_on=True))
        C.frame(ax)
        yt = [centres[y] for y in S.YEARS]
        ax.set_yticks(yt)
        ax.set_yticklabels([str(y) for y in S.YEARS])
        ax.set_ylabel('Held-out year', fontsize=S.FS_LABEL)
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(yt)
        ax2.set_yticklabels([counts[y] for y in S.YEARS])
        ax2.set_ylabel('Test days', fontsize=S.FS_LABEL, rotation=90)
        C.frame(ax2)
        bounds = [(DT.date(y, 1, 1) - d0).days for y in S.YEARS] + [N]
        ax.set_xticks(bounds)
        ax.set_xticklabels([''] * len(bounds))
        ax.tick_params(axis='x', which='major', length=5.5)
        minor = []
        for y in S.YEARS:
            for w in (13, 26, 39):
                minor.append((DT.date.fromisocalendar(y, w, 1) - d0).days + 3.5)
        ax.set_xticks(minor, minor=True)
        ax.tick_params(axis='x', which='minor', length=WEEK_TICK_PT, width=0.5, color=S.INK2)
        mids = []
        for y in S.YEARS:
            a = (DT.date(y, 1, 1) - d0).days
            b = (DT.date(y, 12, 31) - d0).days + 1
            mids.append((a + b) / 2.0)
        yax = fig.add_axes(ax.get_position())
        yax.set_gid('noframe')
        for s_ in yax.spines.values():
            s_.set_visible(False)
        yax.patch.set_visible(False)
        yax.set_xlim(ax.get_xlim())
        yax.set_yticks([])
        yax.set_xticks(mids)
        yax.set_xticklabels([str(y) for y in S.YEARS])
        yax.tick_params(axis='x', length=0, pad=WEEK_TICK_PT + S.TICK_PAD_PT)
        yax.set_xlabel('Calendar year', fontsize=S.FS_LABEL)
        hd = [Rectangle((0, 0), 1, 1, facecolor=COL['test'], edgecolor='none'), Rectangle((0, 0), 1, 1, facecolor=COL['train'], edgecolor='none'), Rectangle((0, 0), 1, 1, facecolor=COL['val'], edgecolor='none'), Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=SEAM_EDGE, lw=SEAM_LW)]
        labs = ['Test days of the held-out year', 'Training days', 'Validation days', 'Held-back and buffer days']
        legs = C.house_key(fig, [ax], hd, labs, side='above', handleheight=0.9)
        return (fig, ax, legs, centres, runs_drawn, seams, bounds, labs)
    MARG = 0.1
    top0, H0 = (0.8, 3.5)
    fig, ax, legs, centres, runs_drawn, seams, bounds, labs = draw(top0, H0)
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    S.plt.close(fig)
    top = top0 - (H0 - tb.y1) + MARG
    H = round(tb.y1 - tb.y0 + 2 * MARG, 3)
    fig, ax, legs, centres, runs_drawn, seams, bounds, labs = draw(top, H)
    N = len(days)
    plotted = {'label': 'REC, added after the plan was fixed', 'days_from': '2015-01-01', 'n_days': N, 'test_day_counts_printed': counts, 'held_back_days': int(held.sum()), 'buffer_days': int(buffer_.sum()), 'withheld_days': int(withheld.sum()), 'held_back_weeks': n_weeks, 'held_back_blocks_day_index': blocks, 'withheld_runs_day_index': seams, 'buffer_day_dates': [days[i].isoformat() for i in np.where(buffer_)[0]], 'validation_rule': 'ISO week number divisible by 5, training years, less held-back and buffer days', 'role_runs_by_fold': runs_drawn, 'role_counts_by_fold': {str(y): {rl: int((role[y] == rl).sum()) for rl in ('test', 'train', 'val', 'out')} for y in S.YEARS}, 'x_year_bounds_day_index': bounds, 'x_minor_iso_weeks': [13, 26, 39], 'colours': dict(COL, seam_edge=SEAM_EDGE), 'key': labs, 'figure_in': [W, H]}
    for y in S.YEARS:
        c = plotted['role_counts_by_fold'][str(y)]
        assert c['test'] == int(nb['days.%d' % y]['value'])
    plotted['fc_check'] = C.fc_check(fig)
    plotted['text_check'] = C.text_check(fig)
    fcc, tc = (plotted['fc_check'], plotted['text_check'])
    print('fc_check pass', fcc['pass'], 'case_pass', fcc['case_pass'], fcc['case'], fcc['overlaps'], fcc['outside'], fcc['small'], fcc['open_frames'])
    print('text_check pass', tc['pass'], tc['overlaps'], tc['within_0.08in_of_edge'], 'fig', tc['figure_in'])
    print('counts', plotted['role_counts_by_fold'])
    S.save(fig, STEM, plotted, [NUMBERS, TESTDAYS])
    cap = '\\textbf{Study design of the emulators of modeled daily mean surface NO$_2$ on the contiguous US grid: the role of each day of 2015 to 2019 in each of the five folds.} Each year (one row) was held out in turn and scored on its test days (dark); the networks of that fold were trained on the other four years (pale). Validation days (mid tone), used only to choose the checkpoint and stop training, were every day of each ISO week whose number is divisible by five in the training years. White columns with a thin outline are the %s held-back days in %s whole weeks and the %s buffer days beside them, set aside before any training and removed from every fold. The numbers at right give the test days of each held-out year. Long ticks mark the start of each year; short ticks, ISO weeks 13, 26 and 39.' % (lb_t, nb['rec.lbblocks']['text'], bf_t)
    alt = 'Calendar of five fold rows, 2015 to 2019, showing test, training, validation and held-back days.'
    longd = 'Five horizontal strips, one per held-out year 2015 to 2019 (left axis, Held-out year), across the days of 2015 to 2019 (bottom axis, Calendar year). In each strip the held-out year is dark (test days); the other four years are pale (training days) with mid-tone stripes every fifth ISO week (validation days). Narrow white columns with a thin outline run through all five strips at the same dates: the held-back and buffer days, three blocks a year in 2015 to 2018 and the last weeks of 2019. Right axis, Test days: %s.' % ', '.join(('%d %s' % (y, counts[y]) for y in S.YEARS))
    C.write_text(STEM, cap, alt, longd)
    print('words', len(cap.split()))
if __name__ == '__main__':
    main()
