"""Checks the text of a finished figure before it is saved: every visible text at least 7 pt, no overlapping text boxes, every box at least 0.08 in inside the canvas, and no line breaks; a failure stops the save."""
import itertools
import matplotlib.text as mtext
MIN_PT = 7.0
EDGE_IN = 0.08

def _texts(fig):
    ticks = {}
    for ax in fig.axes:
        for which, axis in (('x', ax.xaxis), ('y', ax.yaxis)):
            for lab in axis.get_ticklabels():
                ticks[id(lab)] = '%d%s' % (id(ax), which)
    out = []
    for t in fig.findobj(mtext.Text):
        if not t.get_visible():
            continue
        s = t.get_text()
        if not s or not s.strip():
            continue
        out.append((t, ticks.get(id(t), 't%d' % id(t))))
    return out

def check(fig, name):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    dpi = fig.dpi
    W, H = fig.get_size_inches()
    boxes = []
    problems = []
    for t, grp in _texts(fig):
        fs = t.get_fontsize()
        s = t.get_text()
        if fs < MIN_PT - 1e-09:
            problems.append('below 7 pt (%.2f): %r' % (fs, s[:40]))
        if '\n' in s:
            problems.append('line break in text: %r' % s[:40])
        bb = t.get_window_extent(r)
        x0, y0, x1, y1 = (bb.x0 / dpi, bb.y0 / dpi, bb.x1 / dpi, bb.y1 / dpi)
        if x0 < EDGE_IN or y0 < EDGE_IN or x1 > W - EDGE_IN or (y1 > H - EDGE_IN):
            problems.append('within 0.08 in of the canvas edge: %r (%.3f %.3f %.3f %.3f in)' % (s[:40], x0, y0, x1, y1))
        boxes.append((t, grp, (x0, y0, x1, y1)))
    for ax in fig.axes:
        ab = ax.get_window_extent(r)
        for lab, along in ((ax.xaxis.label, 'x'), (ax.yaxis.label, 'y')):
            if not lab.get_text().strip() or not lab.get_visible():
                continue
            lb = lab.get_window_extent(r)
            if along == 'x' and lb.width > ab.width + 0.5:
                problems.append('x label wider than its axis: %r' % lab.get_text()[:40])
            if along == 'y' and lb.height > ab.height + 0.5:
                problems.append('y label taller than its axis: %r' % lab.get_text()[:40])
    near = None
    n_overlap = 0
    for (ta, ga, a), (tb, gb, b) in itertools.combinations(boxes, 2):
        ix = min(a[2], b[2]) - max(a[0], b[0])
        iy = min(a[3], b[3]) - max(a[1], b[1])
        if ix > 0 and iy > 0:
            n_overlap += 1
            problems.append('overlap: %r and %r' % (ta.get_text()[:30], tb.get_text()[:30]))
            continue
        if ga == gb:
            continue
        gap = max(max(0.0, -ix), max(0.0, -iy)) * 72.0
        if near is None or gap < near[0]:
            near = (gap, ta.get_text()[:30], tb.get_text()[:30])
    res = {'figure': name, 'texts': len(boxes), 'min_font_pt': min((t.get_fontsize() for t, _, _ in boxes)), 'max_font_pt': max((t.get_fontsize() for t, _, _ in boxes)), 'overlaps': n_overlap, 'nearest_pair_pt': None if near is None else round(near[0], 2), 'nearest_pair': None if near is None else [near[1], near[2]], 'size_in': [float(W), float(H)], 'problems': problems, 'status': 'PASS' if not problems else 'FAIL'}
    print('text check %s: %s; %d texts; fonts %.1f-%.1f pt; overlaps %d; nearest pair %s pt %s' % (name, res['status'], res['texts'], res['min_font_pt'], res['max_font_pt'], n_overlap, res['nearest_pair_pt'], res['nearest_pair']))
    for p in problems:
        print('  PROBLEM:', p)
    if problems:
        raise SystemExit('text check FAIL: %s' % name)
    return res
