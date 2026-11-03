"""Helpers of the figure scripts: panel tags, number formats, the writers of the caption and alternative-text files, and the figure texts in reading order."""
import os
import re
import build_style as S
GAP_TAG_PT = 3.0
EVAL_CODE = os.path.join(S.ROOT, 'pipeline', '8_evaluation', 'eval')

def code_const(name, module='metrics.py'):
    p = os.path.join(EVAL_CODE, module)
    with open(p, encoding='utf-8') as f:
        for n, line in enumerate(f, 1):
            mm = re.match('^%s\\s*=\\s*(.+?)\\s*(#.*)?$' % re.escape(name), line)
            if mm:
                import ast
                return (ast.literal_eval(mm.group(1)), '%s line %d' % (module, n))
    raise SystemExit('constant %s not found in %s' % (name, module))

def panel_name(fig, ax, letter, name, y=1.0, pad_pt=4.0):
    w_in = ax.get_position().width * fig.get_size_inches()[0]
    h_in = ax.get_position().height * fig.get_size_inches()[1]
    yy = y + pad_pt / 72.0 / h_in
    xt = -(GAP_TAG_PT / 72.0) / w_in
    ax.text(xt, yy, '(%s)' % letter, transform=ax.transAxes, ha='right', va='baseline', fontsize=8, fontweight='bold', color=S.INK)
    ax.text(0.0, yy, name, transform=ax.transAxes, ha='left', va='baseline', fontsize=8, color=S.INK)

def pct(level):
    v = round(100.0 * float(level), 2)
    s = ('%.2f' % v).rstrip('0').rstrip('.')
    return s

def num(v, nd=3):
    s = '%.*f' % (nd, float(v))
    if s.startswith('-'):
        s = '$-$' + s[1:]
    return s

def tick(v, nd=1):
    s = '%.*f' % (nd, float(v))
    if float(s) == 0.0:
        s = s.lstrip('-')
    return s.replace('-', '−')

def num_plain(v, nd=3):
    s = '%.*f' % (nd, float(v))
    return s.replace('-', 'minus ') if s.startswith('-') else s

def _ascii(text):
    bad = [c for c in text if ord(c) > 126 or (ord(c) < 32 and c != '\n')]
    if bad:
        raise SystemExit('non-ASCII in output text: %r' % bad[:5])
    return text

def write_caption(stem, text):
    text = re.sub('\\s+', ' ', text).strip() + '\n'
    p = os.path.join(S.FIG, stem + '_caption.tex')
    with open(p, 'w', encoding='ascii', newline='\n') as f:
        f.write(_ascii(text))
    words = len(re.sub('\\\\[a-zA-Z]+|[{}$]', ' ', text).split())
    print('caption %s: %d words' % (stem, words))
    return (p, words)

def write_alt(stem, alt, long_desc):
    alt = re.sub('\\s+', ' ', alt).strip()
    if len(alt) > 125:
        raise SystemExit('alt text over 125 characters (%d)' % len(alt))
    long_desc = re.sub('[ \\t]+', ' ', long_desc).strip()
    p = os.path.join(S.FIG, stem + '_alt.txt')
    with open(p, 'w', encoding='ascii', newline='\n') as f:
        f.write(_ascii(alt + '\n\n' + long_desc + '\n'))
    print('alt %s: %d characters' % (stem, len(alt)))
    return (p, len(alt))

def image_texts(fig):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    out = []
    for t in fig.findobj(lambda a: hasattr(a, 'get_text') and hasattr(a, 'get_window_extent')):
        try:
            s = t.get_text()
        except Exception:
            continue
        if not isinstance(s, str) or not s.strip() or (not t.get_visible()):
            continue
        bb = t.get_window_extent(r)
        out.append((-round(bb.y1 / fig.dpi / 0.04), bb.x0, s))
    out.sort()
    seen = []
    for _, _, s in out:
        seen.append(s)
    return seen

def check_long_holds(long_desc, texts):
    missing = []
    for s in texts:
        k = s.replace('−', '-')
        if k not in long_desc:
            missing.append(k)
    if missing:
        raise SystemExit('long description misses image text: %r' % missing[:8])
    print('long description holds all %d image texts' % len(texts))
