"""The labels and notes that word each F-12 weekday-contrast row by model; they fix the wording of the rows only."""
import re
ADD20_FILE = 'the label rule'
ADD20_LINE_RANGE = (170, 180)
ADD20_LINES = ('The labels of F-12:', 'emission-driven reading: P, A1 and', "A2, A4 (no CLIM-WD in offset or input); possibly carried by CLIM-WD's", 'weekday factor: A3, A3BL, M, M-C7, PIX-R (CLIM-WD their offset and C', "channel), A3OFF (CLIM-WD its offset only), GBT and GBT-RIDGE (and GBT's", "weekday feature) and MET-REG; under C-7a every F-12 row reads 'one", "emission-driven variation among several'. F-12's TEST checks each label", "against the frozen models' calendar inputs and", 'against the study design for the C', 'offset and channel. The labels are data in the evaluation code', 'fixed before any F-12 reading.')
EMISSION = 'emission-driven reading'
CLIMWD = "possibly carried by CLIM-WD's weekday factor"
C7A_ROW = 'one emission-driven variation among several'
NOT_LABELLED = 'not labelled'
LABELS = {'P': (EMISSION, 'no CLIM-WD in offset or input'), 'A1': (EMISSION, 'no CLIM-WD in offset or input'), 'A2': (EMISSION, 'no CLIM-WD in offset or input'), 'A4': (EMISSION, 'no CLIM-WD in offset or input'), 'A3': (CLIMWD, 'CLIM-WD their offset and C channel'), 'A3BL': (CLIMWD, 'CLIM-WD their offset and C channel'), 'M': (CLIMWD, 'CLIM-WD their offset and C channel'), 'M-C7': (CLIMWD, 'CLIM-WD their offset and C channel'), 'PIX-R': (CLIMWD, 'CLIM-WD their offset and C channel'), 'A3OFF': (CLIMWD, 'CLIM-WD its offset only'), 'GBT': (CLIMWD, "and GBT's weekday feature"), 'GBT-RIDGE': (CLIMWD, "and GBT's weekday feature"), 'MET-REG': (CLIMWD, '')}

def _names(part):
    s = ' '.join(part.split()).strip().strip(',').strip()
    if s.startswith('and '):
        s = s[4:]
    s = s.replace(' and ', ', ')
    return [x.strip() for x in s.split(',') if x.strip()]

def parse_add20(text):
    t = ' '.join(text.split())
    k1 = 'emission-driven reading: '
    k2 = ' (no CLIM-WD in offset or input);'
    k3 = "possibly carried by CLIM-WD's weekday factor: "
    k4 = "; under C-7a every F-12 row reads '"
    a = t.index(k1) + len(k1)
    b = t.index(k2, a)
    out = {}
    for n in _names(t[a:b]):
        out[n] = (EMISSION, k2.strip(' ;()'))
    c = t.index(k3, b) + len(k3)
    d = t.index(k4, c)
    rest = t[c:d]
    pos = 0
    for m in re.finditer('\\(([^()]*)\\)', rest):
        for n in _names(rest[pos:m.start()]):
            out[n] = (CLIMWD, m.group(1))
        pos = m.end()
    for n in _names(rest[pos:]):
        out[n] = (CLIMWD, '')
    e = d + len(k4)
    row = t[e:t.index("'", e)]
    return (out, row)

def wording(model, c7a_applies):
    lab, note = LABELS.get(model, (NOT_LABELLED, ''))
    words = lab + (' (%s)' % note if note else '')
    if c7a_applies:
        words += '; ' + C7A_ROW
    return {'model': model, 'label': lab, 'note': note, 'c7a': C7A_ROW if c7a_applies else None, 'words': words, 'source': '%s lines %d-%d' % (ADD20_FILE, ADD20_LINE_RANGE[0], ADD20_LINE_RANGE[1])}
