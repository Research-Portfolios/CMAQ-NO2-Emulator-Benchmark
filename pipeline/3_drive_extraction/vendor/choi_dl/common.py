"""The download code's helpers (places, time stamps, masking and line wrapping), as copied for the Drive extraction."""
import datetime
import os
import re
import tempfile
ROOT = '.'
W = os.path.join(ROOT, 'work')
RUN = os.path.join(ROOT, 'run')
W_TMP = os.path.join(W, 'tmp')
RAW = os.path.join(W, 'raw')
WORK = os.path.join(W, 'work')
EXTRACT = os.path.join(W, 'extract')
NATURAL_EARTH = os.path.join(W, 'natural_earth')
RUN_LOGS = os.path.join(RUN, 'logs')
RUN_PROV = os.path.join(RUN, 'provenance')
RUN_CONFIG = os.path.join(RUN, 'config')
_HOME = os.path.expanduser('~')
_NAME = os.path.basename(_HOME.rstrip('\\/'))

def set_temp():
    os.makedirs(W_TMP, exist_ok=True)
    os.environ['TMP'] = W_TMP
    os.environ['TEMP'] = W_TMP
    tempfile.tempdir = W_TMP

def rel(path):
    p = os.path.abspath(path)
    if p.lower().startswith(ROOT.lower() + '\\'):
        return p[len(ROOT) + 1:]
    return mask(p)

def mask(s):
    s = str(s)
    if _HOME and len(_HOME) > 3:
        s = re.sub(re.escape(_HOME), '<home>', s, flags=re.I)
        s = re.sub(re.escape(_HOME.replace('\\', '/')), '<home>', s, flags=re.I)
    if _NAME and len(_NAME) >= 3:
        s = re.sub(re.escape(_NAME), '<name>', s, flags=re.I)
    s = s.replace('@', '<at>')
    s = re.sub('%40', '<pct40>', s, flags=re.I)
    return s

def now_pair():
    u = datetime.datetime.now(datetime.timezone.utc)
    loc = u.astimezone()
    off = loc.utcoffset()
    hours = off.total_seconds() / 3600.0 if off is not None else 0.0
    lab = {-5.0: 'CDT', -6.0: 'CST'}.get(hours, loc.strftime('%z'))
    return (u.strftime('%Y-%m-%d %H:%M:%S UTC'), loc.strftime('%Y-%m-%d %H:%M:%S ') + lab)

def wrap(first_prefix, text, width=78, cont_prefix='    '):
    lines = []
    prefix, cur = (first_prefix, '')
    for word in str(text).split(' '):
        while True:
            room = width - len(prefix) - (len(cur) + 1 if cur else 0)
            if len(word) <= room:
                cur = cur + ' ' + word if cur else word
                break
            if cur:
                lines.append(prefix + cur)
                prefix, cur = (cont_prefix, '')
                continue
            cut = width - len(prefix)
            lines.append(prefix + word[:cut])
            prefix, word = (cont_prefix, word[cut:])
            if not word:
                break
    if cur or not lines:
        lines.append(prefix + cur)
    return lines
