"""Closes the preprocessing log with an end line, checks the log and the input-check record for 7-bit ASCII, LF line ends and line length, and writes the one-line completion marker."""
import os
import sys
sys.path.insert(0, '.\\run\\code')
from s3 import s3common as S
MARKER = os.path.join(S.LOGS, 'S3_DONE_ASCII.txt')

def check(path):
    with open(path, 'rb') as f:
        b = f.read()
    probs = []
    if b'\r' in b:
        probs.append('CR')
    if any((c > 126 or (c < 32 and c != 10) for c in b)):
        probs.append('non-ASCII or control byte')
    t = b.decode('ascii', 'replace')
    lines = t.split('\n')
    if lines and lines[-1] == '':
        lines = lines[:-1]
    long_ = [i + 1 for i, x in enumerate(lines) if len(x) > 78]
    if long_:
        probs.append('lines over 78: %s' % long_[:10])
    if '@' in t:
        probs.append('at sign')
    trail = [i + 1 for i, x in enumerate(lines) if x != x.rstrip()]
    print('%s: %d lines, %d bytes, sha16 %s, longest %d; trailing spaces on %d lines; problems: %s' % (S.common.rel(path), len(lines), len(b), S.sha256_bytes(b)[:16], max((len(x) for x in lines)), len(trail), '; '.join(probs) or 'none'))
    return not probs

def main():
    a = sys.argv[1:]
    if a[0] == 'end':
        line = '%s END of S3 part 1: steps (1) to (9) done; the marker follows.' % S.now_local()
        S.append_record(S.REPORT, S.wrap('', line, '  '))
        ok = check(S.REPORT) and check(os.path.join(S.LOGS, 'S3_D4_CHECKS_ASCII.txt'))
        print('format ok: %s' % ok)
    elif a[0] == 'marker':
        status, text = (a[1], ' '.join(a[2:]))
        line = '%s status %s: %s' % (S.now_local(), status, text)
        if len(line) > 78 or '@' in line or any((ord(c) > 126 for c in line)):
            raise ValueError('marker line of %d characters' % len(line))
        with open(MARKER, 'w', encoding='ascii', newline='\n') as f:
            f.write(line + '\n')
        print('marker: %s' % line)
if __name__ == '__main__':
    S.run_main(main)
