"""Closes the second pass's log with an end line, checks the log and the input-check record, and writes the one-line completion marker."""
import os
import sys
sys.path.insert(0, '.\\run\\code')
from s3 import s3_finish as F
from s3p2 import p2common as P

def main():
    a = sys.argv[1:]
    if a[0] in ('end', 'check'):
        if a[0] == 'end':
            line = '%s END of the second pass (attempt 1): steps (a) to (e) done; the marker follows.' % P.now_local()
            P.S.append_record(P.REPORT, P.S.wrap('', line, '  '))
        ok = F.check(P.REPORT) and F.check(P.D4_RECORD)
        print('format ok: %s' % ok)
        if not ok:
            sys.exit(1)
    elif a[0] == 'marker':
        status, text = (a[1], ' '.join(a[2:]))
        line = '%s status %s: %s' % (P.now_local(), status, text)
        if len(line) > 78 or '@' in line or any((ord(c) > 126 for c in line)):
            raise ValueError('marker line of %d characters' % len(line))
        with open(P.MARKER + '.partial', 'w', encoding='ascii', newline='\n') as f:
            f.write(line + '\n')
        os.replace(P.MARKER + '.partial', P.MARKER)
        print('marker: %s' % line)
if __name__ == '__main__':
    P.run_main(main)
