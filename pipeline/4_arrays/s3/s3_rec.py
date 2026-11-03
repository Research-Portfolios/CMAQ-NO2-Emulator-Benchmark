"""Appends checked lines (7-bit ASCII, at most 78 characters, no at sign) from a text file written beforehand to the preprocessing log."""
import os
import sys
sys.path.insert(0, '.\\run\\code')
from s3 import s3common as S

def main():
    a = sys.argv[1:]
    src = a[0]
    to = S.REPORT
    if '--to' in a:
        to = a[a.index('--to') + 1]
    with open(src, 'rb') as f:
        b = f.read()
    if b'\r' in b:
        raise ValueError('CR in the staged text')
    text = b.decode('ascii')
    lines = text.rstrip('\n').split('\n')
    S.append_record(to, lines)
    print('appended %d lines; %s now %d bytes' % (len(lines), os.path.basename(to), os.stat(to).st_size))
if __name__ == '__main__':
    S.run_main(main)
