"""Parses each module of the preprocessing package without writing bytecode and checks that it is 7-bit ASCII with LF line ends."""
import ast
import glob
import os
import sys

def main():
    files = sys.argv[1:] or sorted(glob.glob(os.path.join('.\\run\\code\\s3', '*.py')))
    bad = 0
    for p in files:
        with open(p, 'rb') as f:
            b = f.read()
        msg = []
        if b'\r' in b:
            msg.append('CR')
        if any((c > 126 for c in b)):
            msg.append('non-ASCII')
        try:
            ast.parse(b.decode('ascii', 'replace'), p)
        except SyntaxError as e:
            msg.append('syntax: %s' % e)
        bad += bool(msg)
        print('%s %s' % (os.path.basename(p), '; '.join(msg) or 'ok'))
    print('files %d, with problems %d' % (len(files), bad))
if __name__ == '__main__':
    main()
