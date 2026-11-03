"""Prints, for each named file, its size and the first 16 hex digits of its sha256 and, for text files, whether it is 7-bit ASCII with LF line ends and its longest line (read only)."""
import hashlib
import os
import sys
ROOT = '.\\'
AT = chr(64)

def rel(p):
    a = os.path.abspath(p)
    if a.lower().startswith(ROOT.lower()):
        return a[len(ROOT):]
    return a

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def main(argv):
    text = False
    paths = []
    for a in argv:
        if a == '--text':
            text = True
        else:
            paths.append(a)
    for p in paths:
        st = os.stat(p)
        line = '%s bytes %d sha16 %s' % (rel(p), st.st_size, sha256_file(p)[:16])
        if text:
            with open(p, 'rb') as f:
                data = f.read()
            ascii_ok = all((c < 128 for c in data))
            cr = data.count(b'\r')
            lines = data.split(b'\n')
            longest = max((len(x) for x in lines)) if lines else 0
            ats = data.count(AT.encode('ascii'))
            line += ' ascii %s cr %d longest %d at %d lines %d' % ('yes' if ascii_ok else 'NO', cr, longest, ats, data.count(b'\n'))
        print(line)
if __name__ == '__main__':
    main(sys.argv[1:])
