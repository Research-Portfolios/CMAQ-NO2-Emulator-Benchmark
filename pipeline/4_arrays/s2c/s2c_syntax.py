"""Compiles each named script without writing bytecode, prints OK or the error, and removes the stage's own bytecode folder if one is present."""
import os
import shutil
import sys
S2C = '.\\run\\code\\s2c'

def main(argv):
    bad = 0
    for p in argv:
        with open(p, 'rb') as f:
            src = f.read()
        try:
            src.decode('ascii')
            compile(src, p, 'exec')
            print('OK %s (%d bytes, ascii)' % (os.path.basename(p), len(src)))
        except (SyntaxError, UnicodeDecodeError) as e:
            bad += 1
            print('ERROR %s: %s' % (os.path.basename(p), e))
    pc = os.path.join(S2C, '__pycache__')
    if os.path.isdir(pc):
        names = sorted(os.listdir(pc))
        if all((n.startswith('s2c_') for n in names)):
            shutil.rmtree(pc)
            print('removed %s (%d files: %s)' % ('run\\code\\s2c\\__pycache__', len(names), ', '.join(names)))
        else:
            print('left %s: it holds files of another stage' % pc)
    return 1 if bad else 0
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
