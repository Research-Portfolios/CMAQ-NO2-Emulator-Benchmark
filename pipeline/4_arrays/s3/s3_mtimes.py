"""Prints the name, size and modification time of each file in the code's bytecode folders (read only)."""
import datetime
import glob
import os

def main():
    for d in ('.\\run\\code\\choi\\__pycache__', '.\\run\\code\\choi_dl\\__pycache__'):
        for p in sorted(glob.glob(os.path.join(d, '*'))):
            st = os.stat(p)
            print('%s %d %s' % (os.path.relpath(p, '.'), st.st_size, datetime.datetime.fromtimestamp(st.st_mtime).strftime('%Y-%m-%d %H:%M:%S')))
if __name__ == '__main__':
    main()
