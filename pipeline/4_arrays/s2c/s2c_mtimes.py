"""Prints the size and last-modified time of the downloader's progress, status and log files (read only)."""
import datetime
import os
ROOT = '.'
FILES = ('run\\logs\\DL_PROGRESS_ASCII.txt', 'run\\logs\\DL_STATUS.json', 'work/data_launcher.log')

def main():
    for f in FILES:
        st = os.stat(os.path.join(ROOT, f))
        t = datetime.datetime.fromtimestamp(st.st_mtime)
        print('%s: %d bytes, modified %s' % (f, st.st_size, t.strftime('%Y-%m-%d %H:%M:%S')))
if __name__ == '__main__':
    main()
