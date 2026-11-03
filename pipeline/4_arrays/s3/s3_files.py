"""Lists the files the preprocessing wrote, with path, bytes and the first 16 hex digits of the sha256, as record lines."""
import glob
import os
import sys
sys.path.insert(0, '.\\run\\code')
from s3 import s3common as S

def rows(pattern):
    out = []
    for p in sorted(glob.glob(pattern)):
        if os.path.isdir(p):
            continue
        out.append((S.common.rel(p), os.stat(p).st_size, S.sha256_file(p)[:16]))
    return out

def main():
    groups = [('run\\code\\s3\\', os.path.join(S.RUN, 'code', 's3', '*.py')), ("run\\config\\ (S3's)", os.path.join(S.CFG, 'S3_*.json')), ('', os.path.join(S.CFG, 'SPLITS.json')), ('', os.path.join(S.CFG, 'NORM_*.json')), ("run\\logs\\ (S3's input-check record)", os.path.join(S.LOGS, 'S3_D4_CHECKS_ASCII.txt')), ('W\\arrays\\', os.path.join(S.ARRAYS, '*')), ('W\\work\\s3\\', os.path.join(S.WORK_S3, '*')), ('W\\work/', os.path.join(S.TMP_S3, '*'))]
    L = ['', '13. FILES WRITTEN BY S3 (bytes by os.stat; sha16)']
    n = 0
    wpre = 'work\\'
    for title, pat in groups:
        rr = rows(pat)
        if title:
            L.append('  ' + title)
        for path, b, h in rr:
            short = path.replace(wpre, 'W\\')
            name = short if len(short) <= 44 else '...' + short[-41:]
            L.append('    %-44s %11d %s' % (name, b, h))
            n += 1
    nst = len([p for p in glob.glob(os.path.join(S.WORK_S3, 'stage', '*')) if os.path.isfile(p)])
    L += S.wrap('  ', "W\\work\\s3\\stage\\ holds %d files, the staged text of this report's sections (appended to it by s3_rec.py), not hashed. This report run\\logs\\S3_PREPROCESS_ASCII.txt and the marker run\\logs\\S3_DONE_ASCII.txt are named, not hashed (a file cannot hold its own hash)." % nst)
    p = os.path.join(S.WORK_S3, 'stage', 'part_files.txt')
    text = '\n'.join(L) + '\n'
    bad = S.check_record_text(text.rstrip('\n'))
    if bad:
        raise ValueError('; '.join(bad[:5]))
    with open(p, 'w', encoding='ascii', newline='\n') as f:
        f.write(text)
    print('files listed: %d -> %s' % (n, p))
if __name__ == '__main__':
    S.run_main(main)
