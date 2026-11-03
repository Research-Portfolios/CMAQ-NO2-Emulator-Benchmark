"""Lists the files that the second pass wrote, by group, with path, bytes and the first 16 hex digits of the sha256, as record lines."""
import glob
import os
import sys
sys.path.insert(0, '.\\run\\code')
from choi_dl import common
from s3p2 import p2common as P
WPRE = 'work\\'

def rows(pattern, recursive=False):
    out = []
    for p in sorted(glob.glob(pattern, recursive=recursive)):
        if os.path.isdir(p):
            continue
        out.append((common.rel(p), os.stat(p).st_size, P.S.sha256_file(p)[:16]))
    return out

def main():
    code_s3 = os.path.join(P.RUN, 'code', 's3')
    groups = [('run\\code\\s3\\ (changed by (b), and the backups)', [os.path.join(code_s3, n) for n in ('s3common.py', 's3_assemble.py', 's3_index.py', 's3common_Backup_PriorToDluseP2.py', 's3_assemble_Backup_PriorToDluseP2.py', 's3_index_Backup_PriorToDluseP2.py')]), ('run\\code\\s3p2\\ (new)', [os.path.join(P.RUN, 'code', 's3p2', '*.py')]), ('run\\config\\p2\\', [os.path.join(P.CFG_P2, '*')]), ('run\\logs\\ (the input-check record)', [P.D4_RECORD]), ('W\\arrays_p2\\', [os.path.join(P.ARRAYS_P2, '*')]), ('W\\arrays_p2\\work\\', [os.path.join(P.WORK_P2, '*')]), ('W\\natural_earth\\zips_p2\\ (with extracted\\)', [os.path.join(P.NE_ZIPS, '**', '*')]), ("W\\work/ (logs; stage\\ holds this report's staged text)", [os.path.join(P.TMP_P2, '*')])]
    L = []
    n = 0
    for title, pats in groups:
        L.append('  ' + title)
        seen = set()
        for pat in pats:
            for path, b, h in rows(pat, recursive='**' in pat):
                if path in seen:
                    continue
                seen.add(path)
                short = path.replace(WPRE, 'W\\')
                name = short if len(short) <= 45 else '...' + short[-42:]
                L.append('    %-45s %11d %s' % (name, b, h))
                n += 1
    nst = len([p for p in glob.glob(os.path.join(P.TMP_P2, 'stage', '*')) if os.path.isfile(p)])
    L += P.S.wrap('  ', "W\\work/ holds %d files, the staged text of this report's sections (appended by the earlier version's s3\\s3_rec.py --to), not hashed. This report run\\logs\\S3P2_ASCII.txt and the marker run\\logs\\S3P2_DONE_ASCII.txt are named, not hashed." % nst)
    bad = P.S.check_record_text('\n'.join(L))
    if bad:
        raise ValueError('; '.join(bad[:5]))
    with open(sys.argv[1], 'w', encoding='ascii', newline='\n') as f:
        f.write('\n'.join(L) + '\n')
    print('files %d; lines %d' % (n, len(L)))
if __name__ == '__main__':
    P.run_main(main)
