"""Lists further named files with their bytes and the first 16 hex digits of their sha256 as record lines."""
import os
import sys
sys.path.insert(0, '.\\run\\code')
from s3 import s3common as S

def main():
    wpre = 'work\\'
    L = []
    for rel in sys.argv[1:]:
        p = os.path.join(S.ROOT, rel)
        b = os.stat(p).st_size
        h = S.sha256_file(p)[:16]
        short = rel.replace(wpre, 'W\\')
        name = short if len(short) <= 44 else '...' + short[-41:]
        L.append('    %-44s %11d %s' % (name, b, h))
    out = os.path.join(S.WORK_S3, 'stage', 'part_files_added.txt')
    text = '\n'.join(L) + '\n'
    bad = S.check_record_text(text.rstrip('\n'))
    if bad:
        raise ValueError('; '.join(bad[:5]))
    with open(out, 'w', encoding='ascii', newline='\n') as f:
        f.write(text)
    print('listed %d -> %s' % (len(L), out))
if __name__ == '__main__':
    S.run_main(main)
