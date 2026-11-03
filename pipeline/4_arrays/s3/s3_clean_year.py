"""Removes the incomplete arrays of one year of the build, only when that year's build record is absent, and prints each removed file's name and size."""
import os
import sys
sys.path.insert(0, '.\\run\\code')
from s3 import s3common as S

def main():
    y = int(sys.argv[1])
    if os.path.exists(os.path.join(S.WORK_S3, 'build_%d.json' % y)):
        raise SystemExit('year %d is finished; nothing removed' % y)
    for base in ('inputs_f16_%d.npy', 'inputs_f32_%d.npy', 'target_%d.npy'):
        for suf in ('', '.partial'):
            p = os.path.join(S.ARRAYS, base % y + suf)
            if os.path.exists(p):
                n = os.stat(p).st_size
                os.remove(p)
                print('removed %s (%d bytes)' % (os.path.basename(p), n))
    print('done')
if __name__ == '__main__':
    main()
