"""Refreshes the record folder's copies of each source folder's provenance records, replacing a differing copy only where a backup of it exists, and compares every pair again."""
import glob
import os
import sys
CODE = '.\\run\\code'
if CODE not in sys.path:
    sys.path.insert(0, CODE)
from choi_dl import common, prov
FOLDERS = (('rsig', 'raw\\rsig'), ('rsig_work', 'work\\rsig'), ('s3_mcip', 'raw\\mcip'), ('s3_s3meta', 'raw\\s3meta'), ('s3_stackgroups', 'raw\\emis\\stackgroups'), ('s3_emis_merged', 'extract\\emis_merged'), ('s3_pt_marine', 'extract\\pt_marine'), ('ne_natural_earth', 'natural_earth'))
FILES = ('PROVENANCE_ASCII.txt', 'PROVENANCE.jsonl')

def state(p):
    if not os.path.exists(p):
        return None
    return (os.stat(p).st_size, prov.sha256_file(p))

def main():
    common.set_temp()
    left = 0
    for name, folder in FOLDERS:
        src_dir = os.path.join(common.W, folder)
        ok_to_copy = True
        for fn in FILES:
            src = os.path.join(src_dir, fn)
            dst = os.path.join(common.RUN_PROV, '%s_%s' % (name, fn))
            a, b = (state(src), state(dst))
            if a != b and b is not None:
                stem, ext = os.path.splitext(dst)
                bk = glob.glob(glob.escape(stem) + '_Backup_PriorToS2c*' + ext)
                if not bk:
                    ok_to_copy = False
                    print('LEFT %s: it differs from its source and has no backup of this stage' % common.rel(dst))
        if not ok_to_copy:
            left += 1
            continue
        out = prov.copy_to_run(src_dir, name)
        for p in out:
            fn = os.path.basename(p)[len(name) + 1:]
            a = state(os.path.join(src_dir, fn))
            b = state(p)
            print('%s %s: %d bytes, sha16 %s' % ('EQUAL' if a == b else 'DIFFER', common.rel(p), b[0], b[1][:16]))
    return 1 if left else 0
if __name__ == '__main__':
    sys.exit(main())
