"""Writes record lines that summarize the statistics files: their sizes, their sha256 and a few of their values."""
import os
import sys
sys.path.insert(0, '.\\run\\code')
from s3p2 import p2common as P

def main():
    idx = P.jload(os.path.join(P.WORK_P2, 'norm_index.json'))
    L = ['file (in run\\config\\p2\\) bytes']
    rows = []
    for name in idx['written']:
        f = idx['files'][name]
        L.append('  %-44s %6d' % (name + '.json', f['bytes']))
        L.append('    sha256 %s' % f['sha256'])
        d = P.jload(os.path.join(P.ROOT, f['path']))
        rows.append((name, d))

    def short(name):
        s = name[5:]
        if s.startswith('f6a_fold'):
            return 'f6a ' + s[8:]
        s = s.replace('f6_train', 'f6 ').replace('_test20', '-')
        return s.replace('_', ' ')
    L.append('  Values (keys channels.<c> and target; f6 17-19 TRUE is NORM_f6_train2017_')
    L.append('  test2019_TRUE.json, and so on):')
    L.append('  file              M-1 min..max (stored)  T-1 min..max (ppb)')
    for name, d in rows:
        ch = d['channels']
        L.append('  %-17s %.6g..%.6g  %.6g..%.6g' % (short(name), ch['M-1']['min'], ch['M-1']['max'], d['target']['min'], d['target']['max']))
    L.append('  c of the log channels:')
    L.append('  file                   M-9      E-1      E-2      E-3      E-4      P-1')
    for name, d in rows:
        ch = d['channels']
        L.append('  %-17s' % short(name) + ''.join((' %8.3g' % ch[c]['c'] for c in P.LOG_CHANNELS)))
    bad = P.S.check_record_text('\n'.join(L))
    if bad:
        raise ValueError('; '.join(bad[:5]))
    with open(sys.argv[1], 'w', encoding='ascii', newline='\n') as f:
        f.write('\n'.join(L) + '\n')
    print('lines %d' % len(L))
if __name__ == '__main__':
    P.run_main(main)
