"""Computes the disk space that the second pass still needs from its file counts and types, and stops when the free space is below that need plus 30 GB."""
import os
import shutil
import sys
sys.path.insert(0, '/content/emul/run/code')
from s3p2 import p2common as P
HDR = 128
LATER_GB = 1.5

def compute(n16, n32, skip_done=False):
    cells = P.NROWS * P.NCOLS
    days = {y: len(P.days_of_year(y)) for y in P.YEARS}
    items = []
    for y in P.YEARS:
        n = days[y]
        items.append(('inputs_f16_%d.npy' % y, n * n16 * cells * 2 + HDR))
        items.append(('inputs_f32_%d.npy' % y, n * n32 * cells * 4 + HDR))
        items.append(('target_%d.npy' % y, n * cells * 4 + HDR))
        items.append(('inputs_c7_%d.npy' % y, n * 5 * cells * 4 + HDR))
    for y in P.EMIS_YEARS:
        items.append(('emis_true_%d.npy' % y, days[y] * 5 * cells * 4 + HDR))
    for k in ('2017_2019', '2017', '2019'):
        items.append(('emis_clim_%s.npy' % k, 12 * 7 * 5 * cells * 4 + HDR))
    items.append(('mask_conus_land.npy', cells + HDR))
    nk = 5 + 5 * 2 + 5 * 5
    work = [('work: smallk files', nk * 2000000 * 4 + 10 ** 6), ('work: day statistics and records', 20 * 10 ** 6), ('Natural Earth zips and extracted files', 60 * 10 ** 6)]
    if skip_done:
        items = [(k, b) for k, b in items if not os.path.exists(os.path.join(P.ARRAYS_P2, k))]
    items += work
    need = sum((b for _k, b in items)) / 1000000000.0 + LATER_GB
    free = shutil.disk_usage(P.W).free / 1000000000.0
    return {'time': P.now_local(), 'n16': n16, 'n32': n32, 'items': [{'file': k, 'bytes': b} for k, b in items], 'later_stages_gb': LATER_GB, 'need_gb': round(need, 3), 'asked_gb': round(need + 30.0, 3), 'free_gb': round(free, 3), 'verdict': 'MET' if free >= need + 30.0 else 'STOP'}

def main():
    a = sys.argv[1:]
    tag = a[0]
    if '--n16' in a:
        n16 = int(a[a.index('--n16') + 1])
        n32 = int(a[a.index('--n32') + 1])
        split = 'given'
    else:
        n16 = sum((1 for c in P.CH_NAMES if P.DEFAULT_STORAGE[c] != 'float32'))
        n32 = sum((1 for c in P.CH_NAMES if P.DEFAULT_STORAGE[c] == 'float32'))
        split = 'defaults'
    out = compute(n16, n32, '--skip-done' in a)
    out['tag'] = tag
    out['split'] = split
    P.ensure_dirs()
    P.write_json(os.path.join(P.TMP_P2, 'disk_%s.json' % tag), out)
    print('%s disk %s: split %s (n16 %d, n32 %d); need %.3f GB (of it %.1f GB for the later stages); asked %.3f GB; free on the work disk %.3f GB; %s' % (out['time'], tag, split, n16, n32, out['need_gb'], LATER_GB, out['asked_gb'], out['free_gb'], out['verdict']))
    if out['verdict'] != 'MET':
        sys.exit(2)
if __name__ == '__main__':
    P.run_main(main)
