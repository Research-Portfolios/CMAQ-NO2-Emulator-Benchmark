"""Builds, for every day of 2015 to 2019, the array of the five emission channels taken from the monthly weekday emission climatology of 2017 and 2019, with the per-day extremes and smallest positive values for its statistics."""
import os
import sys
import time
sys.path.insert(0, '/content/emul/run/code')
import numpy as np
from numpy.lib.format import open_memmap
from s3 import s3_build as B
from s3 import s3common as S
from s3p2 import p2common as P

def build_year(y, clim):
    t0 = time.time()
    start = P.now_local()
    mem0 = B.wait_memory()
    days = P.days_of_year(y)
    n = len(days)
    path = os.path.join(P.ARRAYS_P2, 'inputs_c7_%d.npy' % y)
    if os.path.exists(path):
        raise FileExistsError(path)
    if os.path.exists(path + '.partial'):
        os.remove(path + '.partial')
    shape = (n, len(P.EMIS)) + P.G
    mk = open_memmap(path + '.partial', mode='w+', dtype=np.float32, shape=shape)
    mk.flush()
    del mk
    daymin = np.full((n, len(P.EMIS)), np.nan)
    daymax = np.full((n, len(P.EMIS)), np.nan)
    isval = np.array([S.is_validation(d) for d in days])
    small = {c: B.SmallK(B.K_SMALL) for c in P.EMIS}
    i0 = 0
    for m in range(1, 13):
        B.wait_memory()
        md = [d for d in days if d.month == m]
        blk = np.stack([np.asarray(clim[m - 1, d.weekday()]) for d in md])
        for k, d in enumerate(md):
            i = i0 + k
            for j, c in enumerate(P.EMIS):
                a = blk[k, j]
                daymin[i, j] = float(a.min())
                daymax[i, j] = float(a.max())
                if not isval[i]:
                    small[c].add(a)
        mk = np.load(path + '.partial', mmap_mode='r+')
        if mk.shape != shape or mk.dtype != np.float32:
            raise ValueError('shape %s dtype %s' % (mk.shape, mk.dtype))
        mk[i0:i0 + len(md)] = blk
        mk.flush()
        del mk, blk
        i0 += len(md)
    if i0 != n:
        raise ValueError('days written %d of %d' % (i0, n))
    os.replace(path + '.partial', path)
    dates = [P.ymd(d) for d in days]
    np.savez(os.path.join(P.WORK_P2, 'daystats_c7_%d.npz' % y), daymin=daymin, daymax=daymax, isval=isval, dates=np.array(dates), channels=np.array(list(P.EMIS)))
    sk, npos = ({}, {})
    for c, s in small.items():
        v, npc = s.result()
        sk[c.replace('-', '')] = v
        npos[c] = npc
    np.savez(os.path.join(P.WORK_P2, 'smallk_c7_%d.npz' % y), **sk)
    info = {'year': y, 'start': start, 'end': P.now_local(), 'seconds': round(time.time() - t0, 1), 'days': n, 'channels': list(P.EMIS), 'source': 'W\\arrays_p2\\emis_clim_2017_2019.npy at [month - 1, weekday (Monday 0)] of each date', 'file': P.file_info(path), 'shape': list(shape), 'dtype': 'float32', 'memory_gb_start': round(mem0, 2), 'memory_gb_end': round(P.mem_gb(), 2), 'smallk_npos_nonvalidation': npos, 'k_small': B.K_SMALL, 'daystats': P.file_info(os.path.join(P.WORK_P2, 'daystats_c7_%d.npz' % y)), 'smallk': P.file_info(os.path.join(P.WORK_P2, 'smallk_c7_%d.npz' % y)), 'disk_free_gb_end': round(P.disk_free_gb(), 2)}
    P.write_json(os.path.join(P.WORK_P2, 'build_c7_%d.json' % y), info)
    return info

def main():
    a = sys.argv[1:]
    years = [int(x) for x in a[a.index('--years') + 1:]] if '--years' in a else list(P.YEARS)
    S.common.set_temp()
    P.ensure_dirs()
    clim = np.load(os.path.join(P.ARRAYS_P2, 'emis_clim_2017_2019.npy'), mmap_mode='r')
    if clim.shape != (12, 7, len(P.EMIS)) + P.G:
        raise ValueError('clim shape %s' % (clim.shape,))
    for y in years:
        if os.path.exists(os.path.join(P.WORK_P2, 'build_c7_%d.json' % y)):
            print('%d: done earlier, skipped' % y, flush=True)
            continue
        info = build_year(y, clim)
        print('%s %d done in %.0f s; memory %.2f GB; disk free %.2f GB' % (P.now_local(), y, info['seconds'], info['memory_gb_end'], info['disk_free_gb_end']), flush=True)
    print('%s all done' % P.now_local())
if __name__ == '__main__':
    P.run_main(main)
