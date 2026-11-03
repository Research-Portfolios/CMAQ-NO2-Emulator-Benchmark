"""Checks the point and marine NOx channel on two sector-days: the per-stack sums of the cell-sum reducer against the extract's species grids and against the second pass's own formation, cell by cell, in float32 steps."""
import io
import os
import sys
import tarfile
sys.path.insert(0, '.\\run\\code')
import numpy as np
from choi import stacks
from s3p2 import p2_cellsums as CS
from s3p2 import p2_emis as PE
from s3p2 import p2common as P
TAR = os.path.join(P.W, 'raw', 'emis', 'stackgroups', 'model_ready_emis_2019_stackgroups_epicsoil_EQUATES_v1.0.tar')
CASES = (('ptegu', '20190701'), ('ptnonipm', '20190709'))
S2A = os.path.join(P.W, 'work', 's2a', 's3')

def groups_bytes(tf, sector):
    names = [m for m in tf.getmembers() if m.isfile()]
    want = [m for m in names if os.path.basename(m.name).startswith('stack_groups_%s_12US1_' % sector)]
    nc4 = [m for m in want if m.name.endswith('.nc4')]
    use = nc4 or [m for m in want if m.name.endswith('.ncf')]
    if len(use) != 1:
        raise ValueError('groups members for %s: %d' % (sector, len(use)))
    m = use[0]
    data = tf.extractfile(m).read()
    if len(data) != m.size:
        raise ValueError('short read')
    return (os.path.basename(m.name), data)

def ulps(a, b):
    ai = a.astype(np.float32).view(np.int32).astype(np.int64)
    bi = b.astype(np.float32).view(np.int32).astype(np.int64)
    return np.abs(ai - bi)

def main():
    P.common.set_temp()
    out = {'start': P.now_local(), 'tar': P.common.rel(TAR), 'tar_sha256': P.S.sha256_file(TAR), 'cases': []}
    with tarfile.open(TAR, 'r:') as tf:
        for sector, date in CASES:
            gname, gb = groups_bytes(tf, sector)
            gds, close = stacks.open_dataset(gb)
            try:
                g = stacks.groups_row_col(gds)
            finally:
                if close:
                    gds.close()
            ex = os.path.join(P.W, 'extract', 'pt_marine', sector, '2019', 'pt_%s_%s.npz' % (sector, date))
            with np.load(ex, allow_pickle=False) as z:
                base = int(z['base'])
                exg = {sp: z[sp].copy() for sp in ('NO', 'NO2', 'HONO')}
            case = {'sector': sector, 'date': date, 'groups_member': gname, 'groups_member_bytes': len(gb), 'n_stacks_groups': int(g['n_stacks']), 'base': base, 'extract': P.common.rel(ex), 'extract_sha256': P.S.sha256_file(ex), 'reads': {}}
            mine = PE.p1_of(exg).astype(np.float32)
            for read in ('selective', 'whole'):
                means = {}
                for sp in ('NO', 'NO2', 'HONO'):
                    p = os.path.join(S2A, 'pt_%s_%s_%s_stack_%s.npy' % (sector, date, read, sp))
                    means[sp] = np.load(p).astype(np.float64)
                grids, info = CS.cell_sums({'means': means}, g, base, True)
                one = grids['P1_NOX']
                r = {'stack_means_dtype': str(np.load(os.path.join(S2A, 'pt_%s_%s_%s_stack_NO.npy' % (sector, date, read))).dtype), 'n_stacks_means': int(means['NO'].size), 'n_summed': info['n_summed'], 'n_outside': info['n_outside'], 'species_grids_equal_extract': {sp: bool(np.array_equal(grids[sp], exg[sp])) for sp in ('NO', 'NO2', 'HONO')}}
                nz = one != 0
                d = np.abs(mine.astype(np.float64) - one.astype(np.float64))
                rel = np.zeros_like(d)
                rel[nz] = d[nz] / np.abs(one[nz].astype(np.float64))
                u = ulps(mine, one)
                r.update({'cells_nonzero_one_definition': int(nz.sum()), 'cells_nonzero_this_stage': int((mine != 0).sum()), 'cells_equal': int((mine == one).sum()), 'cells_differ': int((mine != one).sum()), 'max_abs_diff_mol_s': float(d.max()), 'max_rel_diff': float(rel.max()), 'max_float32_steps': int(u.max()), 'cells_1_step': int((u == 1).sum()), 'cells_2_or_more_steps': int((u >= 2).sum()), 'sum_one_definition_mol_s': float(one.astype(np.float64).sum()), 'sum_this_stage_mol_s': float(mine.astype(np.float64).sum())})
                case['reads'][read] = r
            out['cases'].append(case)
            print('%s %s: %s' % (sector, date, {k: (v['cells_differ'], v['max_float32_steps'], v['max_rel_diff'], v['species_grids_equal_extract']) for k, v in case['reads'].items()}), flush=True)
    out['end'] = P.now_local()
    info = P.write_json(os.path.join(P.WORK_P2, 'p1_check.json'), out)
    print('written %s sha16 %s' % (info['path'], info['sha256'][:16]))
if __name__ == '__main__':
    P.run_main(main)
