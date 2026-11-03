"""Writes the list of files that the later stages copy (arrays, statistics, mask, city cells and emission arrays), each with its size and its sha256 taken again from the file and compared with the stage's records."""
import glob
import os
import sys
sys.path.insert(0, '/content/emul/run/code')
from s3 import s3common as S
from s3p2 import p2common as P

def entry(path, kind, want=None, **kw):
    if P.common.rel(path).lower().startswith(P.common.rel(P.ARRAYS_B).lower() + os.sep):
        raise ValueError('an earlier-version file: %s' % path)
    info = P.file_info(path)
    if want is not None and want != info['sha256']:
        raise ValueError('sha256 differs from the record: %s' % path)
    info['kind'] = kind
    info['sha256_checked_against_record'] = want is not None
    info.update(kw)
    return info

def main():
    S.common.set_temp()
    arrays = P.jload(os.path.join(P.CFG_P2, 'S3_ARRAYS.json'))
    files = []
    for y in P.YEARS:
        fl = arrays['years'][str(y)]['files']
        for k, kind in (('inputs_f16', 'inputs_f16'), ('inputs_f32', 'inputs_f32'), ('target', 'target'), ('inputs_c7', 'f6a_c7')):
            f = fl[k]
            extra = {'year': y, 'shape': f['shape'], 'dtype': f['dtype']}
            if f.get('partial'):
                extra.update(partial=True, unfilled_channels=f['unfilled_channels'], unfilled_indices=f['unfilled_indices'])
            files.append(entry(os.path.join(P.ROOT, f['path']), kind, f['sha256'], **extra))
    for nm, kind in (('mask_conus_land.npy', 'mask'), ('emis_true_2017.npy', 'f6_arm'), ('emis_true_2019.npy', 'f6_arm'), ('emis_true_2017_days.json', 'f6_arm'), ('emis_true_2019_days.json', 'f6_arm'), ('emis_clim_2017_2019.npy', 'f6_arm'), ('emis_clim_2017.npy', 'f6_arm'), ('emis_clim_2019.npy', 'f6_arm')):
        f = arrays['other'][nm]
        extra = {k: f[k] for k in ('shape', 'dtype') if k in f}
        files.append(entry(os.path.join(P.ROOT, f['path']), kind, f['sha256'], **extra))
    idx = P.jload(os.path.join(P.WORK_P2, 'norm_index.json'))
    for name in sorted(idx['files']):
        f = idx['files'][name]
        files.append(entry(os.path.join(P.ROOT, f['path']), 'norm', f['sha256']))
    gm = P.jload(os.path.join(P.WORK_P2, 'grid_mask.json'))
    files.append(entry(os.path.join(P.ROOT, gm['city_cells_file']['path']), 'city_cells', gm['city_cells_file']['sha256']))
    cfgs = []
    for nm in ('S3_CHANNELS.json', 'S3_ARRAYS.json', 'SPLITS.json', 'S3_D5_NOX.json'):
        cfgs.append(P.file_info(os.path.join(P.CFG_P2, nm)))
    others = sorted((os.path.basename(p) for p in glob.glob(os.path.join(P.ARRAYS_P2, '*')) if os.path.isfile(p)))
    listed = set((os.path.basename(f['path']) for f in files))
    doc = {'run': 'emul', 'stage': "the second pass, for the later stages' copy", 'made': P.now_local(), 'design': "the list of files that the later stages copy: the arrays, the target, the study design's per-fold statistics, the mask, the city cells, the two-year emission tests' arrays and the climatological-emission array", 'files': files, 'n_files': len(files), 'bytes': sum((f['bytes'] for f in files)), 'partial': [f['path'] for f in files if f.get('partial')], 'configs_by_outbox': cfgs, 'waiting': idx['waiting'], 'not_listed_in_arrays_p2': sorted((n for n in others if n not in listed))}
    info = P.write_json(os.path.join(P.CFG_P2, 'S3C_LIST.json'), doc)
    print('S3C_LIST.json: %d files, %d bytes, partial %d; %s sha16 %s' % (doc['n_files'], doc['bytes'], len(doc['partial']), info['path'], info['sha256'][:16]))
if __name__ == '__main__':
    P.run_main(main)
