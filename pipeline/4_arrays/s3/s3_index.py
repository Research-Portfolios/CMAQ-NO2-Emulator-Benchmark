"""Writes the index files of the preprocessing: each channel's array, index, source, unit, storage type and round-trip error, the channel sets of the models, and every array with its shape, type, size, sha256 and dates, each sha256 taken again from the file."""
import json
import os
import sys
sys.path.insert(0, '/content/emul/run/code')
import numpy as np
from s3 import s3common as S
PRIMARY = ('M-1', 'M-2', 'M-3', 'M-4', 'M-5', 'M-6', 'M-7', 'M-8', 'M-9', 'L-1', 'L-3', 'L-6', 'E-1', 'E-2', 'E-3', 'E-4', 'P-1')
ARM18 = ('M-1', 'M-2', 'M-3', 'M-4', 'M-5', 'M-6', 'M-7', 'M-8', 'M-9', 'L-1', 'L-2', 'L-3', 'L-5', 'E-1', 'E-2', 'E-3', 'E-4', 'P-1')
LAGS = ('X-1', 'X-2', 'X-3', 'X-4', 'X-5', 'X-6')

def jload(p):
    with open(p, 'r', encoding='ascii') as f:
        return json.load(f)

def npy_info(path):
    a = np.load(path, mmap_mode='r')
    info = S.file_info(path)
    info.update({'shape': list(a.shape), 'dtype': str(a.dtype)})
    del a
    return info

def main():
    rt = jload(os.path.join(S.WORK_S3, 'roundtrip.json'))
    gm = jload(os.path.join(S.WORK_S3, 'grid_mask.json'))
    builds = {y: jload(os.path.join(S.WORK_S3, 'build_%d.json' % y)) for y in S.YEARS}
    f16c = builds[2015]['f16_channels']
    f32c = builds[2015]['f32_channels']
    for y, b in builds.items():
        if b['f16_channels'] != f16c or b['f32_channels'] != f32c:
            raise ValueError('channel split differs in %d' % y)
    chans = []
    for r in rt['channels']:
        name = r['name']
        st = r['storage']
        arr = 'inputs_f32' if st == 'float32' else 'inputs_f16'
        idx = (f32c if st == 'float32' else f16c).index(name)
        aff = None
        if st == 'float16_affine':
            aff = {'stored': '(x - 50000) / 60000, then float16', 'offset': S.PRES_OFFSET, 'scale': S.PRES_SCALE, 'inverse': 'x = y * 60000 + 50000 (Pa)'}
        rr = r['roundtrip']
        chans.append({'name': name, 'index_c2': r['index_c2'], 'label': r['label'], 'source': S.CHANNELS[r['index_c2']][2], 'unit': r['unit'], 'array': arr, 'index': idx, 'storage': st, 'dtype': 'float32' if st == 'float32' else 'float16', 'affine': aff, 'storage_why': r['why'], 'roundtrip': {'test': r['test_storage'], 'n': rr['n'], 'n_finite_stored': r['n_finite_stored'], 'p1_positive': rr['p1_positive'], 'n_used': rr['n_used'], 'p99_rel_error': rr['p99_rel_error'], 'max_rel_error': rr['max_rel_error']}, 'baseline': name in PRIMARY, 'advanced': name in PRIMARY or name in LAGS, 'arm18': name in ARM18, 'log_scale_in_advanced': name in S.LOG_CHANNELS})
    doc = {'run': 'emul', 'stage': 'S3 step (6)', 'made': S.now_local(), 'design': "the study design's channel table (L-4 dropped: ALBEDO is not in METCRO2D); the storage rule", 'order': [c['name'] for c in chans], 'baseline_channels': [c['name'] for c in chans if c['baseline']], 'advanced_channels': [c['name'] for c in chans if c['advanced']], 'arm18_channels': [c['name'] for c in chans if c['arm18']], 'f16_channels': f16c, 'f32_channels': f32c, 'channels': chans, 'target': {'name': 'T-1', 'array': 'target', 'dtype': 'float32', 'unit': 'ppb (ppbV)', 'source': 'RSIG cmaq.equates.conus.aconc.NO2 DAILY_MEAN (UTC days)'}, 'f6_emission_arrays': {'channel_order': ['E-1 NO', 'E-2 NO2', 'E-3 HONO', 'E-4 VOC_INV'], 'true': 'W\\arrays\\emis_true_<y>.npy, day x 4 x 299 x 459, the days of emis_true_<y>_days.json', 'clim': 'W\\arrays\\emis_clim_<y>.npy and emis_clim_2017_2019.npy, 12 x 7 x 4 x 299 x 459, [month - 1, weekday (Monday 0)]'}, 'mask': {'path': gm['step2']['mask_file']['path'], 'sha256': gm['step2']['mask_file']['sha256'], 'cells': gm['step2']['cells_mask']}, 'dropped': {'L-4': 'ALBEDO not in METCRO2D'}, 'added': {'L-6': 'DLUSE from GRIDCRO2D (static), the category code as a number ((b), (d))'}}
    S.write_json(os.path.join(S.CFG, 'S3_CHANNELS.json'), doc)
    arrays = {'run': 'emul', 'stage': 'S3 step (6)', 'made': S.now_local(), 'years': {}, 'other': {}}
    total = 0
    for y in S.YEARS:
        days = [S.ymd(d) for d in S.days_of_year(y)]
        files = {}
        for k, nm in (('inputs_f16', 'inputs_f16_%d.npy'), ('inputs_f32', 'inputs_f32_%d.npy'), ('target', 'target_%d.npy')):
            p = os.path.join(S.ARRAYS, nm % y)
            files[k] = npy_info(p)
            bk = {'inputs_f16': 'f16', 'inputs_f32': 'f32', 'target': 'target'}[k]
            if files[k]['sha256'] != builds[y]['files'][bk]['sha256']:
                raise ValueError('%s sha256 differs from the build' % p)
            total += files[k]['bytes']
        arrays['years'][str(y)] = {'days': len(days), 'dates': days, 'files': files}
    for nm in ('mask_conus_land.npy', 'emis_clim_2017_2019.npy', 'emis_clim_2017.npy', 'emis_clim_2019.npy', 'emis_true_2017.npy', 'emis_true_2019.npy'):
        p = os.path.join(S.ARRAYS, nm)
        arrays['other'][nm] = npy_info(p)
        total += arrays['other'][nm]['bytes']
    for nm in ('emis_true_2017_days.json', 'emis_true_2019_days.json'):
        arrays['other'][nm] = S.file_info(os.path.join(S.ARRAYS, nm))
    arrays['total_bytes_npy'] = total
    arrays['channels_file'] = S.file_info(os.path.join(S.CFG, 'S3_CHANNELS.json'))
    S.write_json(os.path.join(S.CFG, 'S3_ARRAYS.json'), arrays)
    print('S3_CHANNELS.json and S3_ARRAYS.json written; npy bytes %d' % total)
if __name__ == '__main__':
    main()
