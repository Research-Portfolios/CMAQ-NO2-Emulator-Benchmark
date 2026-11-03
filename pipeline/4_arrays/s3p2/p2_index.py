"""Writes the index files of the second pass (each channel's array, index, source, unit, storage and round-trip results; every array with its shape, type, size, sha256 and dates, with the emission slots that wait for the fill), each sha256 taken again from the file."""
import os
import sys
sys.path.insert(0, '/content/emul/run/code')
from s3 import s3_index as I
from s3 import s3common as S
from s3p2 import p2common as P
TRAINER_KEYS = ('test', 'n', 'n_finite_stored', 'p1_positive', 'n_used', 'p99_rel_error', 'max_rel_error')

def deciding(byyear):
    best = None
    for y, r in sorted(byyear.items()):
        if r['verdict'] == 'float32' and r['test'] != 'float32':
            return y
        p = r['p99_rel_error']
        if p is not None and (best is None or p > byyear[best]['p99_rel_error']):
            best = y
    return best or sorted(byyear)[0]

def main():
    rt = P.jload(os.path.join(P.WORK_P2, 'roundtrip_p2.json'))
    gm = P.jload(os.path.join(P.WORK_P2, 'grid_mask.json'))
    builds = {y: P.jload(os.path.join(P.WORK_P2, 'build_%d.json' % y)) for y in P.YEARS}
    c7 = {y: P.jload(os.path.join(P.WORK_P2, 'build_c7_%d.json' % y)) for y in P.YEARS}
    f16c = builds[2015]['f16_channels']
    f32c = builds[2015]['f32_channels']
    for y, b in builds.items():
        if b['f16_channels'] != f16c or b['f32_channels'] != f32c:
            raise ValueError('channel split differs in %d' % y)
    chans = []
    for i, (name, label, source, unit, default) in enumerate(P.CHANNELS):
        st = rt['storage'][name]
        arr = 'inputs_f32' if st == 'float32' else 'inputs_f16'
        idx = (f32c if st == 'float32' else f16c).index(name)
        aff = None
        if st == 'float16_affine':
            aff = {'stored': '(x - 50000) / 60000, then float16', 'offset': S.PRES_OFFSET, 'scale': S.PRES_SCALE, 'inverse': 'x = y * 60000 + 50000 (Pa)'}
        byyear = {y: r[name] for y, r in rt['years'].items() if name in r}
        dy = deciding(byyear)
        rr = byyear[dy]
        built = sorted((int(y) for y in byyear))
        chans.append({'name': name, 'index_c2': i, 'label': label, 'source': source, 'unit': unit, 'array': arr, 'index': idx, 'storage': st, 'dtype': 'float32' if st == 'float32' else 'float16', 'affine': aff, 'storage_why': rt['why'][name], 'roundtrip': dict({k: rr[k] for k in TRAINER_KEYS}, year=int(dy)), 'roundtrip_by_year': byyear, 'built_years': built, 'waiting_years': [y for y in P.YEARS if y not in built], 'baseline': name in I.PRIMARY, 'advanced': name in I.PRIMARY or name in I.LAGS, 'arm18': name in I.ARM18, 'log_scale_in_advanced': name in P.LOG_CHANNELS})
    unf = {str(y): builds[y]['unfilled_channels'] for y in P.YEARS if builds[y]['unfilled_channels']}
    doc = {'run': 'emul', 'stage': 'the second pass, the index', 'made': P.now_local(), 'design': "the study design's channel table (L-6 DLUSE and P-1 added; L-4 not built); the storage rule", 'order': [c['name'] for c in chans], 'baseline_channels': [c['name'] for c in chans if c['baseline']], 'advanced_channels': [c['name'] for c in chans if c['advanced']], 'arm18_channels': [c['name'] for c in chans if c['arm18']], 'f16_channels': f16c, 'f32_channels': f32c, 'channels': chans, 'target': {'name': 'T-1', 'array': 'target', 'dtype': 'float32', 'unit': 'ppb (ppbV)', 'source': 'RSIG cmaq.equates.conus.aconc.NO2 DAILY_MEAN (UTC days)'}, 'f6_emission_arrays': {'channel_order': ['E-1 NO', 'E-2 NO2', 'E-3 HONO', 'E-4 VOC_INV', 'P-1 NOx'], 'true': 'W\\arrays_p2\\emis_true_<y>.npy, day x 5 x 299 x 459, the days of emis_true_<y>_days.json', 'clim': 'W\\arrays_p2\\emis_clim_<y>.npy and emis_clim_2017_2019.npy, 12 x 7 x 5 x 299 x 459, [month - 1, weekday (Monday 0)]'}, 'f6a_emission_arrays': {'array': 'inputs_c7', 'channel_order': ['E-1 NO', 'E-2 NO2', 'E-3 HONO', 'E-4 VOC_INV', 'P-1 NOx'], 'file': "W\\arrays_p2\\inputs_c7_<y>.npy, day x 5 x 299 x 459, float32, each year's dates as its inputs arrays (S3_ARRAYS.json years.<y>.files.inputs_c7)", 'source': 'the emission climatology of 2017 and 2019 (emis_clim_2017_2019.npy) at [month - 1, weekday (Monday 0)] of each date', 'use': 'The climatological-emission folds take these five channels in place of the same channels of inputs_f32 on every date; every other channel from inputs_f16 and inputs_f32'}, 'partial_years': {'unfilled_channels': unf, 'fill': "float32 NaN at those channels' indices of inputs_f32 on every day; the fills stage fills them from each year's C-7a extracts (E-1 to E-4 from the merged files, P-1 from P1_NOX), the study design's checks counted, then the copy list; until then no main fold's statistics are written"}, 'counts': {'channels_built': len(chans), 'primary': sum((c['baseline'] for c in chans)), 'advanced': sum((c['advanced'] for c in chans)), 'arm18': sum((c['arm18'] for c in chans)), 'note': 'the study design gives 17, 23 and, with L-4, 19; L-4 is not built (only if the arm runs), so the arm as built has 18'}, 'roundtrip_rule': rt['rule'], 'p1_formation': "per sector cell (NO + NO2) + HONO of the local machine's float32 species cell grids in float64, summed over the six sectors and cast once to float32; the ONE DEFINITION's per-stack form (the extracts' P1_NOX) differs by at most one float32 step a sector-cell on the two sector-days measured (W\\arrays_p2\\work\\p1_check.json)", 'mask': {'path': gm['step2']['mask_file']['path'], 'sha256': gm['step2']['mask_file']['sha256'], 'cells': gm['step2']['cells_mask'], 'source': gm['step2']['file']}, 'city_cells': 'run\\config\\p2\\CITY_CELLS.json', 'dropped': {'L-4': 'ALBEDO not in METCRO2D; not built (built from GSW and RGRND only if the arm runs)'}, 'added': {'L-6': 'DLUSE from GRIDCRO2D (static), the category code as a number ((b), (d))', 'P-1': 'STACK NOx, float32'}}
    ch_info = P.write_json(os.path.join(P.CFG_P2, 'S3_CHANNELS.json'), doc)
    arrays = {'run': 'emul', 'stage': 'the second pass, the index', 'made': P.now_local(), 'years': {}, 'other': {}}
    total = 0
    for y in P.YEARS:
        days = [P.ymd(d) for d in P.days_of_year(y)]
        files = {}
        for k, nm, bk in (('inputs_f16', 'inputs_f16_%d.npy', 'f16'), ('inputs_f32', 'inputs_f32_%d.npy', 'f32'), ('target', 'target_%d.npy', 'target')):
            p = os.path.join(P.ARRAYS_P2, nm % y)
            files[k] = P.npy_info(p)
            if files[k]['sha256'] != builds[y]['files'][bk]['sha256']:
                raise ValueError('%s sha256 differs from the build' % p)
            total += files[k]['bytes']
        if builds[y]['unfilled_channels']:
            files['inputs_f32'].update({'partial': True, 'unfilled_channels': builds[y]['unfilled_channels'], 'unfilled_indices': [f32c.index(c) for c in builds[y]['unfilled_channels']], 'unfilled_fill': 'float32 NaN, every day; stage (8) fills them'})
        p = os.path.join(P.ARRAYS_P2, 'inputs_c7_%d.npy' % y)
        files['inputs_c7'] = P.npy_info(p)
        if files['inputs_c7']['sha256'] != c7[y]['file']['sha256']:
            raise ValueError('%s sha256 differs from its build' % p)
        total += files['inputs_c7']['bytes']
        arrays['years'][str(y)] = {'days': len(days), 'dates': days, 'files': files}
    for nm in ('mask_conus_land.npy', 'emis_clim_2017_2019.npy', 'emis_clim_2017.npy', 'emis_clim_2019.npy', 'emis_true_2017.npy', 'emis_true_2019.npy'):
        p = os.path.join(P.ARRAYS_P2, nm)
        arrays['other'][nm] = P.npy_info(p)
        total += arrays['other'][nm]['bytes']
    for nm in ('emis_true_2017_days.json', 'emis_true_2019_days.json'):
        arrays['other'][nm] = P.file_info(os.path.join(P.ARRAYS_P2, nm))
    arrays['total_bytes_npy'] = total
    arrays['channels_file'] = ch_info
    P.write_json(os.path.join(P.CFG_P2, 'S3_ARRAYS.json'), arrays)
    print('S3_CHANNELS.json and S3_ARRAYS.json written; npy bytes %d; primary %d, advanced %d, arm18 %d' % (total, doc['counts']['primary'], doc['counts']['advanced'], doc['counts']['arm18']))
if __name__ == '__main__':
    P.run_main(main)
