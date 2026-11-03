"""Writes the splits of the second pass by the first pass's rule and code, adds the climatological-emission folds with the same days as the main folds, and compares the day lists with the first pass's splits."""
import datetime
import os
import sys
sys.path.insert(0, '/content/emul/run/code')
from s3 import s3_splits as SP
from s3 import s3common as S
from s3p2 import p2common as P
LABEL = {2015: 'backward test', 2016: 'interior fold', 2017: 'interior fold', 2018: 'interior fold', 2019: "forward test; under C-7a the main set's inputs carry each year's own emissions, the climatological-emission folds' emission climatology"}

def main():
    inv = S.load_inventory()
    ext = {}
    for y in S.EMIS_YEARS:
        ds = inv['day_sets']['emis_merged'][str(y)]['dates']
        ext[y] = set((datetime.date(int(d[:4]), int(d[4:6]), int(d[6:])) for d in ds))
    out = {'run': 'emul', 'stage': 'the second pass, step (7)', 'made': P.now_local(), 'design': "the study design's splits, with the two-year emission tests and the climatological-emission folds", 'validation_rule': 'every day of each ISO week whose number is divisible by 5 (datetime.date.isocalendar)', 'index_rule': "index into the year's arrays: days since January 1 of that year", 'folds': {}, 'f6': {}, 'f6a': {}}
    allok = True
    for yh in S.YEARS:
        ytr = [y for y in S.YEARS if y != yh]
        tr = [d for y in ytr for d in S.days_of_year(y) if not S.is_validation(d)]
        va = [d for y in ytr for d in S.days_of_year(y) if S.is_validation(d)]
        he = S.days_of_year(yh)
        t = SP.test_split(tr, va, he, ytr, yh)
        allok = allok and t['pass']
        out['folds'][str(yh)] = {'held_out_year': yh, 'label': LABEL[yh], 'training_years': ytr, 'train': SP.pack(tr), 'validation': SP.pack(va), 'held_out': SP.pack(he), 'test': t}
        out['f6a'][str(yh)] = {'held_out_year': yh, 'same_days_as': 'folds.%d' % yh, 'label': 'The climatological-emission stand-in on fold %d' % yh, 'emission_source': "inputs_c7_<y>.npy for every training, validation and held-out date (the emission climatology of 2017 and 2019, emis_clim_2017_2019.npy at [month - 1, weekday]); every other channel from the year's inputs_f16 and inputs_f32", 'training_years': ytr, 'train': SP.pack(tr), 'validation': SP.pack(va), 'held_out': SP.pack(he), 'test': t}
    for ytr, yte in ((2017, 2019), (2019, 2017)):
        tr = [d for d in S.days_of_year(ytr) if d in ext[ytr] and (not S.is_validation(d))]
        va = [d for d in S.days_of_year(ytr) if d in ext[ytr] and S.is_validation(d)]
        he = [d for d in S.days_of_year(yte) if d in ext[yte]]
        t = SP.test_split(tr, va, he, [ytr], yte, universe=ext[ytr])
        allok = allok and t['pass']
        name = 'train%d_test%d' % (ytr, yte)
        out['f6'][name] = {'train_year': ytr, 'test_year': yte, 'arms': {'TRUE': 'emis_true_%d.npy for the training days and emis_true_%d.npy for the test days' % (ytr, yte), 'OTHER': 'each year the emission climatology of the other year: the training days take emis_clim_%d.npy, the test days emis_clim_%d.npy' % (yte, ytr), 'OWN': 'each year its own emission climatology: the training days take emis_clim_%d.npy, the test days emis_clim_%d.npy' % (ytr, yte)}, 'extracted_days_train_year': len(ext[ytr]), 'extracted_days_test_year': len(ext[yte]), 'train': SP.pack(tr), 'validation': SP.pack(va), 'held_out': SP.pack(he), 'test': t}
    out['all_tests_pass'] = allok
    out['summary'] = {k: {'train': v['train']['count'], 'validation': v['validation']['count'], 'held_out': v['held_out']['count']} for k, v in list(out['folds'].items()) + list(out['f6'].items())}
    info = P.write_json(os.path.join(P.CFG_P2, 'SPLITS.json'), out)
    print('all tests pass: %s; %s %d bytes sha16 %s' % (allok, info['path'], info['bytes'], info['sha256'][:16]))
    old_p = os.path.join(P.CFG_B, 'SPLITS.json')
    if not os.path.isfile(old_p):
        print("the earlier version's SPLITS.json absent (%s): the day lists are not compared (record only; chooses nothing)" % P.common.rel(old_p))
    else:
        old = P.jload(old_p)
        same = True
        for grp in ('folds', 'f6'):
            for k, v in out[grp].items():
                for part in ('train', 'validation', 'held_out'):
                    eq = v[part] == old[grp][k][part]
                    same = same and eq
                    if not eq:
                        print('DIFFERS %s %s %s' % (grp, k, part))
        print("day lists equal to the earlier version's SPLITS.json: %s" % same)
    for k, v in out['summary'].items():
        print(k, v)
if __name__ == '__main__':
    P.run_main(main)
