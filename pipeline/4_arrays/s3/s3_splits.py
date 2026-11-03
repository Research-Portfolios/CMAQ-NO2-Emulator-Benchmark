"""Writes the splits: for each held-out year 2015 to 2019 the training days (the other four years less the validation days, every day of each ISO week whose number is divisible by five), the validation days and the held-out days, and the two-year emission tests, testing that the lists are disjoint and complete."""
import datetime
import os
import sys
sys.path.insert(0, '/content/emul/run/code')
from s3 import s3common as S
LABEL = {2015: 'backward test', 2016: 'interior fold', 2017: 'interior fold', 2018: 'interior fold', 2019: 'forward test (forward in its meteorology and its target only, under the emission climatology)'}

def idx_of(d):
    return (d - datetime.date(d.year, 1, 1)).days

def pack(days):
    out = {'dates': [S.ymd(d) for d in days], 'index_by_year': {}}
    for d in days:
        out['index_by_year'].setdefault(str(d.year), []).append(idx_of(d))
    out['count'] = len(days)
    out['count_by_year'] = {y: len(v) for y, v in out['index_by_year'].items()}
    return out

def test_split(train, val, held, years_train, year_held, universe=None):
    t, v, h = (set(train), set(val), set(held))
    res = {'heldout_in_train': len(h & t), 'heldout_in_val': len(h & v), 'train_val_overlap': len(t & v), 'val_not_iso5': sum((1 for d in v if not S.is_validation(d))), 'train_iso5': sum((1 for d in t if S.is_validation(d))), 'train_or_val_outside_years': sum((1 for d in t | v if d.year not in years_train)), 'heldout_outside_year': sum((1 for d in h if d.year != year_held))}
    allt = set((d for y in years_train for d in S.days_of_year(y)))
    if universe is not None:
        allt = set((d for d in allt if d in universe))
    res['train_plus_val_cover_years'] = t | v == allt
    res['pass'] = all((res[k] == 0 for k in res if k != 'pass' and k != 'train_plus_val_cover_years')) and res['train_plus_val_cover_years']
    return res

def main():
    inv = S.load_inventory()
    ext = {}
    for y in S.EMIS_YEARS:
        ds = inv['day_sets']['emis_merged'][str(y)]['dates']
        ext[y] = set((datetime.date(int(d[:4]), int(d[4:6]), int(d[6:])) for d in ds))
    out = {'run': 'emul', 'stage': 'S3 step (7)', 'made': S.now_local(), 'design': "the study design's splits, with the two-year emission tests", 'validation_rule': 'every day of each ISO week whose number is divisible by 5 (datetime.date.isocalendar)', 'index_rule': "index into the year's arrays: days since January 1 of that year", 'folds': {}, 'f6': {}}
    allok = True
    for yh in S.YEARS:
        ytr = [y for y in S.YEARS if y != yh]
        tr = [d for y in ytr for d in S.days_of_year(y) if not S.is_validation(d)]
        va = [d for y in ytr for d in S.days_of_year(y) if S.is_validation(d)]
        he = S.days_of_year(yh)
        t = test_split(tr, va, he, ytr, yh)
        allok = allok and t['pass']
        out['folds'][str(yh)] = {'held_out_year': yh, 'label': LABEL[yh], 'training_years': ytr, 'train': pack(tr), 'validation': pack(va), 'held_out': pack(he), 'test': t}
    for ytr, yte in ((2017, 2019), (2019, 2017)):
        tr = [d for d in S.days_of_year(ytr) if d in ext[ytr] and (not S.is_validation(d))]
        va = [d for d in S.days_of_year(ytr) if d in ext[ytr] and S.is_validation(d)]
        he = [d for d in S.days_of_year(yte) if d in ext[yte]]
        t = test_split(tr, va, he, [ytr], yte, universe=ext[ytr])
        allok = allok and t['pass']
        name = 'train%d_test%d' % (ytr, yte)
        out['f6'][name] = {'train_year': ytr, 'test_year': yte, 'arms': {'TRUE': 'emis_true_%d.npy for the training days and emis_true_%d.npy for the test days' % (ytr, yte), 'OTHER': 'each year the emission climatology of the other year: the training days take emis_clim_%d.npy, the test days emis_clim_%d.npy' % (yte, ytr), 'OWN': 'each year its own emission climatology: the training days take emis_clim_%d.npy, the test days emis_clim_%d.npy' % (ytr, yte)}, 'extracted_days_train_year': len(ext[ytr]), 'extracted_days_test_year': len(ext[yte]), 'train': pack(tr), 'validation': pack(va), 'held_out': pack(he), 'test': t}
    out['all_tests_pass'] = allok
    out['summary'] = {k: {'train': v['train']['count'], 'validation': v['validation']['count'], 'held_out': v['held_out']['count']} for k, v in list(out['folds'].items()) + list(out['f6'].items())}
    info = S.write_json(os.path.join(S.CFG, 'SPLITS.json'), out)
    print('all tests pass: %s; %s' % (allok, info))
    for k, v in out['summary'].items():
        print(k, v)
if __name__ == '__main__':
    main()
