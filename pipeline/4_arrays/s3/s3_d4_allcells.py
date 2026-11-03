"""Sums the fill and NaN counts on all cells of every RSIG series and year, and of the lag answers, from the input-check summary (read only)."""
import json
import os
import sys
sys.path.insert(0, '.\\run\\code')
from s3 import s3common as S

def main():
    with open(os.path.join(S.WORK_S3, 'd4_summary.json'), 'r', encoding='ascii') as f:
        sm = json.load(f)
    fa = sum((v['fill_all'] for v in sm['series_year'].values()))
    na = sum((v['nan_all'] for v in sm['series_year'].values()))
    fm = sum((v['fill_or_nan_mask'] for v in sm['series_year'].values()))
    print('series-years %d; fill on all cells %d; NaN on all cells %d; fill or NaN on masked cells %d' % (len(sm['series_year']), fa, na, fm))
    with open(os.path.join(S.WORK_S3, 'd4_stats.json'), 'r', encoding='ascii') as f:
        st = json.load(f)
    lag = [a for a in st['answers'] if a['period'] == '20141231']
    print('lag answers %d; fill on all cells %d; NaN on all cells %d' % (len(lag), sum((s['fill_all'] for a in lag for s in a['steps'])), sum((s['nan_all'] for a in lag for s in a['steps']))))
if __name__ == '__main__':
    S.run_main(main)
