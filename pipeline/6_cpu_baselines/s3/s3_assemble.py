"""Assembles the channel values of one day as 299 x 459 arrays, shared by the round trip and the array build: the RSIG meteorological fields (wind components as the mean of a cell's four corners), the static MCIP fields, vegetation, the emission map of the day's month and weekday, and the previous day's lagged fields."""
import datetime
import os
import sys
sys.path.insert(0, '/content/emul/run/code')
import numpy as np
from choi import f16
from s3 import s3common as S

def store_fn(kind):
    if kind == 'float16':
        return f16.f16_store
    if kind == 'float16_affine':
        return f16.pres_to_store
    if kind == 'float32':
        return lambda x: np.asarray(x).astype(np.float32)
    raise ValueError(kind)

def load_fn(kind):
    if kind == 'float16':
        return f16.f16_load
    if kind == 'float16_affine':
        return f16.pres_from_store
    if kind == 'float32':
        return lambda y: np.asarray(y, dtype=np.float64)
    raise ValueError(kind)

def load_static(inputs):
    import netCDF4
    e = [x for x in inputs['mcip'] if os.path.basename(x['path']) == 'GRIDCRO2D_20160701.nc4'][0]
    p = S.path_abs(e['path'])
    if S.sha256_file(p) != e['sha256']:
        raise ValueError('GRIDCRO2D differs from the inventory')
    ds = netCDF4.Dataset(p, 'r')
    try:
        out = {}
        for n in ('HT', 'LWMASK', 'PURB', 'DLUSE'):
            v = ds.variables[n]
            v.set_auto_maskandscale(False)
            out[n] = np.array(v[:])[0, 0].astype(np.float64)
    finally:
        ds.close()
    return {'L-1': out['HT'], 'L-2': out['LWMASK'], 'L-3': out['PURB'] / 100.0, 'L-6': out['DLUSE']}

def load_veg(inputs):
    import netCDF4
    out, used = ({}, {})
    for e in inputs['mcip']:
        name = os.path.basename(e['path'])
        if not name.startswith('METCRO2D_'):
            continue
        m = int(name[13:15])
        p = S.path_abs(e['path'])
        if S.sha256_file(p) != e['sha256']:
            raise ValueError('%s differs from the inventory' % name)
        ds = netCDF4.Dataset(p, 'r')
        try:
            t = ds.variables['TFLAG']
            t.set_auto_maskandscale(False)
            tf = np.array(t[:])
            want = S.yyyyddd(datetime.date(2016, m, 15))
            rec = [k for k in range(tf.shape[0]) if int(tf[k, 0, 0]) == want]
            hours = sorted((int(tf[k, 0, 1]) for k in rec))
            if len(rec) != 24 or hours != [h * 10000 for h in range(24)]:
                raise ValueError('%s: records %s' % (name, rec))
            v = ds.variables['VEG']
            v.set_auto_maskandscale(False)
            a = np.array(v[:])[rec, 0].astype(np.float64)
            out[m] = a.sum(axis=0) / 24.0
            used[m] = [rec[0], rec[-1]]
        finally:
            ds.close()
    if sorted(out) != list(range(1, 13)):
        raise ValueError('VEG months %s' % sorted(out))
    return (out, used)

class MonthReader:

    def __init__(self, inputs, check_raw=True):
        self.idx = S.rsig_index(inputs)
        self.check_raw = check_raw

    def entry(self, series, period):
        return self.idx[series, period]

    def read(self, series, period):
        var, _u, dot = S.SERIES[series]
        e = self.idx[series, period]
        r = S.read_answer(e, var, check_raw=self.check_raw)
        v = r['values'][:, 0]
        dates = [int(x) for x in r['tflag'][:, 0, 0]]
        if dot:
            if v.shape[1:] != (300, 460):
                raise ValueError('dot shape %s' % (v.shape,))
            v = S.dot_to_cell(v)
        elif v.shape[1:] != (S.NROWS, S.NCOLS):
            raise ValueError('shape %s' % (v.shape,))
        return (v, dates)

    def read_month(self, series, y, m):
        return self.read(series, '%04d%02d' % (y, m))

def met_of_day(month_data, k):
    out = {}
    for ch, ser in S.MET_SERIES.items():
        out[ch] = month_data[ser][0][k]
    return out

def lag_sources(prev_met):
    return {x: prev_met[m] for x, m in S.LAG_OF.items()}

def day_sources(d, met, prev_met, static, veg, clim):
    src = dict(met)
    src.update(static)
    src['L-5'] = veg[d.month]
    w = d.weekday()
    for k, ch in enumerate(S.EMIS):
        src[ch] = clim[d.month - 1, w, k]
    src.update(lag_sources(prev_met))
    return src

def read_lag_answers(reader):
    out = {}
    for x, m in S.LAG_OF.items():
        ser = S.MET_SERIES[m]
        v, dates = reader.read(ser, '20141231')
        if dates != [2014365] or v.shape[0] != 1:
            raise ValueError('lag %s: %s' % (ser, dates))
        out[m] = v[0]
    return out
