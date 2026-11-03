"""Gives the float64 emission fields of each day of 2017 and 2019 by the same calls as the emission step, so that the round trip sees each value before the cast; each day's float32 cast is checked against the stored array."""
import collections
import datetime
import json
import os
import sys
sys.path.insert(0, '/content/emul/run/code')
import numpy as np
from s3 import s3_emis as E1
from s3 import s3common as S
from s3p2 import p2_emis as PE
from s3p2 import p2common as P

class Fields:

    def __init__(self, y, check_stored=True):
        self.y = int(y)
        inputs = S.inventory_inputs()
        with open(os.path.join(S.CFG, 'S2A_PLAN_S3.json'), 'r', encoding='ascii') as f:
            plan = json.load(f)
        kept = S.load_inventory()['sectors']['kept']
        self.daily = [s for s in S.DAILY_SECTORS if s in kept]
        self.rep = [s for s in S.REP_SECTORS if s in kept]
        self.hol = E1.holiday_dates(plan)
        self.merged = {e['date']: e for e in inputs['emis_merged']}
        self.pt = collections.defaultdict(dict)
        for e in inputs['pt_marine']:
            self.pt[e['sector']][e['date']] = e
        self.days = P.days_of_year(self.y)
        self.stored = None
        if check_stored:
            self.stored = np.load(os.path.join(P.ARRAYS_P2, 'emis_true_%d.npy' % self.y), mmap_mode='r')
        self.checked = 0

    def month(self, m):
        y = self.y
        cmean = {}
        for s in self.rep:
            mem = sorted((d for d in self.pt[s] if int(d[:4]) == y and int(d[4:6]) == m))
            by = collections.defaultdict(list)
            for d in mem:
                if d in self.hol[y]:
                    continue
                dd = datetime.date(int(d[:4]), int(d[4:6]), int(d[6:]))
                by[S.CLASS_OF_WEEKDAY[dd.weekday()]].append(d)
            cmean[s] = {}
            for c in ('Mon', 'TueFri', 'Sat', 'Sun'):
                use = by.get(c) or by.get('TueFri')
                acc = np.zeros(P.G, np.float64)
                for d in use:
                    acc += PE.p1_of(E1.load_npz(self.pt[s][d], PE.PT_SPECIES))
                cmean[s][c] = acc / len(use)
        out = {}
        for i, d in enumerate(self.days):
            if d.month != m:
                continue
            key = P.ymd(d)
            z = E1.load_npz(self.merged[key], PE.SPECIES)
            f = np.zeros((PE.NCH,) + P.G, np.float64)
            for k, sp in enumerate(PE.SPECIES):
                f[k] = z[sp]
            p1 = np.zeros(P.G, np.float64)
            for s in self.daily:
                p1 += PE.p1_of(E1.load_npz(self.pt[s][key], PE.PT_SPECIES))
            for s in self.rep:
                p1 += cmean[s][S.CLASS_OF_WEEKDAY[d.weekday()]]
            f[4] = p1
            if self.stored is not None:
                if not np.array_equal(f.astype(np.float32), self.stored[i]):
                    raise ValueError('emis_true_%d day %s differs from its float64 fields' % (y, key))
                self.checked += 1
            out[key] = f
        return out
