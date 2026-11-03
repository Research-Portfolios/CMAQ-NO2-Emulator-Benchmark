"""Downloads the Natural Earth files with cartopy's own downloader, at most one request a second, records each answer's status and headers, then hashes every file and writes its provenance row."""
import os
import sys
import time
import warnings
CODE_DIR = '/content/emul/run/code'
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
from choi_dl import common, prov
HOST = 'naturalearth.s3.amazonaws.com'
FILES = (('10m', 'cultural', 'admin_1_states_provinces'), ('10m', 'cultural', 'admin_1_states_provinces_lakes'), ('10m', 'cultural', 'admin_0_boundary_lines_land'), ('50m', 'cultural', 'admin_0_boundary_lines_land'), ('50m', 'cultural', 'admin_0_countries'), ('10m', 'cultural', 'populated_places'), ('10m', 'physical', 'coastline'), ('50m', 'physical', 'coastline'), ('50m', 'physical', 'land'), ('10m', 'physical', 'lakes'), ('50m', 'physical', 'lakes'))
VERSION = "Natural Earth, as named by its files (ne_<resolution>_<name>; cartopy does not extract the zip's VERSION file; the zip's ETag and Last-Modified are recorded)"
NOTE = 'public domain; Made with Natural Earth'
CONTIGUOUS_48_AND_DC = ('Alabama', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'District of Columbia', 'Florida', 'Georgia', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming')
CITIES = ('New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'Austin')

def template():
    from cartopy.io import shapereader
    return shapereader.NEShpDownloader._NE_URL_TEMPLATE

def fetch(log=print):
    common.set_temp()
    import cartopy
    from cartopy import io as cio
    from cartopy.io import shapereader
    from pathlib import Path
    os.makedirs(common.NATURAL_EARTH, exist_ok=True)
    cartopy.config['data_dir'] = Path(common.NATURAL_EARTH)
    cartopy.config['pre_existing_data_dir'] = Path(common.NATURAL_EARTH)
    answers = []
    orig = cio.Downloader._urlopen

    def recording_urlopen(self, url):
        r = orig(self, url)
        h = getattr(r, 'headers', {})
        answers.append({'url': url, 'status': getattr(r, 'status', None), 'etag': str(h.get('ETag', '')).strip('"'), 'last_modified': h.get('Last-Modified'), 'length': h.get('Content-Length')})
        return r
    cio.Downloader._urlopen = recording_urlopen
    out = []
    last = 0.0
    try:
        for res, cat, name in FILES:
            url = template().format(resolution=res, category=cat, name=name)
            n0 = len(answers)
            wait = last + 1.0 - time.time()
            if wait > 0:
                time.sleep(wait)
            t0 = time.time()
            u, loc = common.now_pair()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                shp = shapereader.natural_earth(resolution=res, category=cat, name=name)
            if len(answers) > n0:
                last = t0
            rec = {'resolution': res, 'category': cat, 'name': name, 'url': url, 'shp': str(shp), 'downloaded': len(answers) > n0, 'retrieved_utc': u, 'retrieved_local': loc, 'seconds': time.time() - t0}
            if len(answers) > n0:
                rec.update({k: answers[-1][k] for k in ('status', 'etag', 'last_modified', 'length')})
            out.append(rec)
            log('NE %s %s %s: %s in %.1f s' % (res, cat, name, 'downloaded (status %s)' % rec.get('status') if rec['downloaded'] else 'present', rec['seconds']))
    finally:
        cio.Downloader._urlopen = orig
    return out

def rows(fetched, log=print):
    by_stem = {}
    for r in fetched:
        stem = 'ne_%s_%s' % (r['resolution'], r['name'])
        by_stem[r['category'], stem] = r
    have = prov.load(common.NATURAL_EARTH)
    out = []
    for dp, dn, fns in os.walk(common.NATURAL_EARTH):
        for fn in sorted(fns):
            if fn in ('PROVENANCE_ASCII.txt', 'PROVENANCE.jsonl'):
                continue
            p = os.path.join(dp, fn)
            sha = prov.sha256_file(p)
            nb = os.stat(p).st_size
            out.append((p, nb, sha))
            rp = common.rel(p)
            if rp in have and have[rp].get('sha256') == sha:
                continue
            cat = os.path.basename(dp)
            stem = os.path.splitext(fn)[0]
            r = by_stem.get((cat, stem), {})
            prov.append(common.NATURAL_EARTH, path=p, source='%s (member %s of the zip, extracted by cartopy 0.25.0)' % (r.get('url', 'unknown'), fn), retrieved_utc=r.get('retrieved_utc'), retrieved_local=r.get('retrieved_local'), http_status=r.get('status'), bytes=nb, sha256=sha, version=VERSION, etag=r.get('etag'), last_modified=r.get('last_modified'), note=NOTE)
    log('NE rows: %d files hashed' % len(out))
    return out

def checks():
    from cartopy.io import shapereader
    adm1 = os.path.join(common.NATURAL_EARTH, 'shapefiles', 'natural_earth', 'cultural', 'ne_10m_admin_1_states_provinces.shp')
    pp = os.path.join(common.NATURAL_EARTH, 'shapefiles', 'natural_earth', 'cultural', 'ne_10m_populated_places.shp')
    r = shapereader.Reader(adm1)
    us = []
    fields = None
    for rec in r.records():
        at = rec.attributes
        if fields is None:
            fields = sorted(at)
        if at.get('adm0_a3') == 'USA' or at.get('iso_a2') == 'US':
            us.append({'name': at.get('name'), 'iso_3166_2': at.get('iso_3166_2'), 'type_en': at.get('type_en')})
    names = {u['name'] for u in us}
    states = {'n_us_records': len(us), 'found': [n for n in CONTIGUOUS_48_AND_DC if n in names], 'missing': [n for n in CONTIGUOUS_48_AND_DC if n not in names], 'other_us_records': sorted((n for n in names if n not in CONTIGUOUS_48_AND_DC)), 'fields_used': ['adm0_a3', 'iso_a2', 'name', 'iso_3166_2', 'type_en']}
    r2 = shapereader.Reader(pp)
    cand = {c: [] for c in CITIES}
    for rec in r2.records():
        at = rec.attributes
        nm = at.get('NAME') if 'NAME' in at else at.get('name')
        a0 = at.get('ADM0NAME') if 'ADM0NAME' in at else at.get('adm0name')
        a3 = at.get('ADM0_A3') if 'ADM0_A3' in at else at.get('adm0_a3')
        if nm in cand and (a3 == 'USA' or a0 == 'United States of America'):
            cand[nm].append({'name': nm, 'adm1': at.get('ADM1NAME', at.get('adm1name')), 'lat': float(at.get('LATITUDE', at.get('latitude', float('nan')))), 'lon': float(at.get('LONGITUDE', at.get('longitude', float('nan')))), 'pop_max': at.get('POP_MAX', at.get('pop_max')), 'pop_min': at.get('POP_MIN', at.get('pop_min'))})
    cities = {}
    for c, lst in cand.items():
        if lst:
            best = max(lst, key=lambda d: d['pop_max'] or 0)
            cities[c] = {'n_records': len(lst), 'chosen': best}
        else:
            cities[c] = {'n_records': 0, 'chosen': None}
    return {'states': states, 'cities': cities}
