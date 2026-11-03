"""Helpers for the S3 test scripts: the object keys of the merged, point, stack groups and MCIP files, the tar contents files with their provenance rows, JSON reading and writing, a log and a timer."""
import json
import os
import sys
import time
CODE_DIR = '/content/emul/run/code'
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
from choi_dl import common, prov, s3io
WORK_S3 = os.path.join(common.WORK, 's2a', 's3')
S3META = os.path.join(common.RAW, 's3meta')
MCIP_DIR = os.path.join(common.RAW, 'mcip')
SG_DIR = os.path.join(common.RAW, 'emis', 'stackgroups')
LOG = os.path.join(WORK_S3, 's2a_s3_log.txt')
KEYS_JSON = os.path.join(WORK_S3, 'keys.json')
VERSION_EMIS = 'EQUATES v1.0 (the key names)'
VERSION_MCIP = 'EQUATES v1.0; MCIP v5.1; WRF v4.1.1 (the key names)'
YEARS = (2017, 2019)
MCIP_DAYS = ['20160701'] + ['2016%02d15' % m for m in range(1, 13)]

def log(text):
    os.makedirs(WORK_S3, exist_ok=True)
    line = '%s %s' % (common.now_pair()[1][:19], common.mask(text))
    with open(LOG, 'ab') as f:
        f.write((line + '\n').encode('ascii', 'replace'))
    print(line, flush=True)

def merged_key(y, m, ext='tar'):
    return 'CMAQ_12US1/INPUT/%d/emis/model_ready_emis_%d_merged_nobeis_norwc_%02d_EQUATES_v1.0.%s' % (y, y, m, ext)

def pt_key(y, m, ext='tar'):
    return 'CMAQ_12US1/INPUT/%d/emis/model_ready_emis_ptsectors_plus_rwc_%d_%02d_EQUATES_v1.0.%s' % (y, y, m, ext)

def sg_key(y, ext='tar'):
    return 'CMAQ_12US1/INPUT/%d/emis/model_ready_emis_%d_stackgroups_epicsoil_EQUATES_v1.0.%s' % (y, y, ext)

def mcip_key(day, ext='tar'):
    return 'CMAQ_12US1/INPUT/%s/met/mcip_v51_wrf_v411_noltng/%s/MCIPv51_WRFv411_noltng_12US1.35L.%s.%s' % (day[:4], day[4:6], day, ext)

def all_tar_keys():
    out = []
    for y in YEARS:
        for m in range(1, 13):
            out.append(('merged', merged_key(y, m)))
    for y in YEARS:
        for m in range(1, 13):
            out.append(('ptsectors', pt_key(y, m)))
    for y in YEARS:
        out.append(('stackgroups', sg_key(y)))
    for d in MCIP_DAYS:
        out.append(('mcip', mcip_key(d)))
    return out

def contents_key(tar_key):
    assert tar_key.endswith('.tar')
    return tar_key[:-4] + '.contents.txt'

def get_contents(client, key, head):
    path = os.path.join(S3META, key.split('/')[-1])
    if not prov.verified(S3META, path):
        os.makedirs(S3META, exist_ok=True)
        info = client.stream_to_file(key, 0, head['size'], path, if_match=head['etag'])
        u, loc = common.now_pair()
        prov.append(S3META, path=path, source='s3://%s/%s' % (s3io.BUCKET, key), byte_range='0-%d' % (head['size'] - 1), retrieved_utc=u, retrieved_local=loc, http_status=info.get('status'), bytes=os.stat(path).st_size, sha256=info['sha256'], version=VERSION_EMIS if '/emis/' in key else VERSION_MCIP, etag=head['etag'], last_modified=head['last_modified'], note='contents file of the tar (tar -tv lines); kept raw; parsed for member names, types and sizes only')
    with open(path, 'rb') as f:
        text = f.read().decode('utf-8', 'replace')
    members, bad = s3io.parse_contents(text)
    return (text, members, bad)

def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.partial'
    with open(tmp, 'wb') as f:
        f.write(s3io.dumps(obj).encode('ascii'))
    os.replace(tmp, path)

def load_json(path):
    with open(path, 'rb') as f:
        return json.loads(f.read().decode('ascii'))

def timer():
    t0 = time.time()
    return lambda: time.time() - t0
