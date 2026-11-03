"""The detached downloader: runs every remaining RSIG and S3 request in order with its provenance rows, resuming past files whose rows verify, keeps the progress files, holds a lock so that one instance runs, and writes a closing line with each part's counts."""
import json
import os
import shutil
import sys
import traceback
sys.path.insert(0, '.\\run\\code')
from choi_dl import common
from choi_dl.progress import Progress
LABEL = 'data'
MARKER = os.path.join(common.ROOT, 'work/data_launcher.log')
PLAN_RSIG = os.path.join(common.RUN_CONFIG, 'S2A_PLAN_RSIG.json')
PLAN_S3 = os.path.join(common.RUN_CONFIG, 'S2A_PLAN_S3.json')
LOCK = os.path.join(common.W_TMP, 'work.lock')
DISK_FLOOR_GB = 30.0

def _marker(text):
    with open(MARKER, 'ab') as f:
        f.write((common.mask(text) + '\n').encode('ascii'))

def _pid_alive(pid):
    try:
        import psutil
        if not psutil.pid_exists(pid):
            return False
        cmd = ' '.join(psutil.Process(pid).cmdline()).lower()
        return 'downloader.py' in cmd
    except Exception:
        return False

def _take_lock():
    if os.path.exists(LOCK):
        try:
            old = int(open(LOCK, 'rb').read().decode('ascii').split()[0])
        except Exception:
            old = None
        if old and old != os.getpid() and _pid_alive(old):
            return old
    with open(LOCK, 'wb') as f:
        f.write(('%d %s\n' % (os.getpid(), common.now_pair()[1])).encode('ascii'))
    return None

def free_gb():
    return shutil.disk_usage(common.W).free / 1000000000.0

def disk_ok(progress, where):
    gb = free_gb()
    if gb < DISK_FLOOR_GB:
        progress.note('STOP disk: %.2f GB free on the work disk before %s (floor %.0f)' % (gb, where, DISK_FLOOR_GB))
        return False
    return True

def summary_text(name, s):
    if not s:
        return '%s not run' % name
    return '%s done %d skipped %d failed %d refused %d held %d' % (name, s.get('done', 0), s.get('skipped', 0), len(s.get('failed', [])), len(s.get('refused', [])), len(s.get('held', [])))

def main(argv):
    dry = '--dry-run' in argv
    common.set_temp()
    if dry:
        from choi_dl import dl_rsig, dl_s3
        plan_r = json.load(open(PLAN_RSIG, 'rb'))
        plan_s = json.load(open(PLAN_S3, 'rb'))
        p = Progress(progress_path=os.path.join(common.W_TMP, 'dryrun_progress.txt'), status_path=os.path.join(common.W_TMP, 'dryrun_status.json'))
        print(json.dumps({'rsig': dl_rsig.run(plan_r, p, dry_run=True), 's3': dl_s3.run(plan_s, p, dry_run=True)}, indent=1, sort_keys=True))
        return 0
    other = _take_lock()
    if other:
        _marker('run: %s not started %s; another instance is live (pid %d)' % (LABEL, common.now_pair()[1], other))
        return 3
    status = 'failed: did not finish'
    sums = {}
    progress = Progress()
    _marker('run: %s started %s; pid %d' % (LABEL, common.now_pair()[1], os.getpid()))
    try:
        from choi_dl import dl_rsig, dl_s3
        plan_r = json.load(open(PLAN_RSIG, 'rb'))
        plan_s = json.load(open(PLAN_S3, 'rb'))
        progress.note('START %s pid %d; free disk %.2f GB' % (LABEL, os.getpid(), free_gb()))
        if not disk_ok(progress, 'rsig'):
            status = 'failed: disk'
            return 2
        sums['rsig'] = dl_rsig.run(plan_r, progress)
        if not disk_ok(progress, 's3'):
            status = 'failed: disk'
            return 2
        sums['s3'] = dl_s3.run(plan_s, progress)
        bad = sum((len(s.get('failed', [])) + len(s.get('refused', [])) for s in sums.values() if s))
        status = 'ok' if bad == 0 else 'failed: %d items failed or refused' % bad
        return 0 if bad == 0 else 1
    except Exception as e:
        status = 'failed: %s' % type(e).__name__
        try:
            progress.note('ERROR %s: %s' % (type(e).__name__, str(e)[:300]))
            tb = traceback.format_exc().splitlines()[-6:]
            for t in tb:
                progress.note('  ' + t.strip()[:200])
        except Exception:
            pass
        return 1
    finally:
        try:
            progress.note('END %s status %s' % (LABEL, status))
        except Exception:
            pass
        _marker('run: %s finished; status %s; %s; %s; ended %s' % (LABEL, status, summary_text('rsig', sums.get('rsig')), summary_text('s3', sums.get('s3')), common.now_pair()[1]))
        try:
            if open(LOCK, 'rb').read().decode('ascii').split()[0] == str(os.getpid()):
                os.remove(LOCK)
        except Exception:
            pass
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
