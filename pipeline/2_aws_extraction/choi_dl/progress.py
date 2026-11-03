"""Writes the downloader's progress file (one ASCII line per item) and its status file of counts, replaced whole after each item; a write that fails is retried and then held for the next write, so that a busy file never stops the queue."""
import json
import os
import threading
import time
from choi_dl import common
PROGRESS = os.path.join(common.RUN_LOGS, 'DL_PROGRESS_ASCII.txt')
STATUS = os.path.join(common.RUN_LOGS, 'DL_STATUS.json')
RETRY_TRIES = 20
RETRY_GAP_S = 0.25
HELD_MAX = 2000
STATUSES = ('done', 'skipped', 'failed', 'refused')

def retry(fn, tries=None, gap=None):
    tries = max(1, int(RETRY_TRIES if tries is None else tries))
    gap = float(RETRY_GAP_S if gap is None else gap)
    for i in range(tries):
        try:
            fn()
            return True
        except OSError:
            if i + 1 < tries:
                time.sleep(gap)
        except Exception:
            return False
    return False

def append_bytes(path, data, tries=None, gap=None):

    def _do():
        with open(path, 'ab') as f:
            f.write(data)
    return retry(_do, tries, gap)

def _new_queue():
    return {'total': None, 'done': 0, 'skipped': 0, 'failed': [], 'refused': [], 'bytes': 0}

def _int(v):
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0

class Progress:

    def __init__(self, progress_path=PROGRESS, status_path=STATUS, pid=None):
        self.progress_path = progress_path
        self.status_path = status_path
        self.lock = threading.Lock()
        self.status = {'pid': pid if pid is not None else os.getpid(), 'started_local': common.now_pair()[1], 'updated_local': None, 'stage': None, 'queues': {}}
        self.held = []
        self.held_dropped = 0
        self.write_failures = {'progress': 0, 'status': 0}
        os.makedirs(os.path.dirname(progress_path), exist_ok=True)

    def _append(self, data):
        buf = b''.join(self.held) + data
        if append_bytes(self.progress_path, buf):
            self.held = []
            return True
        self.write_failures['progress'] += 1
        self.held.append(data)
        if len(self.held) > HELD_MAX:
            n = len(self.held) - HELD_MAX
            del self.held[:n]
            self.held_dropped += n
        return False

    def _write_status(self):
        try:
            self.status['updated_local'] = common.now_pair()[1]
            self.status['write_failures'] = dict(self.write_failures)
            self.status['progress_held'] = len(self.held)
            self.status['progress_dropped'] = self.held_dropped
            data = json.dumps(self.status, sort_keys=True, indent=1).encode('ascii')
        except Exception:
            return False
        tmp = self.status_path + '.tmp'

        def _tmp():
            with open(tmp, 'wb') as f:
                f.write(data)
        ok = retry(_tmp) and retry(lambda: os.replace(tmp, self.status_path))
        if not ok:
            self.write_failures['status'] += 1
        return ok

    @staticmethod
    def _encode(lines):
        return ('\n'.join(lines) + '\n').encode('ascii', 'replace')

    def note(self, text):
        with self.lock:
            try:
                t = common.now_pair()[1][:19]
                data = self._encode(common.wrap('%s ' % t, common.mask(text)))
            except Exception:
                data = None
            if data is not None:
                self._append(data)
            self._write_status()

    def stage(self, name, total=None):
        with self.lock:
            self.status['stage'] = name
            q = self.status['queues'].setdefault(name, _new_queue())
            if total is not None:
                q['total'] = int(total)
        self.note('STAGE %s start; items %s' % (name, total))

    def item(self, queue, item_id, status, nbytes=0, seconds=None, note=''):
        if status not in STATUSES:
            raise ValueError('unknown item status %r' % status)
        with self.lock:
            q = self.status['queues'].setdefault(queue, _new_queue())
            if status in ('done', 'skipped'):
                q[status] += 1
            else:
                q[status].append(str(item_id))
            n = _int(nbytes)
            q['bytes'] += n
            try:
                t = common.now_pair()[1][:19]
                text = '%s %s %s %s b=%d' % (t, queue, item_id, status, n)
                if seconds is not None:
                    text += ' t=%.1f' % seconds
                if note:
                    text += ' ' + note
                data = self._encode(common.wrap('', common.mask(text)))
            except Exception:
                data = None
            if data is not None:
                self._append(data)
            self._write_status()
