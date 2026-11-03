"""Writes the trainer's progress file and status file, each write retried and never raising an error."""
import json
import os
import threading
from choi_dl import common
from choi_dl.progress import append_bytes, retry

class TrainProgress:

    def __init__(self, progress_path, status_path, pid=None):
        self.progress_path = progress_path
        self.status_path = status_path
        self.lock = threading.Lock()
        self.status = {'label': 'train', 'pid': pid if pid is not None else os.getpid(), 'started_local': common.now_pair()[1], 'updated_local': None, 'current': None, 'trainings': {}, 'counts': {}}
        self.held = []
        os.makedirs(os.path.dirname(progress_path), exist_ok=True)
        os.makedirs(os.path.dirname(status_path), exist_ok=True)

    def _write_status(self):
        try:
            self.status['updated_local'] = common.now_pair()[1]
            c = {}
            for v in self.status['trainings'].values():
                k = str(v.get('status', '?')).split(':')[0]
                c[k] = c.get(k, 0) + 1
            self.status['counts'] = c
            data = json.dumps(self.status, sort_keys=True, indent=1, default=str).encode('ascii', 'replace')
        except Exception:
            return False
        tmp = self.status_path + '.tmp'

        def _tmp():
            with open(tmp, 'wb') as f:
                f.write(data)
        return retry(_tmp) and retry(lambda: os.replace(tmp, self.status_path))

    def note(self, text):
        with self.lock:
            try:
                t = common.now_pair()[1][:19]
                lines = common.wrap('%s ' % t, common.mask(text))
                data = ('\n'.join(lines) + '\n').encode('ascii', 'replace')
            except Exception:
                data = None
            if data is not None:
                buf = b''.join(self.held) + data
                if append_bytes(self.progress_path, buf):
                    self.held = []
                else:
                    self.held.append(data)
                    self.held = self.held[-2000:]
            self._write_status()

    def training(self, name, **fields):
        with self.lock:
            e = self.status['trainings'].setdefault(name, {})
            e.update(fields)
            if fields.get('status') == 'running':
                self.status['current'] = name
            elif self.status.get('current') == name and 'status' in fields:
                self.status['current'] = None
            self._write_status()

    def set(self, **fields):
        with self.lock:
            self.status.update(fields)
            self._write_status()
