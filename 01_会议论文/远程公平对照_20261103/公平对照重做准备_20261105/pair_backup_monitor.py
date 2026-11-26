"""A fail-closed, single-use coordinator for two injected SFTP-like handles.

This module creates no network connection, starts no process and does not infer
training completion. A separate, authorized caller must verify host identity,
process liveness, execution context and each trainer's final receipt.
"""
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from threading import Event, Lock
import stat

import epoch_backup_guard as guard
import epoch_backup_transport as transport

ARMS = ('BASE', 'FULL')


class MonitorHalted(RuntimeError):
    pass


class _AcknowledgementGate:
    def __init__(self, connection, stop_event, acknowledgement_lock):
        self.connection = connection
        self.stop_event = stop_event
        self.acknowledgement_lock = acknowledgement_lock

    def lstat(self, path):
        return self.connection.lstat(path)

    def open(self, path, mode):
        if mode == 'wx' and PurePosixPath(path).name == 'ack.ready':
            return self._open_marker(path, mode)
        return self.connection.open(path, mode)

    @contextmanager
    def _open_marker(self, path, mode):
        # Serialize a stop request with marker publication. A marker that was
        # already published cannot be revoked; no new marker is issued after a
        # stop request obtains this lock.
        with self.acknowledgement_lock:
            guard.require(not self.stop_event.is_set(), 'Stop requested before acknowledgement marker')
            with self.connection.open(path, mode) as stream:
                yield stream


class PairBackupMonitor:
    """A new session accepts a fresh local run id and never retries failures."""

    def __init__(self, connections, remote_roots, local_root, allowed_root, run_id, protocol_sha256,
                 reserve_bytes=512 * 1024**2):
        guard.require(isinstance(connections, dict) and set(connections) == set(ARMS), 'Two connections are required')
        guard.require(isinstance(remote_roots, dict) and set(remote_roots) == set(ARMS), 'Two remote roots are required')
        for arm in ARMS:
            guard.validate_identity(run_id, arm, protocol_sha256)
            transport.remote_stage_path(remote_roots[arm], run_id, arm, 0)
            guard.require(callable(getattr(connections[arm], 'lstat', None)) and
                          callable(getattr(connections[arm], 'open', None)), 'Connection interface incomplete')
        self.local_root = guard.checked_root(local_root, allowed_root)
        self.allowed_root = Path(allowed_root).resolve(strict=True)
        run_path = self.local_root / run_id
        guard.require(not run_path.exists() and not run_path.is_symlink() and
                      not getattr(run_path, 'is_junction', lambda: False)(),
                      'Existing backup run cannot be reused automatically')
        guard.require(type(reserve_bytes) is int and reserve_bytes >= 0, 'Invalid space reserve')
        self._stop_event = Event()
        self._acknowledgement_lock = Lock()
        self._requested_reason = None
        self.connections = {arm: _AcknowledgementGate(connections[arm], self._stop_event,
                                                       self._acknowledgement_lock) for arm in ARMS}
        self.remote_roots = dict(remote_roots)
        self.run_id, self.protocol_sha256, self.reserve_bytes = run_id, protocol_sha256, reserve_bytes
        self._next = {arm: 0 for arm in ARMS}
        self._failure = None
        self._lock = Lock()

    def snapshot(self):
        with self._lock:
            if self._failure is not None or self._stop_event.is_set():
                status = 'HALTED_NO_AUTOMATIC_RETRY'
            elif all(self._next[arm] == guard.EPOCHS for arm in ARMS):
                status = 'ALL_BACKUPS_ACKNOWLEDGED_TRAINING_COMPLETION_UNVERIFIED'
            else:
                status = 'ACTIVE_PROCESS_LIVENESS_UNVERIFIED'
            return {'status': status, 'next_epoch': dict(self._next),
                    'failure': dict(self._failure) if self._failure else None}

    def halt(self, reason):
        guard.require(reason in ('OPERATOR_STOP', 'CONNECTION_FAILURE', 'PROCESS_IDENTITY_UNVERIFIED'),
                      'Invalid halt reason')
        # Do not wait for a long copy before making the request visible to the
        # acknowledgement gate. The marker operation itself is synchronized.
        with self._acknowledgement_lock:
            self._requested_reason = reason
            self._stop_event.set()
        with self._lock:
            if self._failure is None:
                self._failure = {'arm': None, 'epoch': None, 'reason': reason}

    def poll_once(self, arm):
        """Check only the next stage; no wait, retry or process-state inference.

        Copies are serialized across arms so that, after a failure is observed,
        no subsequent acknowledgement for either arm is sent by this session.
        An acknowledgement already sent cannot be revoked.
        """
        guard.require(arm in ARMS, 'Invalid arm')
        with self._lock:
            if self._failure is not None or self._stop_event.is_set():
                failure = self._failure or {'arm': None, 'epoch': None, 'reason': self._requested_reason}
                return {'status': 'HALTED_NO_AUTOMATIC_RETRY', 'arm': arm, 'failure': dict(failure)}
            epoch = self._next[arm]
            if epoch == guard.EPOCHS:
                return {'status': 'ARM_BACKUPS_ACKNOWLEDGED_TRAINING_COMPLETION_UNVERIFIED', 'arm': arm}
            stage = transport.remote_stage_path(self.remote_roots[arm], self.run_id, arm, epoch)
            try:
                try:
                    info = self.connections[arm].lstat(str(stage / 'manifest.json'))
                except FileNotFoundError:
                    return {'status': 'AWAITING_NEXT_MANIFEST_PROCESS_LIVENESS_UNVERIFIED', 'arm': arm, 'epoch': epoch}
                guard.require(stat.S_ISREG(info.st_mode), 'Manifest is not a regular file')
                result = transport.backup_stage(self.connections[arm], self.remote_roots[arm], self.local_root,
                                                self.allowed_root, self.run_id, arm, epoch, self.protocol_sha256,
                                                reserve_bytes=self.reserve_bytes)
                guard.require(result['status'] == 'LOCAL_BACKUP_VERIFIED_AND_ACKNOWLEDGED' and
                              result['epoch_index'] == epoch, 'Unexpected backup result')
                self._next[arm] += 1
                return result
            except Exception as exc:
                with self._acknowledgement_lock:
                    self._stop_event.set()
                self._failure = {'arm': arm, 'epoch': epoch, 'reason': self._requested_reason or type(exc).__name__}
                raise MonitorHalted('Backup failed; the pair monitor is halted without automatic retry') from exc
