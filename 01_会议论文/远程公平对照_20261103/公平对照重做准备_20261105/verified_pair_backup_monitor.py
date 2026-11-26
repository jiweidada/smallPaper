"""Attach point-in-time process identity checks to a pair backup coordinator.

Connections and expected identities are injected; no connection, process or
training job is created. Expected identities must later be bound to a trusted
launch record. An identity check is not proof of final training completion.
"""
from pathlib import PurePosixPath
import copy

import epoch_backup_guard as guard
import epoch_backup_transport as transport
import remote_process_identity as identity
from pair_backup_monitor import PairBackupMonitor


class IdentityCheckedSFTP:
    def __init__(self, connection, expected):
        self.connection = connection
        self.expected = copy.deepcopy(expected)
        identity.validate_expected(self.expected)
        guard.require(callable(getattr(connection, 'readlink', None)), 'Connection lacks readlink')

    def lstat(self, path):
        result = self.connection.lstat(path)
        if PurePosixPath(path).name == 'manifest.json':
            identity.verify(self.connection, self.expected)
        return result

    def open(self, path, mode):
        if mode == 'wx' and PurePosixPath(path).name in ('ack.uploading', 'ack.ready'):
            identity.verify(self.connection, self.expected)
        return self.connection.open(path, mode)


def validate_arm_identity(arm, remote_root, expected):
    identity.validate_expected(expected)
    root = PurePosixPath(remote_root)
    argv = expected['argv']
    guard.require(len(argv) == 7 and PurePosixPath(argv[0]).is_absolute() and
                  str(PurePosixPath(argv[0])) == argv[0] and argv[1:3] == ['-B', '-u'] and
                  argv[3] == str(root / 'guarded_fair_runner.py') and
                  argv[4:] == ['train', '--arm', arm], 'Expected command is not bound to the arm and run root')


class VerifiedPairBackupMonitor:
    def __init__(self, connections, remote_roots, expected_identities, local_root, allowed_root,
                 run_id, protocol_sha256, reserve_bytes=512 * 1024**2):
        arms = ('BASE', 'FULL')
        guard.require(isinstance(connections, dict) and set(connections) == set(arms) and
                      isinstance(remote_roots, dict) and set(remote_roots) == set(arms) and
                      isinstance(expected_identities, dict) and set(expected_identities) == set(arms),
                      'Two arms, roots and identities are required')
        checked = {}
        fingerprints = []
        for arm in arms:
            transport.remote_stage_path(remote_roots[arm], run_id, arm, 0)
            expected = copy.deepcopy(expected_identities[arm])
            validate_arm_identity(arm, remote_roots[arm], expected)
            fingerprints.append((expected['boot_id'], expected['pid_namespace'], expected['pid'], expected['start_ticks']))
            checked[arm] = IdentityCheckedSFTP(connections[arm], expected)
        guard.require(fingerprints[0] != fingerprints[1], 'Both arms cannot share one process identity')
        self.monitor = PairBackupMonitor(checked, remote_roots, local_root, allowed_root, run_id,
                                         protocol_sha256, reserve_bytes=reserve_bytes)

    def poll_once(self, arm):
        return self.monitor.poll_once(arm)

    def snapshot(self):
        return self.monitor.snapshot()

    def halt(self, reason):
        return self.monitor.halt(reason)
