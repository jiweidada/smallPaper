"""Check process identity parsing with in-memory fake proc files only."""
from io import BytesIO
from unittest import TestCase, main

import remote_process_identity as identity

BOOT = '12345678-1234-5678-9abc-123456789abc'
NAMESPACE = 'pid:[4026531836]'
PID, TICKS = 4242, 123456
ARGV = ['/opt/python', '-B', '-u', '/work/guarded_runner.py', 'train', '--arm', 'BASE']


def stat_record(pid=PID, ticks=TICKS, state='S'):
    fields = [state] + ['1'] * 18 + [str(ticks)] + ['1'] * 4
    return (str(pid) + ' (python ) (worker)) ' + ' '.join(fields) + '\n').encode()


class FakeProc:
    def __init__(self):
        self.boots = [BOOT, BOOT]
        self.namespaces = [NAMESPACE, NAMESPACE]
        self.stats = [stat_record(), stat_record()]
        self.cmdline = b'\x00'.join(x.encode() for x in ARGV) + b'\x00'

    def open(self, path, mode):
        assert mode == 'rb'
        if path == '/proc/sys/kernel/random/boot_id':
            return BytesIO((self.boots.pop(0) + '\n').encode())
        if path == f'/proc/{PID}/stat':
            return BytesIO(self.stats.pop(0))
        if path == f'/proc/{PID}/cmdline':
            return BytesIO(self.cmdline)
        raise FileNotFoundError(path)

    def readlink(self, path):
        assert path == '/proc/self/ns/pid'
        return self.namespaces.pop(0)


def expected():
    return {'boot_id': BOOT, 'pid_namespace': NAMESPACE, 'pid': PID, 'start_ticks': TICKS, 'argv': list(ARGV)}


class IdentityTests(TestCase):
    def test_matching_process_is_only_a_point_in_time_observation(self):
        result = identity.verify(FakeProc(), expected())
        self.assertEqual(result['status'], 'LIVE_IDENTITY_MATCHED_AT_OBSERVATION')
        self.assertEqual(result['start_ticks'], TICKS)
        self.assertNotIn('argv', result)

    def test_reused_pid_with_new_start_time_is_unverified(self):
        proc = FakeProc()
        proc.stats = [stat_record(ticks=TICKS + 1), stat_record(ticks=TICKS + 1)]
        with self.assertRaises(identity.IdentityUnverified):
            identity.verify(proc, expected())

    def test_process_replacement_during_observation_is_unverified(self):
        proc = FakeProc()
        proc.stats = [stat_record(), stat_record(ticks=TICKS + 1)]
        with self.assertRaises(identity.IdentityUnverified):
            identity.verify(proc, expected())

    def test_boot_or_namespace_change_is_unverified(self):
        proc = FakeProc()
        proc.boots[1] = 'abcdefab-1234-5678-9abc-123456789abc'
        with self.assertRaises(identity.IdentityUnverified):
            identity.verify(proc, expected())
        proc = FakeProc()
        proc.namespaces[1] = 'pid:[4026531837]'
        with self.assertRaises(identity.IdentityUnverified):
            identity.verify(proc, expected())

    def test_zombie_or_unknown_state_is_unverified(self):
        for state in ('Z', 'X', '?'):
            proc = FakeProc()
            proc.stats = [stat_record(state=state), stat_record(state=state)]
            with self.assertRaises(identity.IdentityUnverified):
                identity.verify(proc, expected())

    def test_command_change_or_truncated_cmdline_is_unverified(self):
        proc = FakeProc()
        proc.cmdline = b'/opt/python\x00different\x00'
        with self.assertRaises(identity.IdentityUnverified):
            identity.verify(proc, expected())
        proc = FakeProc()
        proc.cmdline = proc.cmdline[:-1]
        with self.assertRaises(identity.IdentityUnverified):
            identity.verify(proc, expected())

    def test_missing_proc_file_is_unverified_not_a_completion(self):
        proc = FakeProc()
        def missing(path, mode):
            raise FileNotFoundError(path)
        proc.open = missing
        with self.assertRaises(identity.IdentityUnverified):
            identity.verify(proc, expected())

    def test_malformed_expected_identity_is_rejected(self):
        for changed in ({'pid': True}, {'start_ticks': 0}, {'boot_id': 'bad'},
                        {'pid_namespace': 'pid:[0]'}, {'argv': ['/ok', '']}, {'argv': ['/ok\x00bad']}):
            with self.assertRaises(identity.IdentityUnverified):
                identity.verify(FakeProc(), {**expected(), **changed})

    def test_oversized_stat_is_unverified(self):
        proc = FakeProc()
        proc.stats = [b'x' * 16385, stat_record()]
        with self.assertRaises(identity.IdentityUnverified):
            identity.verify(proc, expected())


if __name__ == '__main__':
    main(verbosity=2)
