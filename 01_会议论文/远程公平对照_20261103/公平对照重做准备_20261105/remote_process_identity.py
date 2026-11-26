"""Read-only Linux process identity check over an injected SFTP-like handle.

No connection is created here. Failure means identity is unverified; it does
not establish that a job finished, failed, or may safely be restarted.
"""
import re

BOOT_ID = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z')
PID_NAMESPACE = re.compile(r'pid:\[[1-9][0-9]*\]\Z')


class IdentityUnverified(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise IdentityUnverified(message)


def validate_expected(expected):
    keys = {'boot_id', 'pid_namespace', 'pid', 'start_ticks', 'argv'}
    require(isinstance(expected, dict) and set(expected) == keys, 'Incomplete expected process identity')
    require(isinstance(expected['boot_id'], str) and BOOT_ID.fullmatch(expected['boot_id']), 'Invalid boot id')
    require(isinstance(expected['pid_namespace'], str) and PID_NAMESPACE.fullmatch(expected['pid_namespace']),
            'Invalid PID namespace')
    require(type(expected['pid']) is int and 0 < expected['pid'] <= 2**31 - 1, 'Invalid PID')
    require(type(expected['start_ticks']) is int and expected['start_ticks'] > 0, 'Invalid start ticks')
    argv = expected['argv']
    require(isinstance(argv, list) and 0 < len(argv) <= 128 and
            all(isinstance(x, str) and 0 < len(x) <= 4096 and '\x00' not in x for x in argv), 'Invalid argv')


def read_bounded(sftp, path, limit):
    with sftp.open(path, 'rb') as stream:
        data = stream.read(limit + 1)
    require(isinstance(data, bytes) and 0 < len(data) <= limit, 'Invalid proc file')
    return data


def read_boot_id(sftp):
    try:
        value = read_bounded(sftp, '/proc/sys/kernel/random/boot_id', 128).decode('ascii').strip()
    except UnicodeDecodeError as exc:
        raise IdentityUnverified('Invalid boot id encoding') from exc
    require(BOOT_ID.fullmatch(value), 'Invalid observed boot id')
    return value


def read_namespace(sftp):
    value = sftp.readlink('/proc/self/ns/pid')
    require(isinstance(value, str) and PID_NAMESPACE.fullmatch(value), 'Invalid observed PID namespace')
    return value


def read_stat(sftp, pid):
    data = read_bounded(sftp, f'/proc/{pid}/stat', 16384)
    require(data.startswith((str(pid) + ' (').encode('ascii')) and b')' in data, 'Invalid proc stat prefix')
    tail = data.rsplit(b')', 1)[1].split()
    require(len(tail) >= 20 and len(tail[0]) == 1, 'Invalid proc stat fields')
    require(tail[0] in (b'R', b'S', b'D', b'T', b't', b'K', b'W', b'P', b'I'), 'Process is not live')
    require(tail[19].isdigit() and int(tail[19]) > 0, 'Invalid process start ticks')
    return int(tail[19])


def read_argv(sftp, pid):
    data = read_bounded(sftp, f'/proc/{pid}/cmdline', 32768)
    require(data.endswith(b'\x00'), 'Invalid proc cmdline terminator')
    try:
        argv = [item.decode('utf-8') for item in data[:-1].split(b'\x00')]
    except UnicodeDecodeError as exc:
        raise IdentityUnverified('Invalid proc cmdline encoding') from exc
    require(all(argv), 'Empty proc cmdline argument')
    return argv


def verify(sftp, expected):
    """Verify boot, namespace, PID/start ticks and argv twice around a read.

    Network/proc errors are deliberately surfaced as unverified identity. The
    result is a point-in-time observation, not a completion or persistence proof.
    """
    validate_expected(expected)
    try:
        boot_before = read_boot_id(sftp)
        namespace_before = read_namespace(sftp)
        start_before = read_stat(sftp, expected['pid'])
        argv = read_argv(sftp, expected['pid'])
        start_after = read_stat(sftp, expected['pid'])
        namespace_after = read_namespace(sftp)
        boot_after = read_boot_id(sftp)
    except IdentityUnverified:
        raise
    except Exception as exc:
        raise IdentityUnverified('Process identity could not be observed') from exc
    require(boot_before == boot_after == expected['boot_id'], 'Boot identity mismatch')
    require(namespace_before == namespace_after == expected['pid_namespace'], 'PID namespace mismatch')
    require(start_before == start_after == expected['start_ticks'], 'PID start-time mismatch')
    require(argv == expected['argv'], 'Process command mismatch')
    return {'status': 'LIVE_IDENTITY_MATCHED_AT_OBSERVATION', 'pid': expected['pid'],
            'start_ticks': start_before, 'boot_id': boot_before, 'pid_namespace': namespace_before}
