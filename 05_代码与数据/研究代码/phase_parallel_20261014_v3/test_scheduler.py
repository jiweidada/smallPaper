"""No CUDA: exercise slot accounting and noninterrupting live-process handoff."""
import json
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from run_parallel_queue import adopted_returncode, identity, same_process, signal_exact, slots_available


class SchedulerTests(unittest.TestCase):
    def test_legacy_training_occupies_a_slot(self):
        self.assertEqual(slots_available(0, 1), 1)
        self.assertEqual(slots_available(1, 1), 0)
        self.assertEqual(slots_available(2, 0), 0)
        self.assertEqual(slots_available(0, 0), 2)

    @unittest.skipUnless(sys.platform == 'linux', 'Linux process ownership check')
    def test_frozen_parent_does_not_freeze_child_and_exit_is_verifiable(self):
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder)
            child_source = "import pathlib,time; p=pathlib.Path('progress'); q=p.with_suffix('.tmp');\nfor n in range(12): q.write_text(str(n)); q.replace(p); time.sleep(.1)\nraise SystemExit(7)\n"
            parent_source = "import subprocess,sys,pathlib,time,json; p=subprocess.Popen([sys.executable,'-c',"+repr(child_source)+"]); pathlib.Path('pid').write_text(str(p.pid)); time.sleep(10); p.wait()"
            parent = subprocess.Popen([sys.executable, '-c', parent_source], cwd=path)
            try:
                deadline = time.time()+5
                while not ((path/'progress').exists() and (path/'pid').exists()):
                    if time.time()>deadline:
                        self.fail('Child did not start')
                    time.sleep(.02)
                controller = identity(parent.pid)
                child = identity(int((path/'pid').read_text()))
                signal_exact(controller, signal.SIGSTOP)
                first = int((path/'progress').read_text())
                time.sleep(.4)
                self.assertGreater(int((path/'progress').read_text()), first)
                while same_process(child)['state'] != 'Z':
                    if time.time()>deadline:
                        self.fail('Child did not exit while parent was held')
                    time.sleep(.02)
                self.assertEqual(adopted_returncode(child), 7)
                wrong = dict(controller, start_ticks=controller['start_ticks']+1)
                with self.assertRaises(RuntimeError):
                    signal_exact(wrong, signal.SIGTERM)
                self.assertEqual(same_process(controller)['state'], 'T')
                signal_exact(controller, signal.SIGTERM)
                signal_exact(controller, signal.SIGCONT)
                self.assertEqual(parent.wait(timeout=3), -signal.SIGTERM)
            finally:
                if parent.poll() is None:
                    parent.kill(); parent.wait()


if __name__ == '__main__':
    unittest.main(verbosity=2)
