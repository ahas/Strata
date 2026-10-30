"""serve/test_stop_session.py - --stop-session: the live session saved when the server stops and restored when it
starts (mock and fake engines: no GPU, no pack).

    python -m unittest serve.test_stop_session -v
"""
from __future__ import annotations

import io
import json
import os
import signal
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import serve.server as S  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, Service, StrataEngine, stop_session_name  # noqa: E402
from serve.test_slots import FakeEngine, FakeProc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class SaveAndRestore(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.engine = FakeEngine()
        self.engine.silence_s = 300.0
        self.svc = Service(self.engine, ByteTokenizer(), ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.slot_save_path = self.dir.name
        self.svc.stop_session = "stop.bin"
        self.path = Path(self.dir.name) / "stop.bin"

    def test_a_stop_saves_the_live_session_into_the_slot_save_path(self):
        said = self.svc.save_stop_session()
        self.assertEqual(self.engine.proc.sent, [f"SAVE {self.path}\n"])
        self.assertTrue(self.path.exists())
        self.assertIn("saved the session for the next start: 62993 tokens", said)
        self.assertTrue(self.svc.draining, "no request is admitted once the stop has begun")

    def test_a_start_restores_what_the_last_stop_saved(self):
        self.svc.save_stop_session()
        self.engine.proc = FakeProc(self.engine)
        said = self.svc.restore_stop_session()
        self.assertEqual(self.engine.proc.sent, [f"RESTORE {self.path}\n"])
        self.assertIn("restored the session saved at the last stop: 62993 tokens", said)
        self.assertTrue(self.path.exists(), "the file stays until the next stop replaces it")

    def test_a_start_with_nothing_saved_sends_nothing(self):
        said = self.svc.restore_stop_session()
        self.assertEqual(self.engine.proc.sent, [])
        self.assertIn("no session saved at the last stop", said)

    def test_a_start_without_the_model_loaded_does_not_restore(self):
        self.path.write_bytes(b"x")
        self.engine.ended = True
        said = self.svc.restore_stop_session()
        self.assertEqual(self.engine.proc.sent, [])
        self.assertIn("not loaded", said)

    def test_a_stop_with_nothing_to_save_removes_the_earlier_file(self):
        self.path.write_bytes(b"an older conversation")
        self.engine.fail = True
        said = self.svc.save_stop_session()
        self.assertFalse(self.path.exists(), "a start must not restore a session older than the slot's last")
        self.assertIn("no complete session to save", said)
        self.assertIn("removed", said)

    def test_a_failure_after_the_new_file_replaced_the_old_keeps_it(self):
        self.path.write_bytes(b"the new file")
        self.engine.script = ["SERR io 1 flushing the folder failed\n"]
        said = self.svc.save_stop_session()
        self.assertTrue(self.path.exists())
        self.assertIn("written but its save failed", said)

    def test_a_dead_engine_is_said_not_raised(self):
        self.engine.script = [None]
        said = self.svc.save_stop_session()
        self.assertIn("no session saved for the next start", said)

    def test_off_without_a_name(self):
        self.svc.stop_session = None
        self.assertEqual(self.svc.save_stop_session(), "")
        self.assertEqual(self.svc.restore_stop_session(), "")
        self.assertEqual(self.engine.proc.sent, [])

    def test_a_running_request_is_cancelled_and_its_session_saved(self):
        self.assertTrue(self.svc.reserve_inference())
        cancel = threading.Event()
        with self.svc.status_lock:
            self.svc.active_cancels.add(cancel)
        self.svc.fifo.acquire()

        def request():                                  # ends at its next step once cancelled, as a generation does
            cancel.wait(10)
            self.svc.fifo.release()
            with self.svc.status_lock:
                self.svc.active_cancels.discard(cancel)
            self.svc.release_inference()
        t = threading.Thread(target=request)
        t.start()
        said = self.svc.save_stop_session(wait_s=10)
        t.join(5)
        self.assertTrue(cancel.is_set())
        self.assertIn("saved the session for the next start", said)
        self.assertEqual(self.engine.proc.sent, [f"SAVE {self.path}\n"])

    def test_a_request_that_never_ends_is_not_waited_for_beyond_the_limit(self):
        self.assertTrue(self.svc.reserve_inference())
        self.addCleanup(self.svc.release_inference)
        t0 = time.monotonic()
        said = self.svc.save_stop_session(wait_s=0.3)
        self.assertLess(time.monotonic() - t0, 5)
        self.assertIn("not saved", said)
        self.assertEqual(self.engine.proc.sent, [])


class TheName(unittest.TestCase):
    def test_off_when_absent_or_empty(self):
        self.assertIsNone(stop_session_name(None, False))
        self.assertIsNone(stop_session_name("", True))

    def test_a_plain_file_name_with_a_slot_save_path(self):
        self.assertEqual(stop_session_name("stop.bin", True), "stop.bin")

    def test_refused(self):
        for value, has_dir in (("stop.bin", False), ("../stop.bin", True), (".stop", True), (3, True)):
            with self.assertRaises(ValueError, msg=repr(value)):
                stop_session_name(value, has_dir)


class _Reached(Exception):
    pass


class ThroughMain(unittest.TestCase):
    """main() reads the name before the engine starts and restores before it serves."""

    def run_main(self, cfg, argv=()):
        seen = {}

        def fake_serve(svc, host=None, port=None):
            seen["svc"] = svc
            seen["ignores_term"] = S.ENGINE_IGNORES_TERM
            raise _Reached

        out = io.StringIO()
        with tempfile.TemporaryDirectory() as d, mock.patch.object(S, "serve", fake_serve), \
                mock.patch.object(S, "ENGINE_IGNORES_TERM", False), mock.patch.object(sys, "stdout", new=out):
            cfg = {k: (v.replace("{dir}", d) if isinstance(v, str) else v) for k, v in cfg.items()}
            p = Path(d) / "cfg.json"
            p.write_text(json.dumps(cfg), encoding="utf-8")
            with mock.patch.object(sys, "argv", ["server.py", "--engine", "mock", "--port", "0", "--config", str(p),
                                                 *argv]):
                try:
                    code = S.main()
                except _Reached:
                    code = 0
                except SystemExit as e:
                    code = e.code
        return code, seen, out.getvalue()

    def test_the_config_name_reaches_the_service_and_the_engine_spawn(self):
        code, seen, out = self.run_main({"slot_save_path": "{dir}/slots", "stop_session": "stop.bin"})
        self.assertEqual(code, 0)
        self.assertEqual(seen["svc"].stop_session, "stop.bin")
        self.assertTrue(seen["ignores_term"])
        self.assertIn("no session saved at the last stop (stop.bin)", out)

    def test_the_flag_names_it_too(self):
        code, seen, _ = self.run_main({"slot_save_path": "{dir}/slots"}, ["--stop-session", "s.bin"])
        self.assertEqual((code, seen["svc"].stop_session), (0, "s.bin"))

    def test_off_by_default(self):
        code, seen, _ = self.run_main({"slot_save_path": "{dir}/slots"})
        self.assertEqual(code, 0)
        self.assertIsNone(seen["svc"].stop_session)
        self.assertFalse(seen["ignores_term"])

    def test_refused_without_a_slot_save_path(self):
        code, seen, _ = self.run_main({"stop_session": "stop.bin"})
        self.assertNotEqual(code, 0)
        self.assertNotIn("svc", seen)


@unittest.skipIf(os.name == "nt", "SIGTERM is POSIX")
class AStoppedServer(unittest.TestCase):
    """The whole path: the server started, a SIGTERM as systemd sends it, the save attempted before the engine is
    closed, and the server ended cleanly (the mock engine keeps no session, so the save is refused and said)."""

    def test_sigterm_attempts_the_save_before_the_engine_closes(self):
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "cfg.json"
            cfg.write_text(json.dumps({"slot_save_path": f"{d}/slots", "stop_session": "stop.bin"}), encoding="utf-8")
            (Path(d) / "slots").mkdir(mode=0o700)
            (Path(d) / "slots" / "stop.bin").write_bytes(b"an older conversation")
            proc = subprocess.Popen([sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "mock", "--port", "0",
                                     "--config", str(cfg)], cwd=str(ROOT), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)
            try:
                lines = []
                for line in proc.stdout:
                    lines.append(line)
                    if line.startswith("ready:"):
                        break
                proc.send_signal(signal.SIGTERM)
                out = "".join(lines) + proc.stdout.read()
                self.assertEqual(proc.wait(timeout=30), 0, out)
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
                proc.stdout.close()
            self.assertIn("the session saved at the last stop is not restored", out)
            self.assertIn("no session saved for the next start", out)
            self.assertLess(out.index("no session saved for the next start"), out.index("[strata] stopped"))
            self.assertFalse((Path(d) / "slots" / "stop.bin").exists(), out)


FAKE = r'''
import sys
print('READY 4096 stop', flush=True)
for line in sys.stdin:
    if line.startswith('QUIT'):
        break
'''


@unittest.skipIf(os.name == "nt", "SIGTERM is POSIX")
class TheEngineOutlivesAStopSentToTheUnit(unittest.TestCase):
    """systemd stops a unit by sending SIGTERM to every process in it: with --stop-session the engine ignores it, so
    the server can still save the live session before it ends the engine itself."""

    def engine(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        script = Path(d.name) / "fake-engine"
        script.write_text("#!" + sys.executable + "\n" + FAKE)
        script.chmod(0o700)
        e = StrataEngine(str(script), [])
        self.addCleanup(e.close)
        return e

    def term(self, e):
        os.kill(e.proc.pid, signal.SIGTERM)
        end = time.monotonic() + 2
        while e.proc.poll() is None and time.monotonic() < end:
            time.sleep(0.05)
        return e.proc.poll()

    def test_ignored_with_a_stop_session(self):
        with mock.patch.object(S, "ENGINE_IGNORES_TERM", True):
            e = self.engine()
        self.assertIsNone(self.term(e), "the engine must outlive a SIGTERM sent to the whole unit")
        e.close()
        self.assertIsNone(e.proc, "the server still ends it (QUIT)")

    def test_ignored_when_spawned_from_another_thread(self):
        """A restart spawns the engine from a request's thread."""
        box = {}
        with mock.patch.object(S, "ENGINE_IGNORES_TERM", True):
            t = threading.Thread(target=lambda: box.update(e=self.engine()))
            t.start()
            t.join(30)
        self.assertIsNone(self.term(box["e"]))

    def test_default_unchanged(self):
        e = self.engine()
        self.assertIsNotNone(self.term(e), "without --stop-session SIGTERM ends the engine as before")


if __name__ == "__main__":
    unittest.main()
