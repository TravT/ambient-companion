#!/usr/bin/env python3
"""Unit tests for config.py, daemon.py helpers, ingestion guards and housekeeping.

No hardware needed: every adb/HTTP call is mocked.
"""

import importlib
import os
import shlex
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

PKG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_DIR))

import config
import daemon
import housekeeping
import optical_ingestion


class TestConfig(unittest.TestCase):
    def test_defaults_match_original_host_behaviour(self):
        self.assertEqual(config.DEVICE_TARGET, "100.115.165.41:5555")
        self.assertEqual(config.LLAMA_SERVER_URL, "http://127.0.0.1:8085/v1/chat/completions")
        self.assertEqual(config.VLM_MODEL, "qwen2.5vl:3b")

    def test_env_overrides_data_dir(self):
        with mock.patch.dict(os.environ, {"AMBIENT_DATA_DIR": "/data"}):
            reloaded = importlib.reload(config)
            try:
                self.assertEqual(reloaded.BENCHMARK_DIR, Path("/data/media/merged/vision/benchmark"))
                self.assertEqual(reloaded.DROPZONE_DIR, Path("/data/dropzone/files"))
            finally:
                os.environ.pop("AMBIENT_DATA_DIR", None)
                importlib.reload(config)

    def test_allowed_dirs_unset_means_unrestricted(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AMBIENT_ALLOWED_DIRS", None)
            self.assertIsNone(config.allowed_dirs())

    def test_allowed_dirs_parses_colon_list(self):
        with mock.patch.dict(os.environ, {"AMBIENT_ALLOWED_DIRS": "/tmp:/var"}):
            dirs = config.allowed_dirs()
        self.assertEqual([p.name for p in dirs], ["tmp", "var"])

    def test_adb_base_uses_gateway_and_target(self):
        base = config.adb_base()
        self.assertEqual(base[:2], ["adb", "-H"])
        self.assertIn(config.DEVICE_TARGET, base)


class TestCritiqueRule(unittest.TestCase):
    def test_plain_yes_is_confident(self):
        self.assertTrue(daemon.critique_is_confident("Yes."))
        self.assertTrue(daemon.critique_is_confident("SIM"))

    def test_no_and_unclear_are_not_confident(self):
        for text in ("No.", "NO", "unclear", "Yes, but unclear", "Não"):
            self.assertFalse(daemon.critique_is_confident(text), text)

    def test_substring_hits_do_not_count_as_no(self):
        # "NOTE" and "KNOWN" contain "NO" but are not a negative answer.
        self.assertTrue(daemon.critique_is_confident("Yes. Noted and known."))

    def test_empty_is_not_confident(self):
        self.assertFalse(daemon.critique_is_confident(""))
        self.assertFalse(daemon.critique_is_confident("No response parsed"))


class TestShellSafety(unittest.TestCase):
    def test_smolvlm_prompt_round_trips_through_shell_parsing(self):
        nasty = "what's in the \"box\"? $(rm -rf /) `id` ; echo 'x'"
        cmd = daemon.build_smolvlm_command("frame.jpg", nasty, max_tokens=25)
        argv = shlex.split(cmd)
        prompt = argv[argv.index("-p") + 1]
        self.assertIn(nasty, prompt)
        self.assertTrue(prompt.startswith("User:<image>"))
        self.assertEqual(argv[argv.index("-n") + 1], "25")

    def test_remote_name_is_sanitised(self):
        self.assertEqual(daemon.safe_remote_name("a b;rm -rf $HOME.jpg"), "a_b_rm_-rf__HOME.jpg")
        self.assertEqual(daemon.safe_remote_name("../../etc/passwd"), ".._.._etc_passwd")

    def test_run_edge_command_quotes_inner_command(self):
        captured = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            return mock.Mock(stdout="ok", returncode=0)

        with mock.patch.object(daemon.subprocess, "run", fake_run):
            daemon.run_edge_command("echo 'hi there'")
        shell_string = captured["argv"][-1]
        # The device shell must see the original command intact.
        outer = shlex.split(shell_string)
        self.assertIn("echo 'hi there'", outer[-1])


class TestThermalFailClosed(unittest.TestCase):
    def test_temperature_none_when_adb_fails(self):
        with mock.patch.object(daemon.subprocess, "run", side_effect=OSError("no adb")):
            self.assertIsNone(daemon.get_edge_temperature())

    def test_temperature_none_on_garbage(self):
        with mock.patch.object(daemon.subprocess, "run", return_value=mock.Mock(stdout="", returncode=0)):
            self.assertIsNone(daemon.get_edge_temperature())

    def test_temperature_parses_decidegrees(self):
        with mock.patch.object(daemon.subprocess, "run", return_value=mock.Mock(stdout="312", returncode=0)):
            self.assertAlmostEqual(daemon.get_edge_temperature(), 31.2)

    def test_cycle_aborts_when_edge_unreachable(self):
        with mock.patch.object(daemon, "get_edge_temperature", return_value=None), \
                mock.patch.object(optical_ingestion, "acquire_image") as acquire:
            res = daemon.execute_ambient_cycle("what is on the desk?", play_audio=False)
        self.assertEqual(res["status"], "aborted_edge_unreachable")
        acquire.assert_not_called()

    def test_cycle_aborts_when_hot(self):
        with mock.patch.object(daemon, "get_edge_temperature", return_value=45.0):
            res = daemon.execute_ambient_cycle("x", play_audio=False)
        self.assertEqual(res["status"], "aborted_thermal")


class TestTts(unittest.TestCase):
    def test_missing_binary_reports_error_not_crash(self):
        with mock.patch.object(daemon, "POCKET_TTS_BIN", None):
            res = daemon.synthesize_speech("hello", "en", Path(tempfile.gettempdir()) / "x.wav")
        self.assertEqual(res["status"], "error")
        self.assertIn("pocket-tts", res["error"])


    def test_generate_subcommand_is_used(self):
        # pocket-tts 3.3.0 rejects `--text` without the `generate` subcommand.
        captured = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            return mock.Mock(stdout="", stderr="", returncode=1)

        with mock.patch.object(daemon, "POCKET_TTS_BIN", "/usr/bin/pocket-tts"), \
                mock.patch.object(daemon.subprocess, "run", fake_run):
            daemon.synthesize_speech("hello", "pt", Path(tempfile.gettempdir()) / "x.wav")
        self.assertEqual(captured["argv"][:2], ["/usr/bin/pocket-tts", "generate"])


class TestEdgeSpeech(unittest.TestCase):
    """Voice runs on the S20 FE: pocket-tts generate in Termux, then paplay."""

    def _edge(self, outputs):
        calls = []

        def fake_edge(cmd, timeout=45, as_root=False):
            calls.append(cmd)
            return outputs.pop(0), 0.5

        return calls, fake_edge

    def test_edge_mode_synthesizes_then_plays(self):
        calls, fake = self._edge(["SYNTH_OK", "", ""])
        with mock.patch.object(config, "TTS_MODE", "edge"), mock.patch.object(config, "TTS_WARM", False), \
                mock.patch.object(daemon, "run_edge_command", fake), \
                mock.patch.object(daemon.subprocess, "run", return_value=mock.Mock(returncode=0)):
            res = daemon.speak("It's a \"cup\"; $(id)", "en")
        self.assertEqual(res["status"], "success")
        self.assertTrue(res["played"])
        self.assertIn("pocket-tts generate", calls[0])
        self.assertIn("paplay", calls[1])
        argv = shlex.split(calls[0].split(" && ")[0])
        self.assertIn("It's a \"cup\"; $(id)", argv)          # text survives quoting intact
        self.assertTrue(argv[argv.index("--voice") + 1].endswith(".safetensors"))

    def test_edge_pt_uses_builtin_voice(self):
        calls, fake = self._edge(["SYNTH_OK", "", ""])
        with mock.patch.object(config, "TTS_MODE", "edge"), mock.patch.object(config, "TTS_WARM", False), \
                mock.patch.object(daemon, "run_edge_command", fake), \
                mock.patch.object(daemon.subprocess, "run", return_value=mock.Mock(returncode=0)):
            daemon.speak("ola", "pt")
        argv = shlex.split(calls[0].split(" && ")[0])
        self.assertEqual(argv[argv.index("--voice") + 1], "rafael")

    def test_edge_synthesis_failure_skips_playback(self):
        calls, fake = self._edge(["Traceback: boom"])
        with mock.patch.object(config, "TTS_MODE", "edge"), mock.patch.object(config, "TTS_WARM", False), \
                mock.patch.object(daemon, "run_edge_command", fake):
            res = daemon.speak("hello", "en")
        self.assertEqual(res["status"], "error")
        self.assertFalse(res["played"])
        self.assertEqual(len(calls), 1)

    def test_warm_server_path_posts_text_literally(self):
        # SERVER_UP, then the curl synthesis, then playback, then cleanup.
        calls, fake = self._edge(["UP", "SYNTH_OK", "", ""])
        nasty = "@/etc/passwd <x> $(id) 'q'"
        with mock.patch.object(config, "TTS_MODE", "edge"), mock.patch.object(config, "TTS_WARM", True), \
                mock.patch.object(daemon, "run_edge_command", fake), \
                mock.patch.object(daemon.subprocess, "run", return_value=mock.Mock(returncode=0)):
            res = daemon.speak(nasty, "en")
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["where"], "s20-warm")
        self.assertIn("pocket-tts serve", calls[0])          # started on demand when /health is down
        self.assertIn("--default-voice", calls[0])
        synth = shlex.split(calls[1])
        self.assertIn("--form-string", synth)                # never -F: curl would read "@file" / "<file"
        self.assertEqual(synth[synth.index("--form-string") + 1], "text=" + nasty)
        self.assertNotIn("voice_url", calls[1])               # English uses the server's default profile
        self.assertIn("paplay", calls[2])

    def test_warm_server_pt_selects_builtin_voice_per_request(self):
        calls, fake = self._edge(["UP", "SYNTH_OK", "", ""])
        with mock.patch.object(config, "TTS_MODE", "edge"), mock.patch.object(config, "TTS_WARM", True), \
                mock.patch.object(daemon, "run_edge_command", fake), \
                mock.patch.object(daemon.subprocess, "run", return_value=mock.Mock(returncode=0)):
            daemon.speak("ola", "pt")
        self.assertIn("voice_url=rafael", calls[1])

    def test_warm_failure_falls_back_to_cli(self):
        # server never comes up -> per-call CLI still speaks
        calls, fake = self._edge(["DOWN", "SYNTH_OK", "", ""])
        with mock.patch.object(config, "TTS_MODE", "edge"), mock.patch.object(config, "TTS_WARM", True), \
                mock.patch.object(daemon, "run_edge_command", fake), \
                mock.patch.object(daemon.subprocess, "run", return_value=mock.Mock(returncode=0)):
            res = daemon.speak("hello", "en")
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["where"], "s20")
        self.assertIn("pocket-tts generate", calls[1])

    def test_off_mode_does_nothing(self):
        with mock.patch.object(config, "TTS_MODE", "off"), \
                mock.patch.object(daemon, "run_edge_command", side_effect=AssertionError("touched edge")):
            res = daemon.speak("hello", "en")
        self.assertEqual(res["status"], "disabled")
        self.assertFalse(res["played"])

    def test_tts_available_edge_checks_binary_on_phone(self):
        with mock.patch.object(config, "TTS_MODE", "edge"), \
                mock.patch.object(daemon, "run_edge_command", return_value=("/data/x/usr/bin/pocket-tts", 0.1)):
            self.assertTrue(daemon.tts_available())
        with mock.patch.object(config, "TTS_MODE", "edge"), \
                mock.patch.object(daemon, "run_edge_command", return_value=("ERROR: adb", 0.1)):
            self.assertFalse(daemon.tts_available())
        with mock.patch.object(config, "TTS_MODE", "off"):
            self.assertFalse(daemon.tts_available())


class TestAllowedDirs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        (self.root / "ok").mkdir()
        self.inside = self.root / "ok" / "a.jpg"
        self.outside = self.root / "b.jpg"
        for p in (self.inside, self.outside):
            p.write_bytes(b"\xff\xd8\xff\xd9")

    def tearDown(self):
        self.tmp.cleanup()

    def test_inside_allowed_dir_is_returned(self):
        with mock.patch.object(config, "allowed_dirs", return_value=[self.root / "ok"]):
            self.assertEqual(optical_ingestion.acquire_image(str(self.inside)), self.inside)

    def test_outside_allowed_dir_is_refused(self):
        with mock.patch.object(config, "allowed_dirs", return_value=[self.root / "ok"]):
            with self.assertRaises(PermissionError):
                optical_ingestion.acquire_image(str(self.outside))

    def test_symlink_escape_is_refused(self):
        link = self.root / "ok" / "escape.jpg"
        link.symlink_to(self.outside)
        with mock.patch.object(config, "allowed_dirs", return_value=[self.root / "ok"]):
            with self.assertRaises(PermissionError):
                optical_ingestion.acquire_image(str(link))

    def test_unrestricted_when_unset(self):
        with mock.patch.object(config, "allowed_dirs", return_value=None):
            self.assertEqual(optical_ingestion.acquire_image(str(self.outside)), self.outside)


class TestHousekeeping(unittest.TestCase):
    def test_purge_local_removes_only_old_matching_files(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            old_match = root / "daemon_edge_384px_a.jpg"
            new_match = root / "daemon_edge_384px_b.jpg"
            old_other = root / "keep_me.jpg"
            for p in (old_match, new_match, old_other):
                p.write_bytes(b"x")
            old = time.time() - 10 * 86400
            os.utime(old_match, (old, old))
            os.utime(old_other, (old, old))

            removed = housekeeping.purge_local(root, ["daemon_*"], max_age_days=7)

            self.assertEqual(removed, 1)
            self.assertFalse(old_match.exists())
            self.assertTrue(new_match.exists())
            self.assertTrue(old_other.exists())

    def test_purge_local_missing_dir_is_noop(self):
        self.assertEqual(housekeeping.purge_local(Path("/nonexistent/zzz"), ["*"], 1), 0)

    def test_purge_edge_runs_find_delete_with_quoted_patterns(self):
        captured = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            return mock.Mock(stdout="", returncode=0)

        with mock.patch.object(housekeeping.subprocess, "run", fake_run):
            ok = housekeeping.purge_edge(max_age_days=7)
        self.assertTrue(ok)
        shell_string = captured["argv"][-1]
        self.assertIn("find /sdcard/Download", shell_string)
        self.assertIn("-mtime +7", shell_string)
        self.assertIn("-delete", shell_string)


if __name__ == "__main__":
    unittest.main()
