"""Test del wrapper Python di `sast-engine slice`. Il motore è uno script `sh` finto: nessuna compilazione Rust, nessuna rete."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import slice_io

RECORD_A = {"sink": {"function": "f", "file": "a.c", "line": 3, "cwe_id": "CWE-120"}, "blocks": [], "truncated": False}
RECORD_B = {"sink": {"function": "g", "file": "b.c", "line": 7, "cwe_id": "CWE-78"}, "blocks": [], "truncated": True}

# Registra in `targets_seen.path` il percorso passato a `--targets` e ne copia il contenuto in `targets_seen.jsonl`
# (accanto allo script), così i test possono controllare cosa ha scritto il wrapper.
RECORD_TARGETS = """\
prev=""
for a in "$@"; do
  if [ "$prev" = "--targets" ]; then
    printf '%s' "$a" > "$(dirname "$0")/targets_seen.path"
    cp "$a" "$(dirname "$0")/targets_seen.jsonl"
  fi
  prev="$a"
done
"""

OK_BODY = RECORD_TARGETS + 'echo "$@" >&2\n' + "".join(f"echo '{json.dumps(r)}'\n" for r in (RECORD_A, RECORD_B))
FAIL_BODY = RECORD_TARGETS + 'echo "boom: grafo non costruibile" >&2\nexit 3\n'


def fake_engine(directory: Path, body: str) -> Path:
    path = directory / "fake-engine"
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


class RunSliceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_run_slice_passes_defaults_and_parses_jsonl(self):
        engine = fake_engine(self.dir, OK_BODY)
        records, stderr = slice_io.run_slice(self.dir, binary=engine)
        self.assertEqual(records, [RECORD_A, RECORD_B])
        self.assertIn("--depth 5 --callers 4 --max-chars 12000", stderr)
        self.assertTrue(stderr.startswith(f"slice {self.dir} "), "il sottocomando e la radice vengono prima delle opzioni")
        self.assertNotIn("--targets", stderr, "senza target il motore usa i sink della KB")

    def test_records_are_split_on_newline_only(self):
        """serde_json non escapa U+2028/U+2029/U+0085 nel sorgente grezzo: `str.splitlines()` spezzerebbe il record."""
        code = "// riga con\u0085separatori unicode"
        raw = dict(RECORD_A, blocks=[{"code": code}])
        (self.dir / "stdout.jsonl").write_text(json.dumps(raw, ensure_ascii=False) + "\n" + json.dumps(RECORD_B) + "\n", encoding="utf-8")
        engine = fake_engine(self.dir, 'cat "$(dirname "$0")/stdout.jsonl"\n')
        records, _ = slice_io.run_slice(self.dir, binary=engine)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["blocks"][0]["code"], code)
        self.assertEqual(records[1], RECORD_B)

    def test_custom_limits_are_passed(self):
        engine = fake_engine(self.dir, OK_BODY)
        _, stderr = slice_io.run_slice(self.dir, depth=2, callers=1, max_chars=500, binary=engine)
        self.assertIn("--depth 2 --callers 1 --max-chars 500", stderr)

    def test_targets_are_written_as_jsonl_and_passed(self):
        targets = [
            {"file": "a.c", "line": 3, "callee": "strcpy", "cwe_id": "CWE-120"},
            {"file": "b.c", "line": 7, "callee": "system", "cwe_id": "CWE-78"},
        ]
        engine = fake_engine(self.dir, OK_BODY)
        _, stderr = slice_io.run_slice(self.dir, targets, binary=engine)
        self.assertIn("--targets", stderr)
        lines = (self.dir / "targets_seen.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual([json.loads(line) for line in lines], targets)

    def test_temporary_targets_file_is_removed_after_success(self):
        engine = fake_engine(self.dir, OK_BODY)
        slice_io.run_slice(self.dir, [{"file": "a.c", "line": 3, "callee": "strcpy", "cwe_id": "CWE-120"}], binary=engine)
        seen = Path((self.dir / "targets_seen.path").read_text(encoding="utf-8"))
        self.assertFalse(seen.exists())

    def test_empty_target_list_is_still_passed(self):
        """`[]` significa «nessun sink», diverso da `None` («usa il catalogo della KB»)."""
        engine = fake_engine(self.dir, OK_BODY)
        _, stderr = slice_io.run_slice(self.dir, [], binary=engine)
        self.assertIn("--targets", stderr)
        self.assertEqual((self.dir / "targets_seen.jsonl").read_text(encoding="utf-8"), "")

    def test_nonzero_exit_raises_runtime_error(self):
        engine = fake_engine(self.dir, FAIL_BODY)
        with self.assertRaises(RuntimeError) as ctx:
            slice_io.run_slice(self.dir, binary=engine)
        self.assertIn("3", str(ctx.exception))
        self.assertIn("boom: grafo non costruibile", str(ctx.exception), "lo stderr del motore deve arrivare nell'errore")

    def test_temporary_targets_file_is_removed_after_error(self):
        engine = fake_engine(self.dir, FAIL_BODY)
        with self.assertRaises(RuntimeError):
            slice_io.run_slice(self.dir, [{"file": "a.c", "line": 3, "callee": "strcpy", "cwe_id": "CWE-120"}], binary=engine)
        seen = Path((self.dir / "targets_seen.path").read_text(encoding="utf-8"))
        self.assertFalse(seen.exists())

    def test_missing_binary_raises_runtime_error_with_hint(self):
        with mock.patch.object(slice_io, "engine_bin", return_value=None):
            with self.assertRaises(RuntimeError) as ctx:
                slice_io.run_slice(self.dir)
        self.assertIn("SAST_ENGINE_BIN", str(ctx.exception))
        self.assertIn("cargo build --release", str(ctx.exception))


class EngineBinTests(unittest.TestCase):
    def test_engine_bin_reads_environment(self):
        with tempfile.TemporaryDirectory() as d:
            engine = fake_engine(Path(d), "exit 0\n")
            with mock.patch.dict(os.environ, {"SAST_ENGINE_BIN": str(engine)}):
                self.assertEqual(slice_io.engine_bin(), engine)

    def test_engine_bin_ignores_environment_pointing_nowhere(self):
        """Una variabile esplicita ma sbagliata non ricade in silenzio sul binario di default."""
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {"SAST_ENGINE_BIN": str(Path(d) / "non-esiste")}):
                self.assertIsNone(slice_io.engine_bin())

    def test_engine_bin_defaults_to_release_build_next_to_the_repo(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("SAST_ENGINE_BIN", None)
            found = slice_io.engine_bin()
        expected = Path(slice_io.__file__).resolve().parents[1] / "sast-engine" / "target" / "release" / "sast-engine"
        self.assertEqual(found, expected if expected.is_file() else None)


class WriteTargetsTests(unittest.TestCase):
    def test_one_json_object_per_line_with_unicode_preserved(self):
        targets = [{"file": "è.c", "line": 1, "callee": "strcpy", "cwe_id": "CWE-120"}, {"file": "b.c", "line": 2, "callee": "gets", "cwe_id": "CWE-120"}]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "t.jsonl"
            slice_io.write_targets(path, targets)
            text = path.read_text(encoding="utf-8")
        self.assertTrue(text.endswith("\n"))
        self.assertIn("è.c", text, "niente escape \\u: il file resta leggibile")
        self.assertEqual([json.loads(line) for line in text.splitlines()], targets)


if __name__ == "__main__":
    unittest.main()
