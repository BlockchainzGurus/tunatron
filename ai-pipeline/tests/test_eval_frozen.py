"""Test di `eval_frozen`: validazione della lista congelata, punteggi S3/S4 e candidati dal motore (finto)."""
import json
import tempfile
import unittest
from pathlib import Path

import eval_frozen as ef

FAKE_ENGINE = """#!/bin/sh
d="$(dirname "$0")"
if [ -f "$d/records.jsonl" ]; then cat "$d/records.jsonl"; fi
exit 0
"""


def entry(**kw) -> dict:
    base = {"file": "app/views.py", "line": 10, "callee": "redirect", "cwe_id": "CWE-601", "expected": "safe", "evidence": "costante"}
    return {**base, **kw}


class LoadFrozenTests(unittest.TestCase):
    def write(self, data) -> Path:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        path = Path(self._tmp.name) / "frozen.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_valid_file_is_loaded_as_is(self):
        data = {"repo": "pygoat", "commit": "abc", "entries": [entry(), entry(line=11, expected="vulnerable")]}
        self.assertEqual(ef.load_frozen(self.write(data)), data)

    def test_load_frozen_rejects_missing_field_and_bad_expected(self):
        for broken in (entry(expected="unsure"), {k: v for k, v in entry().items() if k != "evidence"},
                       entry(line="10"), entry(evidence="")):
            with self.assertRaises(ValueError, msg=broken):
                ef.load_frozen(self.write({"repo": "r", "commit": "c", "entries": [broken]}))
        for incomplete in ({"commit": "c", "entries": []}, {"repo": "r", "entries": []}, {"repo": "r", "commit": "c"},
                           {"repo": "r", "commit": "c", "entries": "x"}, ["not", "an", "object"]):
            with self.assertRaises(ValueError, msg=incomplete):
                ef.load_frozen(self.write(incomplete))

    def test_duplicate_entries_are_rejected(self):
        with self.assertRaises(ValueError):
            ef.load_frozen(self.write({"repo": "r", "commit": "c", "entries": [entry(), entry()]}))


class ScoreFrozenTests(unittest.TestCase):
    def test_score_frozen_counts_fp_tp_truncation_and_cve(self):
        frozen = {"repo": "r", "commit": "c", "entries": [
            entry(line=1, expected="safe"), entry(line=2, expected="safe"), entry(line=3, expected="safe"),
            entry(line=4, expected="vulnerable"), entry(line=5, expected="vulnerable"), entry(line=6, expected="vulnerable")]}
        res = lambda v, t=False, r="": {"vulnerable": v, "truncated": t, "reasoning": r}  # noqa: E731
        results = [res(True), res(False), res(None, True),
                   res(True, r="Like CVE-2021-1234 and CVE-2019-99999"), res(True), res(False, True, "CVE-2020-1")]
        score = ef.score_frozen(frozen, results)
        self.assertEqual(score["fp_still_affected"], round(1 / 3, 4))
        self.assertEqual(score["tp_recall"], round(2 / 3, 4))
        self.assertEqual(score["truncated_rate"], round(2 / 6, 4))
        self.assertEqual(score["cve_mentions"], 3)
        self.assertEqual((score["n"], score["safe"], score["vulnerable"], score["invalid"]), (6, 3, 3, 1))

    def test_score_is_broken_down_by_cwe_because_one_class_can_dominate_the_list(self):
        entries = [entry(line=1, cwe_id="CWE-601", expected="safe"), entry(line=2, cwe_id="CWE-601", expected="safe"),
                   entry(line=3, cwe_id="CWE-78", expected="safe"), entry(line=4, cwe_id="CWE-78", expected="vulnerable")]
        res = lambda v: {"vulnerable": v, "truncated": False, "reasoning": ""}  # noqa: E731
        score = ef.score_frozen({"repo": "r", "commit": "c", "entries": entries}, [res(True), res(False), res(True), res(True)])
        self.assertEqual(score["by_cwe"], {"CWE-601": {"safe": 2, "still_affected": 1, "vulnerable": 0, "recalled": 0},
                                           "CWE-78": {"safe": 1, "still_affected": 1, "vulnerable": 1, "recalled": 1}})

    def test_a_missing_answer_is_neither_a_false_positive_nor_a_hit(self):
        frozen = {"repo": "r", "commit": "c", "entries": [entry(expected="safe"), entry(line=2, expected="vulnerable")]}
        score = ef.score_frozen(frozen, [{"vulnerable": None, "truncated": False, "reasoning": ""}] * 2)
        self.assertEqual((score["fp_still_affected"], score["tp_recall"]), (0.0, 0.0))

    def test_classes_with_no_entries_do_not_divide_by_zero(self):
        score = ef.score_frozen({"repo": "r", "commit": "c", "entries": [entry()]}, [{"vulnerable": True, "truncated": False, "reasoning": ""}])
        self.assertEqual((score["fp_still_affected"], score["tp_recall"]), (1.0, 0.0))

    def test_results_must_match_the_entries(self):
        with self.assertRaises(ValueError):
            ef.score_frozen({"repo": "r", "commit": "c", "entries": [entry()]}, [])


class CandidatesTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.repo = self.dir / "repo"
        (self.repo / "app").mkdir(parents=True)
        (self.repo / "app" / "views.py").write_text("\n".join(f"line{i}" for i in range(1, 11)) + "\n", encoding="utf-8")
        engine_dir = self.dir / "engine"
        engine_dir.mkdir()
        records = [self.record("app/views.py", 5, "os.system"), self.record("app/views.py", 1, "redirect", "CWE-601"),
                   self.record("../outside.py", 2, "eval", "CWE-94"), self.record("app/views.py", 3, "eval", "CWE-94")]
        (engine_dir / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        self.engine = engine_dir / "fake-engine"
        self.engine.write_text(FAKE_ENGINE, encoding="utf-8")
        self.engine.chmod(0o755)

    @staticmethod
    def record(file: str, line: int, callee: str, cwe: str = "CWE-78") -> dict:
        return {"sink": {"function": callee, "file": file, "line": line, "cwe_id": cwe}, "blocks": [], "truncated": False}

    def test_candidates_use_the_engine_and_add_an_excerpt(self):
        found = ef.candidates(self.repo, self.engine)
        self.assertEqual([(c["file"], c["line"], c["callee"], c["cwe_id"]) for c in found],
                         [("../outside.py", 2, "eval", "CWE-94"), ("app/views.py", 1, "redirect", "CWE-601"),
                          ("app/views.py", 3, "eval", "CWE-94"), ("app/views.py", 5, "os.system", "CWE-78")],
                         "ordinati per file, riga, callee")
        by_line = {c["line"]: c for c in found if c["file"] == "app/views.py"}
        self.assertEqual(by_line[5]["excerpt"].split("\n"), ["   3 | line3", "   4 | line4", "   5 | line5", "   6 | line6", "   7 | line7"])
        self.assertEqual(by_line[1]["excerpt"].split("\n")[0], "   1 | line1", "all'inizio del file l'estratto si accorcia")

    def test_a_sink_path_that_leaves_the_repo_gets_no_excerpt_and_no_error(self):
        outside = next(c for c in ef.candidates(self.repo, self.engine) if c["file"].startswith(".."))
        self.assertEqual(outside["excerpt"], "")


if __name__ == "__main__":
    unittest.main()
