"""Test di `pyvul_pilot` (traccia P2): niente rete, repo git locali creati in cartelle temporanee.

`fetch_commit` si prova con un repo vero `file://` (due commit: vulnerabile e corretto); l'orchestrazione sostituisce solo
`slice_io.run_slice` con una slice costruita dal checkout, così si vede che il commit giusto è nella cartella giusta.
"""
import contextlib
import io
import json
import os
import re
import subprocess
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import common
import graph_examples
import localize
import prepare_data
import pyvul_pilot as pp
import slice_io

KB = common.load_kb()

BEFORE = '''import os


def run(cmd):
    os.system("ls " + cmd)


def handler(request):
    return run(request.args.get("c"))
'''
AFTER = BEFORE.replace('os.system("ls " + cmd)', 'os.system("ls " + shlex.quote(cmd))').replace("import os\n", "import os\nimport shlex\n")
FUNC_BEFORE = 'def run(cmd):\n    os.system("ls " + cmd)'
FUNC_AFTER = 'def run(cmd):\n    os.system("ls " + shlex.quote(cmd))'


def git(cwd: Path, *args: str) -> str:
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}
    out = subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false", *args],
                         cwd=cwd, check=True, capture_output=True, text=True, env=env)
    return out.stdout.strip()


def make_repo(root: Path) -> tuple[str, str]:
    """Repo con due commit (vulnerabile, poi corretto): (sha del primo, sha del secondo)."""
    root.mkdir(parents=True)
    git(root, "init", "-q")
    git(root, "config", "uploadpack.allowAnySHA1InWant", "true")
    (root / "app.py").write_text(BEFORE, encoding="utf-8")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "first")
    first = git(root, "rev-parse", "HEAD")
    (root / "app.py").write_text(AFTER, encoding="utf-8")
    git(root, "commit", "-q", "-am", "second")
    return first, git(root, "rev-parse", "HEAD")


def row(repo: str, sha: str, name: str = "run", before: str = FUNC_BEFORE, after: str = FUNC_AFTER, **kw) -> dict:
    base = {"source_dataset": "secureIT-project/CVEfixes", "cve_id": "CVE-2020-0001", "cve_description": "cmd injection",
            "cwe_id": "CWE-78", "function_name": name, "code_before": before, "code_after": after, "commit_hash": sha,
            "repo_url": repo, "programming_language": "python"}
    return {**base, **kw}


class FindFunctionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "repo"
        (self.root / "pkg").mkdir(parents=True)
        (self.root / "pkg" / "mod.py").write_text("import os\n\n\nclass A:\n    def run(self, cmd):\n        os.system(cmd)\n\n\ndef other():\n    pass\n", encoding="utf-8")

    def test_find_function_matches_by_name_and_normalized_body(self):
        code = "    def run(self, cmd):\n            os.system( cmd )\n"  # indentazione e spazi diversi
        self.assertEqual(pp.find_function(self.root, "run", code), "pkg/mod.py")
        self.assertEqual(pp.locate_function(self.root, "run", "def run(self, cmd):\n    os.system(cmd)"), ("pkg/mod.py", 5))

    def test_find_function_returns_none_when_body_differs(self):
        self.assertIsNone(pp.find_function(self.root, "run", "def run(self, cmd):\n    os.popen(cmd)"))
        self.assertIsNone(pp.find_function(self.root, "missing", "def missing():\n    pass"))

    def test_decorators_do_not_have_to_match(self):
        (self.root / "pkg" / "views.py").write_text('@app.route("/x")\ndef view(q):\n    return q\n', encoding="utf-8")
        self.assertEqual(pp.find_function(self.root, "view", "def view(q):\n    return q"), "pkg/views.py")

    def test_symlinks_and_the_git_directory_are_never_read(self):
        outside = Path(self._tmp.name) / "outside.py"
        outside.write_text("def secret():\n    return 1\n", encoding="utf-8")
        os.symlink(outside, self.root / "pkg" / "link.py")
        (self.root / ".git").mkdir()
        (self.root / ".git" / "hook.py").write_text("def secret():\n    return 1\n", encoding="utf-8")
        self.assertIsNone(pp.find_function(self.root, "secret", "def secret():\n    return 1"))


class SelectionTests(unittest.TestCase):
    def test_select_repos_prefers_more_rows_and_is_deterministic(self):
        rows = [{"repo_url": "a"}] * 5 + [{"repo_url": "b"}] * 3 + [{"repo_url": "c"}] * 3 + [{"repo_url": "d"}]
        picked = pp.select_repos(rows, 3, seed=1)
        self.assertEqual(picked[0], "a")
        self.assertEqual(set(picked), {"a", "b", "c"})
        self.assertEqual(picked, pp.select_repos(rows, 3, seed=1))
        self.assertEqual(pp.select_repos(rows, 10, seed=1)[-1], "d")

    def test_ties_are_broken_by_a_seeded_hash(self):
        rows = [{"repo_url": f"r{i}"} for i in range(20)]
        orders = {tuple(pp.select_repos(rows, 5, seed=s)) for s in range(4)}
        self.assertGreater(len(orders), 1)

    def test_usable_rows_need_a_catalog_sink_touched_by_the_fix(self):
        good = row("r", "a" * 40)
        no_sink = row("r", "b" * 40, before="def f(x):\n    return x + 1", after="def f(x):\n    return x + 2")
        unchanged = row("r", "c" * 40, after=FUNC_BEFORE)
        no_after = row("r", "d" * 40, after=None)
        not_python = row("r", "e" * 40, programming_language="java")
        self.assertEqual(pp.usable_rows([good, no_sink, unchanged, no_after, not_python], KB), [good])


class FetchTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.first, self.second = make_repo(self.dir / "origin")
        self.url = f"file://{self.dir / 'origin'}"

    def test_fetch_commit_checks_out_the_sha_and_its_parent_can_be_checked_out_too(self):
        work = self.dir / "work"
        self.assertTrue(pp.fetch_commit(self.url, self.second, work, allow_local=True))
        self.assertIn("shlex.quote", (work / "app.py").read_text(encoding="utf-8"))
        self.assertTrue(pp.checkout(work, self.second + "^"))
        self.assertNotIn("shlex", (work / "app.py").read_text(encoding="utf-8"))
        self.assertTrue(pp.checkout(work, self.second))
        self.assertIn("shlex.quote", (work / "app.py").read_text(encoding="utf-8"))

    def test_fetch_commit_returns_false_on_unknown_sha(self):
        self.assertFalse(pp.fetch_commit(self.url, "f" * 40, self.dir / "work", allow_local=True))

    def test_fetch_commit_rejects_anything_that_is_not_a_full_hex_sha(self):
        """Un valore del dataset non deve poter diventare un'opzione di git."""
        for bad in ("--upload-pack=touch /tmp/x", "HEAD", self.second[:10], self.second + "^", ""):
            self.assertFalse(pp.fetch_commit(self.url, bad, self.dir / "w2", allow_local=True), bad)
        self.assertFalse((self.dir / "w2" / ".git").exists() and (self.dir / "w2" / "app.py").exists())

    def test_only_https_github_urls_are_fetched_unless_local_is_allowed(self):
        self.assertFalse(pp.fetch_commit(self.url, self.second, self.dir / "w3"))
        self.assertFalse(pp.fetch_commit("ssh://git@github.com/o/r", self.second, self.dir / "w4"))
        self.assertFalse(pp.fetch_commit("https://example.com/o/r", self.second, self.dir / "w5"))

    def test_a_timeout_is_a_failure_not_an_exception(self):
        with mock.patch.object(pp.subprocess, "run", side_effect=subprocess.TimeoutExpired("git", 1)):
            self.assertFalse(pp.fetch_commit(self.url, self.second, self.dir / "w6", timeout=1, allow_local=True))
            self.assertFalse(pp.checkout(self.dir, self.second))

    def test_fetch_commit_can_be_repeated_on_the_same_directory(self):
        work = self.dir / "work"
        self.assertTrue(pp.fetch_commit(self.url, self.first, work, allow_local=True))
        self.assertTrue(pp.fetch_commit(self.url, self.second, work, allow_local=True))
        self.assertIn("shlex.quote", (work / "app.py").read_text(encoding="utf-8"))


def fake_run_slice(root: Path, targets: list[dict] | None = None, **_kw) -> tuple[list[dict], str]:
    """Slice della cartella `root` com'è adesso: la funzione del sink a profondità 0 e i suoi chiamanti per nome a profondità 1."""
    records = []
    for t in targets or []:
        text = (Path(root) / t["file"]).read_text(encoding="utf-8")
        funcs = localize.split_python_functions(text)
        holder = next(f for f in funcs if f["start_line"] <= t["line"] <= f["start_line"] + f["code"].count("\n"))

        def block(f: dict, depth: int) -> dict:
            return {"file": t["file"], "name": f["name"], "start_line": f["start_line"], "depth": depth, "lang": "python",
                    "end_line": f["start_line"] + f["code"].count("\n"), "code": f["code"]}

        callers = [block(f, 1) for f in funcs if f is not holder and re.search(rf"\b{holder['name']}\(", f["code"])]
        records.append({"sink": {"function": t["callee"], "file": t["file"], "line": t["line"], "cwe_id": t["cwe_id"]},
                        "blocks": [*callers, block(holder, 0)], "truncated": False})
    return records, ""


class PilotRunTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.first, self.second = make_repo(self.dir / "origin")
        self.url = f"file://{self.dir / 'origin'}"
        self.rows = [row(self.url, self.second)]

    def run_pilot(self, rows=None, **kw):
        args = dict(n_repos=1, work_dir=self.dir / "work", seed=42, val_pct=0, test_pct=0, allow_local=True)
        args.update(kw)
        with mock.patch.object(slice_io, "run_slice", side_effect=fake_run_slice), contextlib.redirect_stderr(io.StringIO()):
            return pp.run_pilot(self.rows if rows is None else rows, KB, **args)

    def test_a_pair_is_the_parent_before_the_fix_and_the_fix_commit_after_it(self):
        splits, stats = self.run_pilot()
        examples = splits["train"]
        self.assertEqual([e["meta"]["label"] for e in examples], [1, 0])
        bad, good = examples
        self.assertNotIn("shlex", json.dumps(bad["meta"]["trace"]["blocks"]), "il vulnerabile è il genitore del commit")
        self.assertIn("shlex.quote", json.dumps(good["meta"]["trace"]["blocks"]), "il corretto è il commit della fix")
        for ex in examples:
            meta = ex["meta"]
            self.assertEqual((meta["track"], meta["source"], meta["synthetic"], meta["lang"], meta["cwe"]),
                             (graph_examples.PY_PYVUL_TRACK, pp.SOURCE, False, "python", "CWE-78"))
            self.assertEqual((meta["group"], meta["commit"], meta["repo"]), (self.second, self.second, self.url))
            self.assertEqual(len(meta["trace"]["blocks"]), 2, "run è chiamata da handler")
            self.assertEqual(meta["trace"]["blocks"][-1]["name"], "run")
            self.assertEqual(common.validate_finding(json.loads(ex["messages"][2]["content"])), [])
        self.assertEqual((stats["pairs"], stats["rows_resolved"]), (1, 1))

    def test_prompts_do_not_contain_the_cve_the_description_or_the_commit_message(self):
        splits, _ = self.run_pilot()
        for ex in splits["train"]:
            text = json.dumps(ex["messages"])
            for forbidden in ("CVE-2020-0001", "cmd injection", "second", "first"):
                self.assertNotIn(forbidden, text)

    def test_split_follows_the_commit_with_prepare_data_key_so_test_commits_never_train(self):
        calls = []
        with mock.patch.object(prepare_data, "split_of", side_effect=lambda *a: calls.append(a) or "test"):
            splits, _ = self.run_pilot(val_pct=3, test_pct=11)
        self.assertEqual({k: len(v) for k, v in splits.items()}, {"train": 0, "val": 0, "test": 2})
        self.assertEqual(calls, [(self.second, 42, 3, 11)])

    def test_unknown_commit_and_unlocatable_function_are_counted_not_fatal(self):
        rows = [row(self.url, "a" * 40), row(self.url, self.second, name="missing", before="def missing():\n    os.system(x)",
                                             after="def missing():\n    os.system(shlex.quote(x))")]
        splits, stats = self.run_pilot(rows, n_repos=1)
        self.assertEqual(sum(len(v) for v in splits.values()), 0)
        self.assertEqual(stats["fetch_failed"], 1)
        self.assertEqual(stats["function_not_found"], 1)

    def test_max_rows_per_repo_limits_the_work(self):
        rows = [row(self.url, self.second, name="run"), row(self.url, self.first, name="run")]
        _, stats = self.run_pilot(rows, max_rows_per_repo=1)
        self.assertEqual(stats["rows_considered"], 1)

    def test_a_repo_that_cannot_be_fetched_is_abandoned_after_three_failures(self):
        rows = [row(self.url, f"{i:040x}") for i in range(10)]
        _, stats = self.run_pilot(rows)
        self.assertEqual(stats["fetch_failed"], 3)
        self.assertEqual(stats["repos_abandoned"], 1)

    def test_finalize_mixes_views_on_train_and_val_and_leaves_the_test_set_full(self):
        splits, _ = self.run_pilot()
        final, removed = pp.finalize_splits(splits, KB, seed=42, p_callers=0.0)
        self.assertEqual({e["meta"]["view"] for e in final["train"]}, {"sink_only"}, "la fix cambia il corpo del sink: decidibile")
        self.assertEqual(removed, {"duplicates": 0, "conflicts": 0})
        with mock.patch.object(prepare_data, "split_of", return_value="test"):
            tested, _ = self.run_pilot()
        final, _ = pp.finalize_splits(tested, KB, seed=42, p_callers=0.0)
        self.assertEqual({e["meta"]["view"] for e in final["test"]}, {"callers"})

    def test_finalize_drops_exact_duplicates_and_prompts_with_both_labels(self):
        splits, _ = self.run_pilot()
        doubled = {"train": splits["train"] * 2, "val": [], "test": []}
        final, removed = pp.finalize_splits(doubled, KB, seed=42, p_callers=1.0)
        self.assertEqual((len(final["train"]), removed["duplicates"]), (2, 2))
        clash = json.loads(json.dumps(splits["train"][0]))
        clash["meta"]["label"] = 0
        final, removed = pp.finalize_splits({"train": [splits["train"][0], clash], "val": [], "test": []}, KB, p_callers=1.0)
        self.assertEqual((final["train"], removed["conflicts"]), ([], 2))

    def test_pilot_report_computes_yield_per_repo(self):
        examples = [{"meta": {"repo": "r1"}}, {"meta": {"repo": "r1"}}, {"meta": {"repo": "r2"}}, {"meta": {"repo": "r2"}}]
        report = pp.pilot_report(rows_total=10, rows_resolved=2, examples=examples, seconds=20.0, bytes_on_disk=3 * 1024 * 1024)
        self.assertEqual((report["rows_total"], report["rows_resolved"], report["examples"]), (10, 2, 4))
        self.assertEqual(report["resolve_rate"], 0.2)
        self.assertEqual(report["examples_per_repo"], {"r1": 2, "r2": 2})
        self.assertEqual(report["mean_examples_per_repo"], 2.0)
        self.assertEqual((report["seconds"], report["seconds_per_row"], report["mb_on_disk"]), (20.0, 2.0, 3.0))

    def test_cli_writes_train_val_eval_and_report(self):
        rows_path = self.dir / "rows.jsonl"
        rows_path.write_text("".join(json.dumps(r) + "\n" for r in self.rows), encoding="utf-8")
        out, ev = self.dir / "graph", self.dir / "eval"
        argv = ["--parquet", str(rows_path), "--repos", "1", "--work-dir", str(self.dir / "cli-work"), "--out-dir", str(out),
                "--eval-dir", str(ev), "--val-pct", "0", "--test-pct", "0", "--allow-local-urls"]
        with mock.patch.object(slice_io, "run_slice", side_effect=fake_run_slice), \
                contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pp.main(argv), 0)
        lines = lambda path: [json.loads(l) for l in path.read_text(encoding="utf-8").split("\n") if l.strip()]  # noqa: E731
        self.assertEqual([e["meta"]["label"] for e in lines(out / "py_pyvul_train.jsonl")], [1, 0])
        self.assertEqual(lines(out / "py_pyvul_val.jsonl"), [])
        self.assertEqual(lines(ev / "py_cvefixes.jsonl"), [])
        report = json.loads((out / "py_pyvul_report.json").read_text(encoding="utf-8"))
        self.assertEqual((report["rows_total"], report["rows_resolved"], report["examples"]), (1, 1, 2))
        self.assertGreater(report["bytes_on_disk"], 0)


if __name__ == "__main__":
    unittest.main()
