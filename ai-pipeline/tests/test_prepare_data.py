"""Test di prepare_data.py con record PrimeVul sintetici (nessun download)."""
import json
import random
import tempfile
import unittest
from pathlib import Path

import common
import prepare_data as pd

KB = common.load_kb()

VULN_FUNC = """static int copy_name(char *dst, const char *src)
{
    strcpy(dst, src);
    return helper(dst);
}"""

FIXED_FUNC = """static int copy_name(char *dst, const char *src)
{
    snprintf(dst, 16, "%s", src);
    return helper(dst);
}"""


def record(**kw):
    base = {
        "idx": 1,
        "project": "proj",
        "commit_id": "c1",
        "target": 1,
        "func": VULN_FUNC,
        "file_name": "a.c",
        "cwe": ["CWE-787"],
        "cve": "CVE-2020-1",
        "cve_desc": "Buffer overflow in copy_name",
    }
    base.update(kw)
    return base


class ParseTests(unittest.TestCase):
    def test_parse_cwes_accepts_lists_and_string_reprs(self):
        self.assertEqual(pd.parse_cwes(["CWE-787", "CWE-20"]), ["CWE-787", "CWE-20"])
        self.assertEqual(pd.parse_cwes("['CWE-704']"), ["CWE-704"])
        self.assertEqual(pd.parse_cwes("None"), [])
        self.assertEqual(pd.parse_cwes(None), [])
        self.assertEqual(pd.parse_cwes([]), [])

    def test_function_name_extraction(self):
        cases = {
            VULN_FUNC: "copy_name",
            "void Foo::bar(int n)\n{\n}": "bar",
            "char *dup_str(const char *s)\n{\n return 0;\n}": "dup_str",
            "MODULE_EXPORT int\nmulti_line(int a,\n           int b)\n{\n}": "multi_line",
            "garbage without parens": "unknown",
        }
        for code, expected in cases.items():
            with self.subTest(expected):
                self.assertEqual(pd.extract_function_name(code), expected)

    def test_language_guess(self):
        self.assertEqual(pd.guess_lang("a.c", "int f(){}"), "c")
        self.assertEqual(pd.guess_lang("a.cc", "int f(){}"), "cpp")
        self.assertEqual(pd.guess_lang("x.hpp", "int f(){}"), "cpp")
        self.assertEqual(pd.guess_lang("None", "void A::f() {}"), "cpp")
        self.assertEqual(pd.guess_lang("None", "int f() {}"), "c")
        self.assertEqual(pd.guess_lang("app/db.py", "def f(): pass"), "python")
        self.assertEqual(pd.guess_lang("None", "def f(a):\n    return a"), "python")


class CallDetectionTests(unittest.TestCase):
    def test_comments_strings_and_keywords_are_ignored(self):
        code = (
            "int f(char *s)\n"
            "{\n"
            "    /* strcpy(a, b); */\n"
            '    puts("memcpy(x)");\n'
            "    // system(cmd);\n"
            "    if (s) {\n"
            "        strcpy(buf, s);\n"
            "    }\n"
            "    return sizeof(buf);\n"
            "}\n"
        )
        calls = pd.find_calls(code)
        self.assertEqual([c for c, _ in calls], ["puts", "strcpy"])
        self.assertEqual(dict(calls)["strcpy"], 7)

    def test_member_and_namespace_callees_keep_their_path(self):
        calls = pd.find_calls("void f() {\n  ctx->run(1);\n  std::memcpy(a, b, n);\n  obj.close();\n}")
        self.assertEqual([c for c, _ in calls], ["ctx->run", "std::memcpy", "obj.close"])

    def test_the_function_own_header_is_not_a_call(self):
        self.assertEqual(pd.find_calls("int helper(char *p) { return p[0]; }"), [])

    def test_detect_sink_honours_the_preferred_cwe(self):
        code = "void f() {\n  char *p = malloc(n);\n  strcpy(p, s);\n}"
        first = pd.detect_sink(KB, code, "c")
        self.assertEqual((first["entry"]["id"], first["callee"], first["line"]), ("CWE-190", "malloc", 2))
        wanted = pd.detect_sink(KB, code, "c", preferred_cwe="CWE-120")
        self.assertEqual((wanted["entry"]["id"], wanted["callee"], wanted["line"]), ("CWE-120", "strcpy", 3))
        self.assertIsNone(pd.detect_sink(KB, code, "c", preferred_cwe="CWE-78"))


class BuildExampleTests(unittest.TestCase):
    def build(self, rec, **kw):
        return pd.build_example(rec, KB, patched={}, max_chars=12000, **kw)

    def test_vulnerable_record_becomes_a_validated_training_example(self):
        ex, reason = self.build(record())
        self.assertIsNone(reason)
        roles = [m["role"] for m in ex["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant"])
        self.assertEqual(ex["messages"][0]["content"], common.system_prompt())

        finding = json.loads(ex["messages"][2]["content"])
        self.assertEqual(common.validate_finding(finding), [])
        self.assertEqual(
            (finding["vulnerable"], finding["cwe_id"], finding["severity"], finding["sink_function"], finding["sink_file"], finding["sink_line"], finding["vex_status"]),
            (True, "CWE-120", "HIGH", "strcpy", "a.c", 3, "affected"),
        )
        self.assertIn("CVE-2020-1", finding["reasoning"])
        self.assertIn("Buffer overflow in copy_name", finding["reasoning"])
        self.assertEqual(finding["suggested_fix"], next(e for e in KB["entries"] if e["id"] == "CWE-120")["fix_hint"])

        user = ex["messages"][1]["content"]
        self.assertIn("CWE-120 — Buffer Copy without Checking Size of Input", user)
        self.assertIn("- Function: strcpy", user)
        self.assertIn("- Line: 3", user)
        self.assertIn("### a.c · copy_name (lines 1-5, depth 0)", user)
        self.assertIn("    3 |     strcpy(dst, src);", user)
        self.assertEqual(ex["meta"]["label"], 1)
        self.assertEqual(ex["meta"]["kind"], "vulnerable")

    def test_assistant_json_keeps_the_schema_field_order(self):
        ex, _ = self.build(record())
        self.assertEqual(list(json.loads(ex["messages"][2]["content"]).keys()), list(common.load_schema()["properties"].keys()))

    def test_patched_function_is_a_hard_negative(self):
        rec = record(target=0, func=FIXED_FUNC.replace("snprintf(dst, 16, \"%s\", src)", "strncpy(dst, src, 15)"))
        ex, reason = self.build(rec)
        self.assertIsNone(reason)
        finding = json.loads(ex["messages"][2]["content"])
        self.assertEqual((finding["vulnerable"], finding["severity"], finding["vex_status"], finding["cwe_id"]), (False, "NONE", "not_affected", "CWE-120"))
        self.assertEqual(finding["suggested_fix"], "No change needed.")
        self.assertEqual(ex["meta"]["kind"], "patched")
        self.assertEqual(common.validate_finding(finding), [])

    def test_plain_benign_function_uses_the_first_matching_sink(self):
        rec = record(target=0, cwe=[], cve="None", cve_desc="None", commit_id="None", func="void g(char *p) {\n  free(p);\n}")
        ex, reason = self.build(rec)
        self.assertIsNone(reason)
        finding = json.loads(ex["messages"][2]["content"])
        self.assertEqual((finding["cwe_id"], finding["sink_function"], finding["sink_line"]), ("CWE-416", "free", 2))
        self.assertEqual(ex["meta"]["kind"], "benign")

    def test_upstream_fix_diff_is_used_when_the_patched_pair_is_known(self):
        patched = {("c1", "copy_name"): FIXED_FUNC}
        ex, _ = pd.build_example(record(), KB, patched=patched, max_chars=12000)
        fix = json.loads(ex["messages"][2]["content"])["suggested_fix"]
        self.assertTrue(fix.startswith("Apply the upstream fix:"), fix)
        self.assertIn("-    strcpy(dst, src);", fix)
        self.assertIn('+    snprintf(dst, 16, "%s", src);', fix)

    def test_skip_reasons(self):
        cases = [
            (record(cwe=["CWE-703"]), "cwe_not_in_kb"),
            (record(func="void f() {\n  helper();\n}"), "no_sink"),
            (record(func="void f() {\n  strcpy(a, b);\n" + "  x = 1;\n" * 5000 + "}"), "too_long"),
            (record(func=""), "empty_function"),
            (record(target=0, cwe=[], cve="None", func="void f() { helper(); }"), "no_sink"),
        ]
        for rec, expected in cases:
            with self.subTest(expected):
                ex, reason = self.build(rec)
                self.assertIsNone(ex)
                self.assertEqual(reason, expected)

    def test_sinkless_function_mode_uses_the_function_itself_as_sink(self):
        rec = record(func="void f(struct a *p) {\n  p->x = 1;\n}", cwe=["CWE-476"])
        ex, reason = pd.build_example(rec, KB, patched={}, max_chars=12000, sinkless="function")
        self.assertIsNone(reason)
        finding = json.loads(ex["messages"][2]["content"])
        self.assertEqual((finding["cwe_id"], finding["sink_function"], finding["sink_line"], finding["vulnerable"]), ("CWE-476", "f", 1, True))
        self.assertEqual(ex["meta"]["sink_kind"], "function")
        self.assertIn("- Function: f", ex["messages"][1]["content"])
        self.assertEqual(common.validate_finding(finding), [])

    def test_call_sinks_are_tagged_and_sinkless_mode_does_not_change_them(self):
        ex, _ = pd.build_example(record(), KB, patched={}, max_chars=12000, sinkless="function")
        self.assertEqual((ex["meta"]["sink_kind"], ex["meta"]["sink_function"]), ("call", "strcpy"))

    def test_sinkless_mode_still_needs_a_known_cwe_and_never_applies_to_unlabeled_code(self):
        unknown = record(func="void f() {\n  helper();\n}", cwe=["CWE-9999"])
        self.assertEqual(pd.build_example(unknown, KB, patched={}, max_chars=12000, sinkless="function")[1], "cwe_not_in_kb")
        unlabeled = record(target=0, cwe=[], cve="None", commit_id="None", func="void f() { helper(); }")
        self.assertEqual(pd.build_example(unlabeled, KB, patched={}, max_chars=12000, sinkless="function")[1], "no_sink")

    # ---- sink localizzati --------------------------------------------------------------------------------------
    def test_diff_localized_sink_beats_an_earlier_call_on_an_unchanged_line(self):
        vuln = "static int f(char *d, char *s)\n{\n    memcpy(tmp, s, 4);\n    strcpy(d, s);\n    return 0;\n}"
        fixed = vuln.replace("strcpy(d, s)", "strncpy(d, s, 15)")
        ex, reason = self.build(record(func=vuln, counterpart=fixed), sinkless="function")
        self.assertIsNone(reason)
        finding = json.loads(ex["messages"][2]["content"])
        self.assertEqual((finding["sink_function"], finding["sink_line"]), ("strcpy", 4))
        self.assertEqual((ex["meta"]["sink_kind"], ex["meta"]["localized"]), ("call", "diff"))

    def test_function_level_sink_points_at_the_first_changed_line(self):
        vuln = "int f(struct a *p)\n{\n    int x = p->v;\n    use(p->w);\n    return x;\n}"
        fixed = vuln.replace("    use(p->w);", "    if (!p)\n        return -1;\n    use(p->w);")
        ex, reason = self.build(record(func=vuln, counterpart=fixed, cwe=["CWE-476"]), sinkless="function")
        self.assertIsNone(reason)
        finding = json.loads(ex["messages"][2]["content"])
        self.assertEqual((finding["sink_function"], finding["sink_line"]), ("f", 4))
        self.assertEqual((ex["meta"]["sink_kind"], ex["meta"]["localized"]), ("function", "diff"))

    def test_with_a_counterpart_unrelated_call_sinks_are_never_promoted_to_the_sink(self):
        vuln = "int f(char *d, char *s)\n{\n    memcpy(d, s, 4);\n    int y = 1;\n    return y;\n}"
        fixed = vuln.replace("int y = 1;", "int y = 2;")
        self.assertEqual(self.build(record(func=vuln, counterpart=fixed), sinkless="skip")[1], "no_sink")
        ex, _ = self.build(record(func=vuln, counterpart=fixed), sinkless="function")
        self.assertEqual((ex["meta"]["sink_kind"], ex["meta"]["sink_line"]), ("function", 4))

    def test_negative_twin_is_localized_on_its_own_changed_line(self):
        fixed = FIXED_FUNC.replace('snprintf(dst, 16, "%s", src)', "strncpy(dst, src, 15)")
        ex, reason = self.build(record(target=0, func=fixed, counterpart=VULN_FUNC), sinkless="function")
        self.assertIsNone(reason)
        finding = json.loads(ex["messages"][2]["content"])
        self.assertEqual((finding["vulnerable"], finding["sink_function"], finding["sink_line"]), (False, "strncpy", 3))
        self.assertEqual(ex["meta"]["kind"], "patched")

    def test_the_vulnerable_lookup_provides_the_counterpart_of_a_patched_function(self):
        fixed = FIXED_FUNC.replace('snprintf(dst, 16, "%s", src)', "strncpy(dst, src, 15)")
        ex, _ = pd.build_example(record(target=0, func=fixed), KB, patched={}, max_chars=12000, sinkless="function",
                                 vulnerable_map={("c1", "copy_name"): VULN_FUNC})
        self.assertEqual((ex["meta"]["sink_function"], ex["meta"]["localized"]), ("strncpy", "diff"))

    JULIET_BAD = ("void CWE690_x_bad()\n{\n    char * data;\n    /* FLAW: no check */\n    strcpy(data, \"x\");\n    free(data);\n}")

    def test_juliet_flaw_hint_is_the_sink_and_the_example_is_marked_synthetic(self):
        rec = {"idx": "j1", "project": "juliet", "commit_id": "None", "target": 1, "func": self.JULIET_BAD, "file_name": "CWE690_x_01.c",
               "cwe": ["CWE-690"], "cve": "None", "sink_hint": {"callee": "strcpy", "line": 5}, "source": "juliet", "synthetic": True, "kind": "vulnerable"}
        ex, reason = self.build(rec)
        self.assertIsNone(reason)
        finding = json.loads(ex["messages"][2]["content"])
        self.assertEqual((finding["cwe_id"], finding["sink_function"], finding["sink_line"]), ("CWE-476", "strcpy", 5))
        self.assertEqual((ex["meta"]["localized"], ex["meta"]["source"], ex["meta"]["synthetic"]), ("comment", "juliet", True))

    def test_explicit_kind_marks_juliet_good_functions_as_hard_negatives(self):
        good = "static void goodG2B()\n{\n    char *p = malloc(10);\n    if (p) free(p);\n}"
        rec = {"idx": "j2", "project": "juliet", "commit_id": "None", "target": 0, "func": good, "cwe": ["CWE-416"], "cve": "None",
               "kind": "patched", "sink_hint": {"callee": "free", "line": 4}, "source": "juliet", "synthetic": True}
        ex, _ = self.build(rec)
        self.assertEqual(ex["meta"]["kind"], "patched")

    def test_python_records_use_python_call_detection(self):
        vuln = "def q(cur, uid):\n    cur.execute(f\"SELECT * FROM t WHERE id={uid}\")\n    return cur.fetchall()"
        fixed = vuln.replace('f"SELECT * FROM t WHERE id={uid}"', '"SELECT * FROM t WHERE id=%s", (uid,)')
        ex, reason = self.build(record(func=vuln, counterpart=fixed, cwe=["CWE-89"], file_name="db.py"), sinkless="function")
        self.assertIsNone(reason)
        finding = json.loads(ex["messages"][2]["content"])
        self.assertEqual((finding["cwe_id"], finding["sink_function"], finding["sink_line"]), ("CWE-89", "cur.execute", 2))
        self.assertEqual(ex["meta"]["lang"], "python")
        self.assertIn("```python", ex["messages"][1]["content"])

    def test_record_source_defaults_to_primevul_and_is_not_synthetic(self):
        ex, _ = self.build(record())
        self.assertEqual((ex["meta"]["source"], ex["meta"]["synthetic"]), ("primevul", False))

    def test_commit_message_note_is_added_to_the_reasoning_when_there_is_no_cve_text(self):
        ex, _ = self.build(record(cve_desc="None", note="[media] avoid use-after-free"))
        self.assertIn("avoid use-after-free", json.loads(ex["messages"][2]["content"])["reasoning"])

    def test_missing_file_name_gets_a_stable_label(self):
        ex, _ = self.build(record(file_name="None"))
        self.assertEqual(json.loads(ex["messages"][2]["content"])["sink_file"], "proj/copy_name.c")


class SplitAndBalanceTests(unittest.TestCase):
    def test_hash_split_is_deterministic_and_groups_by_commit(self):
        self.assertEqual(pd.is_validation("c1", 10, seed=1), pd.is_validation("c1", 10, seed=1))
        share = sum(pd.is_validation(f"commit{i}", 10, seed=1) for i in range(2000)) / 2000
        self.assertTrue(0.06 < share < 0.14, share)

    def test_negative_plan_prefers_hard_negatives_but_respects_the_ratio(self):
        # (vulnerabili, patch, benigni disponibili, rapporto) -> (patch tenute, benigni tenuti)
        self.assertEqual(pd.negative_plan(10, 4, 100, 1.0), (4, 6))
        self.assertEqual(pd.negative_plan(10, 14, 100, 1.0), (10, 0))
        self.assertEqual(pd.negative_plan(10, 0, 100, 2.5), (0, 25))
        self.assertEqual(pd.negative_plan(10, 4, 3, 1.0), (4, 3), "non si inventano esempi che non ci sono")
        self.assertEqual(pd.negative_plan(0, 5, 5, 1.0), (0, 0))

    def test_reservoir_selection_is_seeded_and_exact(self):
        rng = random.Random(7)
        picked = pd.choose_indices(1000, 50, seed=7)
        self.assertEqual(len(picked), 50)
        self.assertEqual(picked, pd.choose_indices(1000, 50, seed=7))
        self.assertNotEqual(picked, pd.choose_indices(1000, 50, seed=8))
        self.assertEqual(len(pd.choose_indices(10, 50, seed=1)), 10)
        del rng


def write_jsonl(path: Path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


class EndToEndTests(unittest.TestCase):
    def make_inputs(self, root: Path):
        def rows(prefix, n_vuln, n_benign):
            out = []
            for i in range(n_vuln):
                name = f"copy_{prefix}{i}"
                out.append(record(idx=f"{prefix}v{i}", commit_id=f"{prefix}c{i}", func=VULN_FUNC.replace("copy_name", name)))
                out.append(record(idx=f"{prefix}p{i}", commit_id=f"{prefix}c{i}", target=0,
                                  func=FIXED_FUNC.replace("copy_name", name).replace("snprintf(dst, 16, \"%s\", src)", "strncpy(dst, src, 15)")))
            for i in range(n_benign):
                out.append(record(idx=f"{prefix}b{i}", commit_id="None", target=0, cwe=[], cve="None", cve_desc="None", func=f"void g{i}(char *p) {{\n  free(p);\n}}"))
            out.append(record(idx=f"{prefix}x", target=0, cwe=[], cve="None", commit_id="None", func="void nosink() { helper(); }"))
            return out

        write_jsonl(root / "train.jsonl", rows("t", 6, 30))
        write_jsonl(root / "valid.jsonl", rows("v", 2, 10))
        write_jsonl(root / "test.jsonl", rows("e", 3, 20))

    def run_main(self, raw: Path, out: Path):
        return pd.main([
            "--train", str(raw / "train.jsonl"),
            "--valid", str(raw / "valid.jsonl"),
            "--test", str(raw / "test.jsonl"),
            "--out-dir", str(out),
            "--seed", "3",
        ])

    def test_pipeline_writes_balanced_splits_and_stats(self):
        with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as out_dir:
            raw, out = Path(raw_dir), Path(out_dir)
            self.make_inputs(raw)
            self.assertEqual(self.run_main(raw, out), 0)

            for name, n_vuln in (("train", 6), ("val", 2), ("test", 3)):
                lines = (out / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
                examples = [json.loads(l) for l in lines]
                labels = [e["meta"]["label"] for e in examples]
                self.assertEqual(sum(labels), n_vuln, name)
                self.assertEqual(len(labels) - sum(labels), n_vuln, f"{name}: classi bilanciate (ratio 1.0)")
                kinds = {e["meta"]["kind"] for e in examples}
                self.assertIn("patched", kinds)
                for e in examples:
                    self.assertEqual(common.validate_finding(json.loads(e["messages"][2]["content"])), [])
            stats = json.loads((out / "stats.json").read_text(encoding="utf-8"))
            self.assertEqual(stats["train"]["kept"]["vulnerable"], 6)
            self.assertEqual(stats["train"]["skipped"]["no_sink"], 1)
            self.assertFalse((out / ".benign_tmp.jsonl").exists())

    def test_benign_ratio_adds_plain_negatives_on_top_of_patched_ones(self):
        with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as out_dir:
            raw, out = Path(raw_dir), Path(out_dir)
            self.make_inputs(raw)
            pd.main(["--train", str(raw / "train.jsonl"), "--valid", str(raw / "valid.jsonl"), "--out-dir", str(out), "--seed", "3", "--benign-ratio", "2", "--label-ratio", "0"])
            examples = [json.loads(l) for l in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()]
            kinds = [e["meta"]["kind"] for e in examples]
            self.assertEqual((kinds.count("vulnerable"), kinds.count("patched"), kinds.count("benign")), (6, 6, 6))

    def test_patched_negatives_are_downsampled_to_the_requested_ratio(self):
        with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as out_dir:
            raw, out = Path(raw_dir), Path(out_dir)
            fixed = FIXED_FUNC.replace("snprintf(dst, 16, \"%s\", src)", "strncpy(dst, src, 15)")
            rows = [record(idx="v0", commit_id="c0", func=VULN_FUNC), record(idx="v1", commit_id="c1", func=VULN_FUNC.replace("copy_name", "copy_two"))]
            rows += [record(idx=f"p{i}", commit_id="c0", target=0, func=fixed.replace("copy_name", f"other_{i}")) for i in range(6)]
            write_jsonl(raw / "train.jsonl", rows)
            pd.main(["--train", str(raw / "train.jsonl"), "--valid", str(raw / "train.jsonl"), "--out-dir", str(out), "--benign-ratio", "1"])
            kinds = [json.loads(l)["meta"]["kind"] for l in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual((kinds.count("vulnerable"), kinds.count("patched")), (2, 2))

    def test_output_is_deterministic(self):
        with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as o1, tempfile.TemporaryDirectory() as o2:
            raw = Path(raw_dir)
            self.make_inputs(raw)
            self.run_main(raw, Path(o1))
            self.run_main(raw, Path(o2))
            for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
                self.assertEqual((Path(o1) / name).read_bytes(), (Path(o2) / name).read_bytes(), name)

    def test_without_a_valid_file_the_validation_set_is_carved_from_train_by_commit(self):
        with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as out_dir:
            raw, out = Path(raw_dir), Path(out_dir)
            rows = []
            for i in range(40):  # abbastanza commit perché entrambi gli split risultino popolati
                rows.append(record(idx=f"v{i}", commit_id=f"c{i}", func=VULN_FUNC.replace("copy_name", f"copy_{i}")))
                rows.append(record(idx=f"p{i}", commit_id=f"c{i}", target=0,
                                   func=FIXED_FUNC.replace("copy_name", f"copy_{i}").replace("snprintf(dst, 16, \"%s\", src)", "strncpy(dst, src, 15)")))
            write_jsonl(raw / "train.jsonl", rows)
            code = pd.main(["--train", str(raw / "train.jsonl"), "--out-dir", str(out), "--seed", "3", "--val-pct", "50"])
            self.assertEqual(code, 0)
            train_commits = {json.loads(l)["meta"]["commit_id"] for l in (out / "train.jsonl").read_text().splitlines()}
            val_commits = {json.loads(l)["meta"]["commit_id"] for l in (out / "val.jsonl").read_text().splitlines()}
            self.assertTrue(val_commits)
            self.assertFalse((train_commits - {"None"}) & (val_commits - {"None"}), "una commit non può stare in entrambi gli split")
            self.assertFalse((out / "test.jsonl").exists())


if __name__ == "__main__":
    unittest.main()


# ======================================================================================================== multi-sorgente
import collections  # noqa: E402

import sources  # noqa: E402


class SplitAndDedupTests(unittest.TestCase):
    def test_split_of_is_deterministic_and_respects_the_percentages(self):
        counts = collections.Counter(pd.split_of(f"g{i}", 1, 10, 20) for i in range(4000))
        self.assertTrue(0.16 < counts["test"] / 4000 < 0.24, counts)
        self.assertTrue(0.07 < counts["val"] / 4000 < 0.13, counts)
        self.assertEqual(pd.split_of("g1", 1, 10, 20), pd.split_of("g1", 1, 10, 20))

    def test_without_a_test_share_nothing_goes_to_test(self):
        self.assertNotIn("test", {pd.split_of(f"g{i}", 1, 10, 0) for i in range(2000)})

    def test_func_hash_ignores_whitespace(self):
        self.assertEqual(pd.func_hash("a  b\n  c"), pd.func_hash("a b c"))
        self.assertNotEqual(pd.func_hash("a b"), pd.func_hash("a c"))

    def test_examples_carry_the_hashes_of_function_and_counterpart(self):
        ex, _ = pd.build_example(record(counterpart=FIXED_FUNC), KB, patched={}, max_chars=12000, sinkless="function")
        self.assertEqual(ex["meta"]["func_hash"], pd.func_hash(VULN_FUNC))
        self.assertEqual(ex["meta"]["counterpart_hash"], pd.func_hash(FIXED_FUNC))

    def test_dedupe_drops_what_the_test_set_already_contains_including_the_other_twin(self):
        def ex(code, other=None):
            return {"meta": {"func_hash": pd.func_hash(code), "counterpart_hash": pd.func_hash(other) if other else None, "label": 1}}

        test = [ex("T1", "T1fixed")]
        train = [ex("T1"), ex("T1fixed", "T1"), ex("other"), ex("other")]
        kept, removed = pd.dedupe(train, pd.forbidden_hashes(test))
        self.assertEqual(len(kept), 1)
        self.assertEqual(removed, {"in_test": 2, "duplicate": 1})


class FunctionShareTests(unittest.TestCase):
    @staticmethod
    def make(n_call, n_function):
        def ex(kind, label, i):
            return {"meta": {"sink_kind": kind, "label": label, "func_hash": f"{kind}{label}{i}"}}

        return ([ex("call", i % 2, i) for i in range(n_call)] + [ex("function", i % 2, i) for i in range(n_function)])

    def test_function_level_examples_are_capped_to_the_requested_share_keeping_label_balance(self):
        kept, dropped = pd.limit_function_level(self.make(10, 100), 0.5, seed=1)
        fn = [e for e in kept if e["meta"]["sink_kind"] == "function"]
        self.assertEqual((len(kept) - len(fn), len(fn), dropped), (10, 10, 90))
        self.assertEqual(sorted(collections.Counter(e["meta"]["label"] for e in fn).values()), [5, 5])

    def test_share_one_keeps_everything_and_zero_keeps_only_call_sinks(self):
        data = self.make(10, 100)
        self.assertEqual(pd.limit_function_level(data, 1.0, seed=1), (data, 0))
        kept, dropped = pd.limit_function_level(data, 0.0, seed=1)
        self.assertEqual((len(kept), dropped), (10, 100))

    def test_selection_is_deterministic_and_never_adds_examples(self):
        data = self.make(4, 6)
        a, _ = pd.limit_function_level(data, 0.6, seed=3)
        self.assertEqual(a, pd.limit_function_level(data, 0.6, seed=3)[0])
        self.assertLessEqual(len(a), len(data))

    def test_without_call_sinks_nothing_is_dropped_by_the_cap(self):
        data = self.make(0, 20)
        self.assertEqual(pd.limit_function_level(data, 0.5, seed=1), (data, 0))


class LabelSkewTests(unittest.TestCase):
    @staticmethod
    def ex(label, synthetic, i):
        return {"meta": {"label": label, "synthetic": synthetic, "func_hash": f"{label}{synthetic}{i}"}}

    def test_the_majority_label_is_trimmed_dropping_synthetic_examples_first(self):
        data = ([self.ex(1, True, i) for i in range(10)] + [self.ex(1, False, i) for i in range(4)] + [self.ex(0, False, i) for i in range(8)])
        kept, dropped = pd.limit_label_skew(data, 1.25, seed=1)
        self.assertEqual(dropped, 4)
        self.assertEqual(sum(1 for e in kept if e["meta"]["label"] == 1), 10)
        self.assertEqual(sum(1 for e in kept if e["meta"]["label"] == 1 and not e["meta"]["synthetic"]), 4, "gli esempi reali restano")
        self.assertEqual(sum(1 for e in kept if e["meta"]["label"] == 0), 8)

    def test_real_examples_are_trimmed_only_when_synthetic_ones_are_not_enough(self):
        data = [self.ex(1, False, i) for i in range(20)] + [self.ex(0, False, i) for i in range(8)]
        kept, dropped = pd.limit_label_skew(data, 1.0, seed=1)
        self.assertEqual((dropped, len(kept)), (12, 16))

    def test_balanced_or_disabled_means_untouched_and_order_is_preserved(self):
        data = [self.ex(i % 2, False, i) for i in range(10)]
        self.assertEqual(pd.limit_label_skew(data, 1.25, seed=1), (data, 0))
        skewed = [self.ex(1, True, i) for i in range(10)] + [self.ex(0, False, 0)]
        self.assertEqual(pd.limit_label_skew(skewed, 0, seed=1), (skewed, 0))
        kept, _ = pd.limit_label_skew(skewed, 1.25, seed=1)
        self.assertEqual([e["meta"]["func_hash"] for e in kept], sorted((e["meta"]["func_hash"] for e in kept), key=lambda h: [x["meta"]["func_hash"] for x in skewed].index(h)))


class MultiSourceEndToEndTests(unittest.TestCase):
    PY_BEFORE = 'def q(cur, uid):\n    cur.execute(f"SELECT * FROM t WHERE id={uid}")\n'
    PY_AFTER = 'def q(cur, uid):\n    cur.execute("SELECT * FROM t WHERE id=%s", (uid,))\n'
    JULIET_BAD = ("void CWE690_x_%02d_bad()\n{\n    char * data;\n    /* FLAW: no check */\n    strcpy(data, \"x\");\n}\n")
    JULIET_GOOD = ("static void goodG2B()\n{\n    char * data = malloc(10);\n    /* FIX: check first */\n    if (data) { free(data); }\n}\n")

    def write_inputs(self, root: Path):
        sv = []
        for i in range(60):
            vuln = f"int f{i}(char *d, char *s)\n{{\n    strcpy(d, s);\n    return {i};\n}}"
            fixed = vuln.replace("strcpy(d, s)", "strncpy(d, s, 15)")
            base = {"project": "linux", "project_url": "u", "filepath": "d/x.c", "commit_id": f"sv{i}", "commit_message": "fix", "func_name": f"f{i}",
                    "cve_list": ["CVE-1"], "cwe_list": ["CWE-120"]}
            sv.append({**base, "idx": 2 * i, "is_vulnerable": True, "func_body": vuln, "fixed_func_idx": 2 * i + 1.0})
            sv.append({**base, "idx": 2 * i + 1, "is_vulnerable": False, "func_body": fixed, "fixed_func_idx": None})
        write_jsonl(root / "sv.jsonl", sv)
        py = [{"source_dataset": "secureIT-project/CVEfixes", "cve_id": f"CVE-{i}", "cve_description": "sqli", "cwe_id": "CWE-89", "function_name": "q",
               "code_before": self.PY_BEFORE.replace("uid", f"uid{i}"), "code_after": self.PY_AFTER.replace("uid", f"uid{i}"), "commit_hash": f"py{i}",
               "repo_url": "https://github.com/o/r", "programming_language": "python"} for i in range(60)]
        write_jsonl(root / "py.jsonl", py)
        import csv

        with (root / "juliet.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["filename", "bad", "good"])
            w.writeheader()
            for i in range(30):
                w.writerow({"filename": f"testcases/CWE690_x/s01/CWE690_x_{i:02d}.c", "bad": self.JULIET_BAD % i, "good": self.JULIET_GOOD})

    def run_main(self, raw: Path, out: Path, *extra):
        return pd.main(["--train", str(raw / "train.jsonl"), "--valid", str(raw / "valid.jsonl"), "--test", str(raw / "test.jsonl"),
                        "--secvuleval", str(raw / "sv.jsonl"), "--python-pairs", str(raw / "py.jsonl"), "--juliet", str(raw / "juliet.csv"),
                        "--out-dir", str(out), "--seed", "3", "--holdout-pct", "25", "--val-pct", "10", *extra])

    def setUp(self):
        self._raw, self._out = tempfile.TemporaryDirectory(), tempfile.TemporaryDirectory()
        self.raw, self.out = Path(self._raw.name), Path(self._out.name)
        EndToEndTests().make_inputs(self.raw)
        self.write_inputs(self.raw)

    def tearDown(self):
        self._raw.cleanup()
        self._out.cleanup()

    def load(self, name):
        return [json.loads(l) for l in (self.out / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_every_source_contributes_and_juliet_never_reaches_the_test_set(self):
        self.assertEqual(self.run_main(self.raw, self.out), 0)
        sources_by_split = {name: collections.Counter(e["meta"]["source"] for e in self.load(name)) for name in ("train", "val", "test")}
        self.assertEqual(set(sources_by_split["train"]), {"primevul", "secvuleval", "cvefixes-python", "juliet"})
        self.assertNotIn("juliet", sources_by_split["test"])
        self.assertTrue({"primevul", "secvuleval", "cvefixes-python"} <= set(sources_by_split["test"]), sources_by_split["test"])

    def test_no_function_is_shared_between_training_and_test(self):
        self.run_main(self.raw, self.out)
        test_hashes = {e["meta"]["func_hash"] for e in self.load("test")}
        for name in ("train", "val"):
            self.assertFalse(test_hashes & {e["meta"]["func_hash"] for e in self.load(name)}, name)

    def test_stats_break_down_by_source(self):
        self.run_main(self.raw, self.out)
        stats = json.loads((self.out / "stats.json").read_text(encoding="utf-8"))
        self.assertIn("secvuleval", stats["train"]["by_source"])
        self.assertGreater(stats["train"]["by_source"]["secvuleval"]["kept"]["vulnerable"], 0)
        self.assertIn("duplicates_removed", stats["train"])

    def test_localized_call_sinks_dominate_the_synthetic_and_diff_based_sources(self):
        self.run_main(self.raw, self.out)
        sv = [e for e in self.load("train") if e["meta"]["source"] == "secvuleval" and e["meta"]["kind"] == "vulnerable"]
        self.assertTrue(sv)
        self.assertTrue(all(e["meta"]["sink_function"] == "strcpy" and e["meta"]["localized"] == "diff" for e in sv))
        for e in self.load("train"):
            self.assertEqual(common.validate_finding(json.loads(e["messages"][2]["content"])), [])

    def test_output_is_deterministic_with_all_sources(self):
        self.run_main(self.raw, self.out)
        with tempfile.TemporaryDirectory() as other:
            self.run_main(self.raw, Path(other))
            for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
                self.assertEqual((self.out / name).read_bytes(), (Path(other) / name).read_bytes(), name)

    def write_codeql(self):
        base = self.raw / "codeql" / "python" / "ql" / "test" / "query-tests" / "Security" / "CWE-089-SqlInjection"
        base.mkdir(parents=True)
        for i in range(40):
            (base / f"case{i}.py").write_text(
                f"def view{i}(cursor, name):\n"
                f"    # GOOD\n"
                f"    cursor.execute('SELECT {i} WHERE n = %s', name)\n"
                f"    cursor.execute('SELECT {i} WHERE n = %s' % name)  # $ Alert\n",
                encoding="utf-8",
            )
        return self.raw / "codeql"

    def test_codeql_tests_are_a_source_with_per_sink_verdicts_and_no_leaked_labels(self):
        root = self.write_codeql()
        self.run_main(self.raw, self.out, "--codeql", str(root))
        found = [e for name in ("train", "val", "test") for e in self.load(name) if e["meta"]["source"] == "codeql-tests"]
        self.assertTrue(found)
        self.assertEqual({e["meta"]["localized"] for e in found}, {"comment"})
        self.assertEqual({e["meta"]["label"] for e in found}, {0, 1})
        for e in found:
            user = e["messages"][1]["content"]
            self.assertNotIn("$ Alert", user)
            self.assertNotIn("# GOOD", user)
        test_sources = {e["meta"]["source"] for e in self.load("test")}
        self.assertIn("codeql-tests", test_sources)

    def test_function_share_only_trims_training_and_validation(self):
        def function_level(prefix, n):
            return [record(idx=f"{prefix}{i}", commit_id=f"{prefix}fc{i}", func=f"void {prefix}fn{i}(struct a *p)\n{{\n  p->x = {i};\n}}", cwe=["CWE-476"])
                    for i in range(n)]

        for name, prefix in (("train", "t"), ("test", "e")):
            path = self.raw / f"{name}.jsonl"
            path.write_text(path.read_text(encoding="utf-8") + "\n".join(json.dumps(r) for r in function_level(prefix + "x", 6)) + "\n", encoding="utf-8")
        self.run_main(self.raw, self.out, "--function-share", "0")
        self.assertEqual({e["meta"]["sink_kind"] for e in self.load("train")}, {"call"})
        self.assertIn("function", {e["meta"]["sink_kind"] for e in self.load("test")}, "il test non viene ridotto")
        stats = json.loads((self.out / "stats.json").read_text(encoding="utf-8"))
        self.assertGreater(stats["train"]["function_level_dropped"], 0)
        self.assertEqual(stats["test"]["function_level_dropped"], 0)

    def test_juliet_per_cwe_cap(self):
        self.run_main(self.raw, self.out, "--juliet-per-cwe", "3")
        groups = {e["meta"]["commit_id"] for name in ("train", "val") for e in self.load(name) if e["meta"]["source"] == "juliet"}
        self.assertLessEqual(len(groups), 3)
