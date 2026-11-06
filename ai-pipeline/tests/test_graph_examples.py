"""Test di `graph_examples`: esempi con slice, view con e senza chiamanti, conflitti, coppie e split.

I record di slice si costruiscono a mano (già anonimizzati, nomi `fn_<hex>`): niente file Juliet né motore.
"""
import copy
import json
import re
import unittest

import common
from c_anonymize import LEAK_RE
from graph_examples import (
    G2B_FALLBACK,
    G2B_ORIGINS,
    GENERIC,
    NEUTRAL_FALLBACK,
    NEUTRAL_ORIGINS,
    ORIGIN_FALLBACK,
    ORIGINS,
    PHRASES,
    SINK_ONLY_REASONING,
    VARIANT_KINDS,
    build_example,
    cap_balanced,
    choose_views,
    drop_conflicts,
    holdout_variants,
    link_pairs,
    make_candidates,
    messages_from_trace,
    prompt_key,
    sink_only,
    split_of_juliet,
)
from juliet_graph import JULIET_CWES

KB = common.load_kb()
ENTRY = {e["id"]: e for e in KB["entries"]}
HEADER_RE = re.compile(r"^### .*profondità \d+\)$", re.M)
META = {"source": "juliet-graph", "synthetic": False, "track": "J", "group": "CWE78_x_54", "variant": "54"}

# Sink gemello: lo stesso corpo nella catena vulnerabile e in quella sicura; decide solo il chiamante.
SINK_CODE = "void fn_e0e0e0(char * data)\n{\n    if (system(data) != 0)\n    {\n        exit(1);\n    }\n}"
BAD_CALLER = 'void fn_d0d0d0(void)\n{\n    char * data = getenv("ADD");\n    fn_e0e0e0(data);\n}'
GOOD_CALLER = 'void fn_d0d0d0(void)\n{\n    char * data = "ls -l";\n    fn_e0e0e0(data);\n}'
OTHER_SINK = "void fn_e0e0e0(char * data)\n{\n    char * cmd = data;\n    system(cmd);\n}"


def block(name: str, depth: int, code: str, start: int, lang: str = "c", file: str = "src/0a1b2c3d.c") -> dict:
    return {"file": file, "name": name, "start_line": start, "end_line": start + code.count("\n"),
            "depth": depth, "lang": lang, "code": code}


def record(caller: str = BAD_CALLER, sink_code: str = SINK_CODE, *, cwe: str = "CWE-78", callee: str = "system",
           truncated: bool = False) -> dict:
    """Record a 2 blocchi: chiamante (profondità 1) poi sink (profondità 0), come li ordina il motore."""
    blocks = [block("fn_d0d0d0", 1, caller, 10), block("fn_e0e0e0", 0, sink_code, 40)]
    return {"sink": {"function": callee, "file": "src/0a1b2c3d.c", "line": 42, "cwe_id": cwe},
            "blocks": blocks, "truncated": truncated}


def chain(n: int, cwe: str = "CWE-78", callee: str = "system", lang: str = "c") -> dict:
    """Catena di `n` blocchi `fn_a<d>` dalla profondità n-1 alla 0; il blocco a profondità 0 chiama `callee`."""
    blocks = [block(f"fn_a{d}a{d}a{d}", d, f"void fn_a{d}a{d}a{d}(char * data)\n{{\n    use(data);\n}}", 100 * (n - d),
                    lang=lang) for d in range(n - 1, -1, -1)]
    blocks[-1]["code"] = f"void fn_a0a0a0(char * data)\n{{\n    {callee}(data);\n}}"
    sink_line = blocks[-1]["start_line"] + 2
    return {"sink": {"function": callee, "file": "src/0a1b2c3d.c", "line": sink_line, "cwe_id": cwe},
            "blocks": blocks, "truncated": False}


def example(rec: dict, label: int, kind: str | None = None, origin: str | None = "environment", view: str = "callers",
            meta: dict | None = None) -> dict:
    kind = kind or ("bad" if label else "g2b")
    return build_example(rec, KB, label=label, variant_kind=kind, origin=origin, meta=dict(meta or META), view=view)


def user(ex: dict) -> str:
    return next(m["content"] for m in ex["messages"] if m["role"] == "user")


def finding(ex: dict) -> dict:
    return json.loads(ex["messages"][-1]["content"])


def fake(i: int, label: int, cwe: str = "CWE-78", variant: str = "01") -> dict:
    """Esempio minimo per le funzioni che guardano solo il prompt utente e `meta`."""
    return {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": f"prompt {i}"}],
            "meta": {"label": label, "cwe": cwe, "variant": variant, "synthetic": False}}


class BuildExampleTests(unittest.TestCase):
    def test_example_has_multi_block_prompt_and_trace_meta(self):
        rec = record()
        ex = example(rec, 1)
        self.assertEqual([m["role"] for m in ex["messages"]], ["system", "user", "assistant"])
        self.assertEqual(len(HEADER_RE.findall(user(ex))), 2)
        meta = ex["meta"]
        self.assertEqual(len(meta["trace"]["blocks"]), 2)
        self.assertEqual(meta["trace"], {"cwe_id": "CWE-78", "sink": rec["sink"], "blocks": rec["blocks"]})
        self.assertEqual(meta["view"], "callers")
        self.assertEqual(common.validate_finding(finding(ex)), [])
        expected = {"label": 1, "kind": "vulnerable", "cwe": "CWE-78", "lang": "c", "sink_function": "system",
                    "sink_line": 42, "sink_kind": "call", "variant_kind": "bad", "origin": "environment", "pair_id": None,
                    **META}
        self.assertEqual({k: meta[k] for k in expected}, expected)
        self.assertRegex(meta["func_hash"], r"^[0-9a-f]{32}$")
        self.assertEqual(example(rec, 0)["meta"]["kind"], "patched")

    def test_messages_are_the_rendering_of_the_trace(self):
        ex = example(record(), 1)
        self.assertEqual(ex["messages"][:2], messages_from_trace(ex["meta"]["trace"], KB))
        expected = common.build_messages("CWE-78", common.render_card(ENTRY["CWE-78"]), record()["sink"], record()["blocks"])
        self.assertEqual(ex["messages"][:2], expected)

    def test_sink_only_view_renders_one_block_and_keeps_the_full_trace(self):
        ex = example(record(), 1, view="sink_only")
        self.assertEqual(len(HEADER_RE.findall(user(ex))), 1)
        self.assertIn("system(data)", user(ex))
        self.assertNotIn("getenv", user(ex))
        self.assertEqual(len(ex["meta"]["trace"]["blocks"]), 2)
        self.assertEqual(ex["meta"]["view"], "sink_only")
        self.assertIn("`fn_e0e0e0` (profondità 0, unico blocco mostrato)", finding(ex)["reasoning"])
        self.assertNotIn("fn_d0d0d0", finding(ex)["reasoning"])
        with self.assertRaises(ValueError):
            example(record(), 1, view="whatever")

    def test_reasoning_covers_every_cwe_and_kind(self):
        for cwe in JULIET_CWES:
            for kind in ("bad", "g2b", "b2g", "good"):
                phrase = PHRASES[(cwe, kind)]
                self.assertTrue(phrase.strip(), (cwe, kind))
                self.assertIsNone(LEAK_RE.search(phrase), (cwe, kind, phrase))
        for label in (0, 1):
            self.assertTrue(GENERIC[label].strip())
            self.assertIsNone(LEAK_RE.search(GENERIC[label]), GENERIC[label])

    def test_every_reasoning_combination_is_clean(self):
        tokens = ["fscanf", "rand", "connect_socket", "listen_socket", "fgets", "max", "console", "environment", "file",
                  "min", "negative", "large", "fixed"]
        self.assertLessEqual(set(tokens), set(ORIGINS))
        fragments = [*PHRASES.values(), *GENERIC.values(), *ORIGINS.values(), ORIGIN_FALLBACK,
                     *G2B_ORIGINS.values(), G2B_FALLBACK, *NEUTRAL_ORIGINS.values(), NEUTRAL_FALLBACK,
                     SINK_ONLY_REASONING]
        for text in fragments:
            self.assertIsNone(LEAK_RE.search(text), text)
            self.assertNotIn("CVE-", text)
        for cwe in (*JULIET_CWES, "CWE-601"):
            for kind in VARIANT_KINDS:
                for origin in (*ORIGINS, None, "sconosciuto"):
                    ex = example(record(cwe=cwe), int(kind == "bad"), kind, origin)
                    reasoning = finding(ex)["reasoning"]
                    self.assertIsNone(LEAK_RE.search(reasoning), (cwe, kind, origin, reasoning))
                    self.assertNotIn("CVE-", reasoning)
                    self.assertIn("`system`", reasoning)
                    self.assertIn("`fn_d0d0d0` (profondità 1)", reasoning)

    def test_origin_is_ignored_for_g2b_and_neutral_when_missing(self):
        g2b = finding(example(record(), 0, "g2b", "environment"))["reasoning"]
        self.assertNotIn(ORIGINS["environment"], g2b)
        self.assertIn(G2B_ORIGINS["CWE-78"], g2b)
        bad = finding(example(record(), 1, "bad", "environment"))["reasoning"]
        self.assertIn(ORIGINS["environment"], bad)
        b2g = finding(example(record(), 0, "b2g", "environment"))["reasoning"]
        self.assertIn(ORIGINS["environment"], b2g)
        for cwe in JULIET_CWES:
            missing = finding(example(record(cwe=cwe), 1, "bad", None))["reasoning"]
            self.assertIn(NEUTRAL_ORIGINS[cwe], missing)
            self.assertNotIn(ORIGIN_FALLBACK, missing)
        unknown = finding(example(record(), 1, "bad", "sconosciuto"))["reasoning"]
        self.assertIn(ORIGIN_FALLBACK, unknown)

    ORIGIN_TEXTS = (*ORIGINS.values(), ORIGIN_FALLBACK, *G2B_ORIGINS.values(), G2B_FALLBACK, *NEUTRAL_ORIGINS.values(),
                    NEUTRAL_FALLBACK)

    def assert_sink_only_claims_nothing_unseen(self, rec: dict, cwe: str, callee: str):
        for kind in VARIANT_KINDS:
            for origin in (*ORIGINS, None, "sconosciuto"):
                label = int(kind == "bad")
                alone = example(rec, label, kind, origin, view="sink_only")
                reasoning = finding(alone)["reasoning"]
                for text in self.ORIGIN_TEXTS:
                    self.assertNotIn(text, reasoning, (cwe, kind, origin))
                self.assertNotIn("fn_a1a1a1", reasoning)
                self.assertNotIn("fn_a2a2a2", reasoning)
                self.assertNotIn("risale a", reasoning)
                self.assertIn("profondità 0", reasoning)
                self.assertIn(f"`{callee}` in `fn_a0a0a0`", reasoning)
                self.assertIn(PHRASES.get((cwe, kind), GENERIC[label]), reasoning)
                self.assertIsNone(LEAK_RE.search(reasoning))
                self.assertNotIn("CVE-", reasoning)
                self.assertEqual(common.validate_finding(finding(alone)), [])
                # Stesso prompt, stessa etichetta: la slice senza chiamanti dà lo stesso reasoning del solo sink.
                single = example(sink_only(rec), label, kind, origin)
                self.assertEqual(finding(single)["reasoning"], reasoning)
                full = finding(example(rec, label, kind, origin))["reasoning"]
                self.assertIn("risale a `fn_a2a2a2` (profondità 2)", full)

    def test_sink_only_reasoning_makes_no_claim_about_unseen_blocks(self):
        self.assert_sink_only_claims_nothing_unseen(chain(3), "CWE-78", "system")

    def test_generic_sink_only_reasoning_makes_no_claim_about_unseen_blocks(self):
        self.assert_sink_only_claims_nothing_unseen(chain(3, "CWE-601", "redirect", "python"), "CWE-601", "redirect")

    def test_cwe_outside_the_table_uses_the_generic_phrase(self):
        code = "def fn_c1c1c1(url):\n    return redirect(url)"
        rec = {"sink": {"function": "redirect", "file": "src/9f8e7d6c.py", "line": 6, "cwe_id": "CWE-601"},
               "blocks": [block("fn_c1c1c1", 0, code, 5, lang="py", file="src/9f8e7d6c.py")], "truncated": False}
        ex = example(rec, 1, "bad", None, meta={**META, "track": "P1"})
        reasoning = finding(ex)["reasoning"]
        self.assertTrue(reasoning.strip())
        self.assertIn("`redirect`", reasoning)
        self.assertIn(GENERIC[1], reasoning)
        self.assertEqual(common.validate_finding(finding(ex)), [])

    def test_reasoning_names_root_function_and_depth_and_has_no_cve(self):
        ex = example(chain(5), 1)
        reasoning = finding(ex)["reasoning"]
        self.assertIn("fn_a4a4a4", reasoning)
        self.assertIn("profondità 4", reasoning)
        self.assertIn("`system`", reasoning)
        self.assertNotIn("CVE-", reasoning)
        self.assertEqual(len(HEADER_RE.findall(user(ex))), 5)

    def test_vulnerable_fix_is_catalog_hint_and_safe_fix_is_none_needed(self):
        bad = finding(example(record(), 1))
        self.assertEqual(bad["suggested_fix"], ENTRY["CWE-78"]["fix_hint"])
        self.assertEqual((bad["vulnerable"], bad["vex_status"], bad["severity"]),
                         (True, "affected", ENTRY["CWE-78"]["default_severity"]))
        safe = finding(example(record(GOOD_CALLER), 0))
        self.assertEqual(safe["suggested_fix"], "Nessuna modifica necessaria.")
        self.assertEqual((safe["vulnerable"], safe["vex_status"], safe["severity"]), (False, "not_affected", "NONE"))
        self.assertEqual(common.validate_finding(safe), [])

    def test_label_and_variant_kind_must_agree(self):
        with self.assertRaises(ValueError):
            example(record(), 1, "g2b")
        with self.assertRaises(ValueError):
            example(record(), 0, "bad")
        with self.assertRaises(ValueError):
            example(record(), 1, "evil")
        with self.assertRaises(ValueError):
            example(record(), 1, meta={**META, "label": 0})

    def test_sink_only_keeps_depth_zero_block(self):
        rec = record()
        before = copy.deepcopy(rec)
        out = sink_only(rec)
        self.assertEqual(out["blocks"], [rec["blocks"][1]])
        self.assertEqual(out["sink"], rec["sink"])
        self.assertEqual(rec, before)
        no_sink = record()
        no_sink["blocks"] = no_sink["blocks"][:1]
        with self.assertRaises(ValueError):
            sink_only(no_sink)
        with self.assertRaises(ValueError):
            example(no_sink, 1)

    def test_truncated_slice_is_rejected(self):
        with self.assertRaises(ValueError):
            example(record(truncated=True), 1)


class ConflictAndViewTests(unittest.TestCase):
    def test_drop_conflicts_removes_prompts_seen_with_both_labels(self):
        bad = example(record(), 1)
        same_prompt_safe = example(record(), 0)
        other = example(record(GOOD_CALLER), 0)
        self.assertEqual(prompt_key(bad), prompt_key(same_prompt_safe))
        self.assertNotEqual(prompt_key(bad), prompt_key(other))
        remaining, dropped = drop_conflicts([bad, other, same_prompt_safe])
        self.assertEqual((remaining, dropped), ([other], 2))

    def test_prompt_key_is_a_sha256_of_the_user_message(self):
        ex = example(record(), 1)
        self.assertRegex(prompt_key(ex), r"^[0-9a-f]{64}$")
        changed = copy.deepcopy(ex)
        changed["messages"][0]["content"] = "altro system"
        self.assertEqual(prompt_key(changed), prompt_key(ex))

    def test_choose_views_mixes_seventy_thirty_and_respects_conflicts(self):
        candidates = [{"full": fake(i, i % 2), "sink_only": fake(10_000 + i, i % 2)} for i in range(1000)]
        chosen = choose_views(candidates, seed=42)
        self.assertEqual(len(chosen), 1000)
        share = sum(c is cand["sink_only"] for c, cand in zip(chosen, candidates)) / 1000
        self.assertGreaterEqual(share, 0.25)
        self.assertLessEqual(share, 0.35)
        self.assertEqual(chosen, choose_views(candidates, seed=42))
        blocked = [{"full": cand["full"], "sink_only": None} for cand in candidates]
        self.assertTrue(all(c is cand["full"] for c, cand in zip(choose_views(blocked, seed=42), blocked)))

    def test_make_candidates_drops_sink_only_view_of_twins(self):
        bad = example(record(BAD_CALLER), 1)
        good = example(record(GOOD_CALLER), 0)
        lone = example(record(BAD_CALLER, OTHER_SINK), 1)
        cands = make_candidates([bad, good, lone], KB)
        self.assertEqual([c["full"] for c in cands], [bad, good, lone])
        self.assertIsNone(cands[0]["sink_only"])
        self.assertIsNone(cands[1]["sink_only"])
        alone = cands[2]["sink_only"]
        self.assertIsNotNone(alone)
        self.assertEqual(alone["meta"]["view"], "sink_only")
        self.assertEqual(alone["meta"]["label"], 1)
        self.assertEqual(alone["meta"]["trace"], lone["meta"]["trace"])
        self.assertEqual(alone["messages"][:2], messages_from_trace(lone["meta"]["trace"], KB, sink_only=True))
        self.assertEqual({k: alone["meta"][k] for k in META}, META)
        # Una sola view possibile: senza chiamanti la slice completa È già il solo sink.
        single = example(sink_only(record()), 1)
        self.assertIsNone(make_candidates([single], KB)[0]["sink_only"])


class PairTests(unittest.TestCase):
    def test_link_pairs_pairs_equal_sink_only_prompt_with_opposite_labels(self):
        bad = example(record(BAD_CALLER), 1)
        good = example(record(GOOD_CALLER), 0)
        lone = example(record(BAD_CALLER, OTHER_SINK), 1)
        self.assertNotEqual(prompt_key(bad), prompt_key(good))
        pairs = link_pairs([bad, lone, good], KB)
        self.assertEqual(len(pairs), 2)
        self.assertEqual({p["meta"]["label"] for p in pairs}, {0, 1})
        self.assertEqual(pairs[0]["meta"]["pair_id"], pairs[1]["meta"]["pair_id"])
        self.assertIsNotNone(pairs[0]["meta"]["pair_id"])
        self.assertIsNone(bad["meta"]["pair_id"])  # gli esempi in ingresso non vengono modificati
        # La view dell'esempio non conta: la coppia si riconosce dal solo sink ricostruito da meta.trace.
        good_sink_only = example(record(GOOD_CALLER), 0, view="sink_only")
        again = link_pairs([bad, good_sink_only], KB)
        self.assertEqual(len(again), 2)
        self.assertEqual(again[0]["meta"]["pair_id"], pairs[0]["meta"]["pair_id"])

    def test_link_pairs_leaves_extra_and_traceless_examples_unpaired(self):
        bad1 = example(record(BAD_CALLER), 1)
        bad2 = example(record(BAD_CALLER.replace("ADD", "CMD")), 1)
        good = example(record(GOOD_CALLER), 0)
        traceless = copy.deepcopy(good)
        del traceless["meta"]["trace"]
        broken = copy.deepcopy(good)
        broken["meta"]["trace"]["blocks"] = broken["meta"]["trace"]["blocks"][:1]  # niente profondità 0
        pairs = link_pairs([bad1, bad2, good, traceless, broken, {"messages": []}], KB)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(sorted(p["meta"]["label"] for p in pairs), [0, 1])
        ids = {p["meta"]["pair_id"] for p in link_pairs([bad2, good, bad1], KB)}
        self.assertEqual(ids, {pairs[0]["meta"]["pair_id"]})
        self.assertEqual(link_pairs([bad1, bad2], KB), [])
        # Slice completa identica con etichette opposte: nemmeno i chiamanti decidono, quindi non è una coppia.
        self.assertEqual(link_pairs([bad1, example(record(BAD_CALLER), 0)], KB), [])
        self.assertEqual(len(link_pairs([bad1, example(record(BAD_CALLER), 0), good], KB)), 0)


class SplitAndCapTests(unittest.TestCase):
    VARIANTS = [f"{n:02d}" for n in [*range(1, 19), 21, 22, *range(31, 35), *range(41, 46), *range(51, 55),
                                     *range(61, 69), *range(72, 75), *range(81, 85)]]

    def test_split_is_stable_per_group_and_holds_out_variants(self):
        held = {"54"}
        splits = {split_of_juliet("CWE78_x_01", "01", 42, 10, 10, held) for _ in range(5)}
        self.assertEqual(len(splits), 1)
        self.assertIn(splits.pop(), {"train", "val", "test"})
        for group in ("CWE78_a_54", "CWE120_b_54", "CWE416_c_54"):
            self.assertEqual(split_of_juliet(group, "54", 42, 10, 10, held), "unseen_variants")
        seen = {split_of_juliet(f"CWE78_g{i}_01", "01", 42, 10, 10, held) for i in range(300)}
        self.assertEqual(seen, {"train", "val", "test"})

    def test_holdout_variants_picks_six_of_forty_eight(self):
        self.assertEqual(len(self.VARIANTS), 48)
        held = holdout_variants(self.VARIANTS, seed=42)
        self.assertEqual(len(held), 6)
        self.assertLessEqual(held, set(self.VARIANTS))
        self.assertEqual(held, holdout_variants(list(reversed(self.VARIANTS)) + self.VARIANTS[:3], seed=42))
        self.assertNotEqual(held, holdout_variants(self.VARIANTS, seed=7))

    def test_cap_balanced_limits_total_and_keeps_label_balance(self):
        examples = []
        for cwe, variant, n1, n0 in (("CWE-78", "01", 200, 40), ("CWE-78", "54", 30, 30), ("CWE-120", "01", 90, 10),
                                     ("CWE-416", "41", 5, 60)):
            examples += [fake(len(examples) + i, 1, cwe, variant) for i in range(n1)]
            examples += [fake(len(examples) + i, 0, cwe, variant) for i in range(n0)]
        capped = cap_balanced(examples, max_total=101, seed=42)
        labels = [e["meta"]["label"] for e in capped]
        self.assertLessEqual(len(capped), 101)
        self.assertGreaterEqual(len(capped), 100)
        self.assertLessEqual(abs(labels.count(1) - labels.count(0)), 1)
        self.assertEqual(capped, cap_balanced(examples, max_total=101, seed=42))
        order = [examples.index(e) for e in capped]
        self.assertEqual(order, sorted(order))
        # Uniforme per (cwe, variant): fra i vulnerabili nessun gruppo sotto quota mentre un altro ne ha di più.
        vuln = [(e["meta"]["cwe"], e["meta"]["variant"]) for e in capped if e["meta"]["label"] == 1]
        counts = {k: vuln.count(k) for k in {("CWE-78", "01"), ("CWE-78", "54"), ("CWE-120", "01"), ("CWE-416", "41")}}
        self.assertEqual(counts[("CWE-416", "41")], 5)
        big = [counts[k] for k in counts if k != ("CWE-416", "41")]
        self.assertLessEqual(max(big) - min(big), 1)
        # Con pochi esempi di un'etichetta il totale scende per restare bilanciato.
        few = [fake(i, 1) for i in range(50)] + [fake(100 + i, 0) for i in range(3)]
        out = cap_balanced(few, max_total=40, seed=1)
        self.assertEqual(sorted(e["meta"]["label"] for e in out), [0, 0, 0, 1, 1, 1, 1])


if __name__ == "__main__":
    unittest.main()
