"""Test delle parti pure di benchmark.py + client HTTP verso un finto server vLLM (stdlib)."""
import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import benchmark
import common


class MetricsTests(unittest.TestCase):
    def test_precision_recall_f1_fpr(self):
        m = benchmark.metrics_from_counts(tp=8, fp=2, tn=6, fn=4)
        self.assertAlmostEqual(m["precision"], 0.8)
        self.assertAlmostEqual(m["recall"], 8 / 12)
        self.assertAlmostEqual(m["f1"], 2 * 0.8 * (8 / 12) / (0.8 + 8 / 12))
        self.assertAlmostEqual(m["fpr"], 2 / 8)
        self.assertAlmostEqual(m["accuracy"], 14 / 20)

    def test_degenerate_cases_do_not_divide_by_zero(self):
        m = benchmark.metrics_from_counts(tp=0, fp=0, tn=0, fn=0)
        self.assertEqual((m["precision"], m["recall"], m["f1"], m["fpr"]), (0.0, 0.0, 0.0, 0.0))

    def test_confusion_counts(self):
        self.assertEqual(benchmark.confusion([1, 1, 0, 0, 1], [True, False, True, False, True]), (2, 1, 1, 1))

    def test_evaluate_reports_the_call_sink_subset_and_invalid_outputs(self):
        examples = [
            {"meta": {"label": 1, "sink_kind": "call"}},
            {"meta": {"label": 0, "sink_kind": "call"}},
            {"meta": {"label": 1, "sink_kind": "function"}},
            {"meta": {"label": 0, "sink_kind": "function"}},
        ]
        report = benchmark.evaluate(examples, [True, False, None, True])
        self.assertEqual(report["n"], 4)
        self.assertEqual(report["invalid"], 1)
        self.assertEqual(report["all"]["counts"], {"tp": 1, "fp": 1, "tn": 1, "fn": 1}, "un output non valido conta come 'non vulnerabile'")
        self.assertEqual(report["call_sinks"]["counts"], {"tp": 1, "fp": 0, "tn": 1, "fn": 0})


class BySourceTests(unittest.TestCase):
    EXAMPLES = [
        {"meta": {"label": 1, "sink_kind": "call", "source": "codeql-tests"}},
        {"meta": {"label": 0, "sink_kind": "call", "source": "codeql-tests"}},
        {"meta": {"label": 1, "sink_kind": "function", "source": "primevul"}},
        {"meta": {"label": 0, "sink_kind": "function", "source": "primevul"}},
    ]

    def test_metrics_are_reported_per_source(self):
        report = benchmark.evaluate(self.EXAMPLES, [True, False, False, False])
        self.assertEqual(set(report["by_source"]), {"codeql-tests", "primevul"})
        self.assertEqual(report["by_source"]["codeql-tests"]["counts"], {"tp": 1, "fp": 0, "tn": 1, "fn": 0})
        self.assertEqual(report["by_source"]["primevul"]["counts"], {"tp": 0, "fp": 0, "tn": 1, "fn": 1})

    def test_the_accuracy_table_has_one_row_per_source(self):
        report = benchmark.evaluate(self.EXAMPLES, [True, False, False, False])
        table = benchmark.accuracy_table({"llm-slice": report})
        self.assertIn("| llm-slice | fonte: codeql-tests |", table)
        self.assertIn("| llm-slice | fonte: primevul |", table)
        self.assertIn("| llm-slice | tutti |", table)


class CodeExtractionTests(unittest.TestCase):
    def test_extract_code_inverts_the_numbered_rendering(self):
        code = "int f(int a)\n{\n\n\tif (a)\n\t\treturn 1;\n    return 0;\n}"
        block = {"file": "a.c", "name": "f", "start_line": 1, "end_line": 7, "depth": 0, "lang": "c", "code": code}
        user = common.render_user_prompt("CWE-1", "CARD", {"function": "f", "file": "a.c", "line": 1}, [block])
        self.assertEqual(benchmark.extract_code(user), code)

    def test_noslice_prompt_has_no_rule_and_no_sink(self):
        block = {"file": "a.c", "name": "f", "start_line": 1, "end_line": 2, "depth": 0, "lang": "c", "code": "int f()\n{}"}
        user = common.render_user_prompt("CWE-120", "CARD CWE-120", {"function": "strcpy", "file": "a.c", "line": 2}, [block])
        ex = {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": user}], "meta": {"lang": "c"}}
        msgs = benchmark.noslice_messages(ex)
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])
        text = msgs[0]["content"] + msgs[1]["content"]
        self.assertNotIn("CWE-120", text)
        self.assertNotIn("strcpy", text)
        self.assertNotIn("Regola CWE", text)
        self.assertIn("int f()\n{}", msgs[1]["content"])
        self.assertIn("```c", msgs[1]["content"])


class VerdictTests(unittest.TestCase):
    def test_verdict_from_content(self):
        self.assertIs(benchmark.verdict_from_content('{"vulnerable": true, "x": 1}'), True)
        self.assertIs(benchmark.verdict_from_content('{"vulnerable": false}'), False)
        self.assertIsNone(benchmark.verdict_from_content("not json"))
        self.assertIsNone(benchmark.verdict_from_content('{"other": 1}'))
        self.assertIsNone(benchmark.verdict_from_content('{"vulnerable": "yes"}'))
        self.assertIsNone(benchmark.verdict_from_content(None))


class SemgrepTests(unittest.TestCase):
    def examples(self):
        return [
            {"messages": [{"role": "user", "content": common.render_user_prompt("C", "K", {"function": "f", "file": "a.c", "line": 1},
                          [{"file": "a.c", "name": "f", "start_line": 1, "end_line": 1, "depth": 0, "lang": lang, "code": f"int f{i}() {{}}"}])}],
             "meta": {"lang": lang}}
            for i, lang in enumerate(["c", "cpp", "c"])
        ]

    def test_cases_are_written_with_language_extensions(self):
        with tempfile.TemporaryDirectory() as d:
            index = benchmark.write_semgrep_cases(self.examples(), Path(d))
            names = sorted(Path(p).name for p in index)
            self.assertEqual(names, ["case_000000.c", "case_000001.cpp", "case_000002.c"])
            self.assertEqual(sorted(index.values()), [0, 1, 2])
            first = next(p for p, i in index.items() if i == 0)
            self.assertEqual(Path(first).read_text(encoding="utf-8"), "int f0() {}\n")

    def test_report_parsing_maps_findings_back_to_examples(self):
        with tempfile.TemporaryDirectory() as d:
            index = benchmark.write_semgrep_cases(self.examples(), Path(d))
            files = {i: p for p, i in index.items()}
            report = {"results": [{"path": files[1]}, {"path": files[1]}, {"path": "/elsewhere/other.c"}], "errors": []}
            self.assertEqual(benchmark.parse_semgrep(report, index), {1})


class GpuMemoryTests(unittest.TestCase):
    def test_parse_nvidia_smi_output(self):
        self.assertEqual(benchmark.parse_nvidia_smi_memory("12345\n"), 12345)
        self.assertEqual(benchmark.parse_nvidia_smi_memory("520 MiB\n99 MiB\n"), 520, "si legge la prima GPU")
        with self.assertRaises(ValueError):
            benchmark.parse_nvidia_smi_memory("")

    def test_peak_sampler_reports_the_maximum_over_the_baseline(self):
        values = iter([1000, 3000, 9000, 4000])
        last = [4000]

        def reader():
            try:
                last[0] = next(values)
            except StopIteration:
                pass
            return last[0]

        sampler = benchmark.PeakSampler(reader, interval=0.005)
        with sampler:
            time.sleep(0.15)
        self.assertEqual(sampler.baseline_mib, 1000)
        self.assertEqual(sampler.peak_mib, 9000)
        self.assertEqual(sampler.delta_mib, 8000)


class VllmLogTests(unittest.TestCase):
    def test_weights_and_kv_cache_are_read_from_the_startup_log(self):
        log = (
            "INFO 11-04 19:30:01 [gpu_model_runner.py:1] Model loading took 7.5159 GiB and 5.12 seconds\n"
            "INFO 11-04 19:30:20 [kv_cache_utils.py:1] GPU KV cache size: 1,048,576 tokens\n"
            "INFO 11-04 19:30:20 [kv_cache_utils.py:2] Maximum concurrency for 8,192 tokens per request: 128.00x\n"
        )
        self.assertEqual(benchmark.parse_vllm_log(log), {"weights_gib": 7.5159, "kv_tokens": 1048576})

    def test_missing_lines_give_none(self):
        self.assertEqual(benchmark.parse_vllm_log("nothing useful"), {"weights_gib": None, "kv_tokens": None})


class TableTests(unittest.TestCase):
    def test_markdown_table_with_f1_delta_against_the_first_row(self):
        rows = [
            {"format": "bf16", "weights_gb": 7.6, "kv_tokens": 90000, "vram_gb": 11.2, "disk_gb": 7.9, "tok_s": 40.0, "latency_s": 2.5, "scan_s": 31.0, "f1": 0.80},
            {"format": "nvfp4", "weights_gb": None, "kv_tokens": 210000, "vram_gb": 11.3, "disk_gb": 2.8, "tok_s": 90.5, "latency_s": 1.1, "scan_s": None, "f1": 0.77},
        ]
        table = benchmark.markdown_table(rows)
        lines = table.splitlines()
        self.assertTrue(lines[0].startswith("| Formato"))
        self.assertIn("Pesi in VRAM (GB)", lines[0])
        self.assertIn("| bf16 | 7.6 | 90000 | 11.2 | 7.9 | 40.0 | 2.50 | 31.0 | 0.800 | +0.000 |", lines[2])
        self.assertIn("| nvfp4 | n/d | 210000 | 11.3 | 2.8 | 90.5 | 1.10 | n/d | 0.770 | -0.030 |", lines[3])


class PercentileTests(unittest.TestCase):
    def test_nearest_rank(self):
        values = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]  # non ordinati di proposito
        self.assertEqual(benchmark.percentile(values, 50), 5)
        self.assertEqual(benchmark.percentile(values, 95), 10)
        self.assertEqual(benchmark.percentile([7], 95), 7)
        self.assertEqual(benchmark.percentile([], 50), 0.0)


class VllmVersionTests(unittest.TestCase):
    def test_version_is_read_from_the_api_server_line_or_the_engine_line(self):
        self.assertEqual(benchmark.parse_vllm_version("INFO 11-04 [api_server.py:1] vLLM API server version 0.11.0"), "0.11.0")
        self.assertEqual(benchmark.parse_vllm_version("Initializing a V1 LLM engine (v0.11.2) with config"), "0.11.2")
        self.assertIsNone(benchmark.parse_vllm_version("nothing"))


class HardwareRowTests(unittest.TestCase):
    def test_row_keeps_the_old_fields_and_adds_the_new_ones(self):
        examples = [{"meta": {"label": label, "sink_kind": "call"}} for label in (1, 1, 0, 0)]
        raw = [
            {"content": '{"vulnerable": true}', "completion_tokens": 100, "latency": 1.0},
            {"content": '{"vulnerable": false}', "completion_tokens": 100, "latency": 2.0},
            {"content": '{"vulnerable": false}', "completion_tokens": 100, "latency": 3.0},
            {"content": None, "completion_tokens": 0, "latency": 0.5, "error": "timeout"},
        ]
        row = benchmark.hardware_row("bf16", examples, raw, wall=2.0, log_info={"weights_gib": 7.5, "kv_tokens": 1000}, vllm_version="0.11.0",
                                     vram_delta_mib=2048, disk_gb=7.9, scan_s=None, concurrency=8)
        self.assertEqual(row["format"], "bf16")
        self.assertEqual(row["tok_s"], 150.0)  # 300 token in 2 s
        self.assertEqual(row["latency_s"], 2.0)
        self.assertEqual(row["latency_p50_s"], 2.0)
        self.assertEqual(row["latency_p95_s"], 3.0)
        self.assertEqual(row["vram_gb"], 2.0)
        self.assertEqual(row["weights_gb"], 7.5)
        self.assertEqual(row["kv_tokens"], 1000)
        self.assertEqual(row["disk_gb"], 7.9)
        self.assertEqual((row["n"], row["concurrency"], row["vllm_version"]), (4, 8, "0.11.0"))
        self.assertEqual(row["report"]["n"], 4)
        self.assertEqual(row["f1"], row["report"]["all"]["f1"])
        self.assertEqual(row["invalid"], 1)
        self.assertIn("| bf16 |", benchmark.markdown_table([row]))


class MockVllm:
    """Server HTTP minimo compatibile con /v1/chat/completions, per testare il client senza GPU."""

    def __init__(self, responder, delay=0.0):
        self.requests, self.in_flight, self.max_in_flight = [], 0, 0
        self._lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with outer._lock:
                    outer.requests.append(body)
                    outer.in_flight += 1
                    outer.max_in_flight = max(outer.max_in_flight, outer.in_flight)
                    index = len(outer.requests) - 1
                time.sleep(delay)
                content = responder(body, index)
                payload = json.dumps({"choices": [{"message": {"content": content}}], "usage": {"completion_tokens": 7}}).encode()
                with outer._lock:
                    outer.in_flight -= 1
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


class ChatClientTests(unittest.TestCase):
    def test_request_is_deterministic_guided_and_without_thinking(self):
        with MockVllm(lambda b, i: '{"vulnerable": true}') as m:
            out = benchmark.chat_completion(m.url, "sast-nemotron", [{"role": "user", "content": "x"}])
        body = m.requests[0]
        self.assertEqual(body["model"], "sast-nemotron")
        self.assertEqual(body["temperature"], 0.0)
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(body["response_format"]["type"], "json_schema")
        self.assertEqual(body["response_format"]["json_schema"]["schema"], common.load_schema())
        self.assertEqual(out["content"], '{"vulnerable": true}')
        self.assertEqual(out["completion_tokens"], 7)
        self.assertGreaterEqual(out["latency"], 0)

    def test_run_llm_keeps_order_and_respects_concurrency(self):
        with MockVllm(lambda b, i: json.dumps({"vulnerable": b["messages"][0]["content"] == "yes"}), delay=0.05) as m:
            sets = [[{"role": "user", "content": "yes" if i % 2 == 0 else "no"}] for i in range(8)]
            results = benchmark.run_llm(m.url, "m", sets, concurrency=3)
        self.assertEqual([benchmark.verdict_from_content(r["content"]) for r in results], [i % 2 == 0 for i in range(8)])
        self.assertEqual(m.max_in_flight, 3)

    def test_unreachable_server_yields_error_results_not_exceptions(self):
        results = benchmark.run_llm("http://127.0.0.1:1", "m", [[{"role": "user", "content": "x"}]], concurrency=1)
        self.assertIsNone(results[0]["content"])
        self.assertIn("error", results[0])


class VllmCommandTests(unittest.TestCase):
    def test_command_uses_the_shared_flags(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "config.json").write_text("{}", encoding="utf-8")
            cmd = benchmark.vllm_command("vllm", Path(d), port=8123, gpu_util=0.7)
        self.assertEqual(cmd[:2], ["vllm", "serve"])
        self.assertIn("8123", cmd)
        self.assertIn("0.70", cmd)


class VllmEnvTests(unittest.TestCase):
    """vLLM parte dal suo eseguibile senza attivare il venv: i tool dello stesso venv (ninja per FlashInfer) devono stare nel PATH."""

    def test_venv_bin_dir_goes_first_and_the_rest_is_kept(self):
        base = {"PATH": "/usr/bin:/bin", "HOME": "/root"}
        env = benchmark.vllm_env("/content/venv-vllm/bin/vllm", base)
        self.assertEqual(env["PATH"], "/content/venv-vllm/bin:/usr/bin:/bin")
        self.assertEqual(env["HOME"], "/root")
        self.assertEqual(base["PATH"], "/usr/bin:/bin")  # l'ambiente di partenza non si modifica

    def test_relative_path_becomes_absolute(self):
        # FlashInfer lancia ninja con un'altra cartella di lavoro: un PATH relativo non troverebbe nulla
        env = benchmark.vllm_env(".venv-vllm/bin/vllm", {"PATH": "/usr/bin"})
        first = env["PATH"].split(":")[0]
        self.assertTrue(Path(first).is_absolute())
        self.assertEqual(Path(first), Path(".venv-vllm/bin").absolute())

    def test_bare_name_is_resolved_by_the_path_already(self):
        env = benchmark.vllm_env("vllm", {"PATH": "/usr/bin"})
        self.assertEqual(env["PATH"], "/usr/bin")

    def test_no_empty_path_entry_when_path_is_missing(self):
        # una voce vuota nel PATH significa "cartella corrente"
        env = benchmark.vllm_env("/opt/v/bin/vllm", {})
        self.assertEqual(env["PATH"], "/opt/v/bin")

    def test_defaults_to_the_process_environment(self):
        self.assertTrue(benchmark.vllm_env("/opt/v/bin/vllm")["PATH"].startswith("/opt/v/bin:"))

    def test_wait_ready_error_carries_the_log_tail(self):
        with tempfile.TemporaryDirectory() as d:
            log_path = Path(d) / "vllm.log"
            log_path.write_text("".join(f"riga {i}\n" for i in range(100)) + "FileNotFoundError: 'ninja'\n", encoding="utf-8")
            proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(3)"])
            proc.wait()
            with self.assertRaises(RuntimeError) as ctx:
                benchmark.wait_ready("http://127.0.0.1:9", proc, 5, log_path=log_path)
        message = str(ctx.exception)
        self.assertIn("codice 3", message)
        self.assertIn("FileNotFoundError: 'ninja'", message)
        self.assertIn("riga 99", message)
        self.assertNotIn("riga 10\n", message)  # solo la coda, non tutto il log
        self.assertIn(str(log_path), message)

    def test_wait_ready_without_a_log_keeps_working(self):
        proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(1)"])
        proc.wait()
        with self.assertRaises(RuntimeError) as ctx:
            benchmark.wait_ready("http://127.0.0.1:9", proc, 5)
        self.assertIn("codice 1", str(ctx.exception))

    def test_started_process_finds_the_tools_next_to_vllm(self):
        with tempfile.TemporaryDirectory() as d:
            bin_dir = Path(d) / "venv" / "bin"
            bin_dir.mkdir(parents=True)
            fake_vllm = bin_dir / "vllm"
            fake_vllm.write_text("#!/bin/sh\ncommand -v ninja\n", encoding="utf-8")
            fake_vllm.chmod(0o755)
            fake_ninja = bin_dir / "ninja"
            fake_ninja.write_text("#!/bin/sh\n", encoding="utf-8")
            fake_ninja.chmod(0o755)
            model = Path(d) / "model"
            model.mkdir()
            log_path = Path(d) / "vllm.log"
            with log_path.open("w") as log:
                proc = benchmark.start_vllm(str(fake_vllm), model, 8000, 0.8, log)
                proc.wait(timeout=30)
            self.assertEqual(log_path.read_text().strip(), str(fake_ninja))


if __name__ == "__main__":
    unittest.main()
