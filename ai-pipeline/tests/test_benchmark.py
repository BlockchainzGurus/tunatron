"""Test delle parti pure di benchmark.py + client HTTP verso un finto server vLLM (stdlib)."""
import contextlib
import io
import json
import re
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import benchmark
import common
import graph_examples


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
        self.assertNotIn("CWE rule", text)
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

    def test_wait_ready_shows_the_engine_error_even_when_it_is_far_above_the_tail(self):
        # formato reale di vLLM 0.33: il blocco d'errore del motore, poi la sua copia senza prefisso e, in fondo, il traceback
        # dell'API server (lungo) che dice solo "See root cause above"
        p = "(EngineCore pid=47909) ERROR 11-05 12:11:38 [core.py:1584] "
        engine = [
            p + "EngineCore failed to start.",
            p + "Traceback (most recent call last):",
            p + '  File "/v/vllm/v1/engine/core.py", line 1555, in run_engine_core',
            p + "    engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)",
            p + "                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^",
            p + '  File "/v/vllm/model_executor/layers/vocab_parallel_embedding.py", line 519, in weight_loader',
            p + "    param[: loaded_weight.shape[0]].data.copy_(loaded_weight)",
            p + "RuntimeError: The size of tensor a (3136) must match the size of tensor b (1568) at non-singleton dimension 1",
        ]
        wrapper = ["(APIServer pid=47800) INFO 11-05 12:11:39 [utils.py:628] [shutdown] Process manager: send sigterm to process EngineCore",
                   "Traceback (most recent call last):"]
        wrapper += [f'  File "/v/vllm/entrypoints/frame{i}.py", line {i}, in wrapper' for i in range(60)]
        wrapper.append("RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}")
        with tempfile.TemporaryDirectory() as d:
            log_path = Path(d) / "vllm.log"
            log_path.write_text("\n".join(["(APIServer pid=47800) INFO starting"] + engine + ["Process EngineCore:"] + wrapper) + "\n",
                                encoding="utf-8")
            proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(1)"])
            proc.wait()
            with self.assertRaises(RuntimeError) as ctx:
                benchmark.wait_ready("http://127.0.0.1:9", proc, 5, log_path=log_path)
        message = str(ctx.exception)
        self.assertIn("RuntimeError: The size of tensor a (3136) must match the size of tensor b (1568)", message)
        self.assertIn("vocab_parallel_embedding.py", message)
        self.assertNotIn("(EngineCore pid=", message)  # prefisso tolto
        self.assertNotIn("^^^^", message)              # le righe dei cursori non servono
        self.assertNotIn("frame30.py", message)        # il traceback dell'API server non fa rumore
        self.assertIn(str(log_path), message)

    def test_engine_error_handles_logs_without_that_block(self):
        self.assertEqual(benchmark.engine_error("INFO tutto bene\nWARNING qualcosa\n"), "")
        self.assertEqual(benchmark.engine_error(""), "")

    def test_engine_error_takes_the_last_failure_and_stops_at_the_block_end(self):
        p = "(EngineCore pid=1) ERROR 11-05 12:00:00 [core.py:1] "
        text = "\n".join([p + "EngineCore failed to start.", p + "ValueError: vecchio",
                          p + "EngineCore failed to start.", p + "KeyError: 'nuovo'", "Process EngineCore:", "IndietroNo: fuori dal blocco"])
        self.assertEqual(benchmark.engine_error(text), "KeyError: 'nuovo'")

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


HEADER = re.compile(r"^### .*depth (\d+)\)$", re.M)
KB = common.load_kb()


def graph_block(depth: int, code: str | None = None) -> dict:
    name = f"fn_{depth}{depth}{depth}{depth}{depth}{depth}"
    code = code or f"void {name}(char * data)\n{{\n    use(data);\n}}"
    return {"file": "src/0a1b2c3d.c", "name": name, "start_line": 100 * (5 - depth), "end_line": 100 * (5 - depth) + code.count("\n"),
            "depth": depth, "lang": "c", "code": code}


def graph_example(n_blocks: int = 5, label: int = 1, pair_id: str | None = None, pad: int = 0) -> dict:
    """Esempio con una catena di `n_blocks` blocchi; `pad` righe di commento in più nel chiamante più esterno."""
    blocks = [graph_block(d) for d in range(n_blocks - 1, 0, -1)]
    blocks.append(graph_block(0, "void fn_000000(char * data)\n{\n    system(data);\n}"))
    if pad:
        blocks[0]["code"] += "\n" + "\n".join(f"// padding {i}" for i in range(pad))
    rec = {"sink": {"function": "system", "file": "src/0a1b2c3d.c", "line": blocks[-1]["start_line"] + 2, "cwe_id": "CWE-78"},
           "blocks": blocks, "truncated": False}
    ex = graph_examples.build_example(rec, KB, label=label, variant_kind="bad" if label else "g2b", origin=None,
                                      meta={"source": "juliet-nist", "synthetic": False, "track": "juliet", "group": "g"})
    ex["meta"]["pair_id"] = pair_id
    return ex


def headers(messages: list[dict]) -> list[int]:
    return [int(d) for d in HEADER.findall(messages[1]["content"])]


class GraphSystemsTests(unittest.TestCase):
    def test_sink_only_prompt_has_exactly_one_block_header(self):
        msgs = benchmark.rerender_messages(graph_example(), sink_only=True)
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])
        self.assertEqual(headers(msgs), [0])
        self.assertIn("system(data);", msgs[1]["content"])
        self.assertNotIn("use(data);", msgs[1]["content"])

    def test_max_depth_filters_blocks(self):
        ex = graph_example()
        self.assertEqual(headers(benchmark.rerender_messages(ex, max_depth=1)), [1, 0])
        self.assertEqual(headers(benchmark.rerender_messages(ex, max_depth=0)), [0])
        self.assertEqual(headers(benchmark.rerender_messages(ex)), [4, 3, 2, 1, 0])

    def test_full_rerender_equals_the_stored_prompt(self):
        ex = graph_example()
        self.assertEqual(benchmark.rerender_messages(ex), benchmark.slice_messages(ex))

    def test_example_without_trace_falls_back_to_stored_messages(self):
        legacy = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": "{}"}],
                  "meta": {"label": 1}}
        for kwargs in ({}, {"max_depth": 1}, {"sink_only": True}):
            self.assertEqual(benchmark.rerender_messages(legacy, **kwargs), benchmark.slice_messages(legacy))

    def test_rerendered_sink_only_prompt_is_the_builder_sink_only_prompt(self):
        """R6b: la premessa di S1 («lo stesso prompt solo-sink») vale per costruzione, non per due implementazioni parallele."""
        ex = graph_example()
        trace = ex["meta"]["trace"]
        built = graph_examples.build_example({"sink": trace["sink"], "blocks": trace["blocks"], "truncated": False}, KB, label=1,
                                             variant_kind="bad", origin=None, meta={"source": "x", "synthetic": False}, view="sink_only")
        self.assertEqual(benchmark.rerender_messages(ex, sink_only=True)[1]["content"], built["messages"][1]["content"])

    def test_noslice_prompt_of_a_graph_example_uses_the_sink_block_not_the_outermost_caller(self):
        user = benchmark.noslice_messages(graph_example())[1]["content"]
        self.assertIn("system(data);", user)
        self.assertNotIn("use(data);", user)

    def test_pair_report_accuracy_and_both_correct(self):
        examples = [graph_example(2, 1, "p1"), graph_example(2, 0, "p1"), graph_example(2, 1, "p2"), graph_example(2, 0, "p2")]
        report = benchmark.pair_report(examples, [True, False, True, True])
        self.assertEqual(report["pairs"], 2)
        self.assertEqual(report["accuracy"], 0.75)
        self.assertEqual(report["both_correct"], 0.5)

    def test_pair_report_is_none_without_pairs(self):
        self.assertIsNone(benchmark.pair_report([graph_example(2, 1), graph_example(2, 0)], [True, False]))

    def test_pair_report_ignores_incomplete_pairs_and_counts_invalid_outputs_as_not_vulnerable(self):
        examples = [graph_example(2, 1, "p1"), graph_example(2, 0, "p1"), graph_example(2, 1, "lonely")]
        report = benchmark.pair_report(examples, [None, False, True])
        self.assertEqual((report["pairs"], report["accuracy"], report["both_correct"]), (1, 0.5, 0.0))

    def test_pair_report_separates_length_from_flow_reasoning(self):
        """R19: sui gemelli di Juliet il vulnerabile è quasi sempre il più lungo; S1 va letto anche a lunghezza controllata."""
        examples = [graph_example(2, 1, "long", pad=40), graph_example(2, 0, "long"),      # vulnerabile molto più lungo
                    graph_example(2, 1, "close", pad=1), graph_example(2, 0, "close"),      # lunghezze vicine (entro il 15%)
                    graph_example(2, 1, "inverse"), graph_example(2, 0, "inverse", pad=40)]  # sicuro più lungo
        report = benchmark.pair_report(examples, [True, False] * 3)
        self.assertEqual(report["pairs"], 3)
        controlled = report["length_controlled"]
        self.assertEqual((controlled["pairs"], controlled["both_correct"]), (1, 1.0))
        baseline = report["longer_is_vulnerable"]
        self.assertEqual((baseline["pairs_with_different_length"], baseline["share"]), (3, round(2 / 3, 4)))

    def test_evaluate_adds_pairs_only_when_the_examples_have_pair_ids(self):
        paired = [graph_example(2, 1, "p"), graph_example(2, 0, "p")]
        self.assertEqual(benchmark.evaluate(paired, [True, False])["pairs"]["pairs"], 1)
        self.assertNotIn("pairs", benchmark.evaluate([graph_example(2, 1), graph_example(2, 0)], [True, False]))

    def test_pairs_table_has_one_row_per_system_with_pairs(self):
        paired = [graph_example(2, 1, "p"), graph_example(2, 0, "p")]
        results = {"llm-graph": benchmark.evaluate(paired, [True, False]), "semgrep": benchmark.evaluate([graph_example()], [True])}
        table = benchmark.pairs_table(results)
        self.assertIn("llm-graph", table)
        self.assertNotIn("semgrep", table)


class GraphAccuracyCommandTests(unittest.TestCase):
    def test_parser_accepts_the_new_systems_and_depth(self):
        args = benchmark.parse_args(["accuracy", "--systems", "llm-graph", "llm-sink-only", "--depth", "3"])
        self.assertEqual((args.systems, args.depth), (["llm-graph", "llm-sink-only"], 3))
        self.assertIsNone(benchmark.parse_args(["accuracy"]).depth)
        self.assertLessEqual({"llm-graph", "llm-sink-only"}, set(benchmark.SYSTEMS))

    def test_each_system_gets_its_own_prompt_and_the_report_has_pairs(self):
        examples = [graph_example(4, 1, "p"), graph_example(4, 0, "p")]
        with tempfile.TemporaryDirectory() as d:
            test = Path(d) / "pairs.jsonl"
            test.write_text("".join(json.dumps(e) + "\n" for e in examples), encoding="utf-8")
            seen = {}

            def fake_run_llm(url, model, message_sets, concurrency, max_tokens=768):
                seen[len(seen)] = [headers(m) for m in message_sets]
                return [{"content": json.dumps({"vulnerable": i % 2 == 0})} for i, _ in enumerate(message_sets)]

            out = Path(d) / "r.json"
            argv = ["accuracy", "--test", str(test), "--systems", "llm-graph", "llm-sink-only", "--depth", "1", "--out", str(out)]
            with mock.patch.object(benchmark, "run_llm", side_effect=fake_run_llm), contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(benchmark.main(argv), 0)
            self.assertEqual(seen[0], [[1, 0], [1, 0]], "llm-graph: profondità fino a --depth")
            self.assertEqual(seen[1], [[0], [0]], "llm-sink-only: solo il blocco del sink")
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(report["llm-graph"]["pairs"]["both_correct"], 1.0)
            self.assertEqual(report["llm-sink-only"]["pairs"]["both_correct"], 1.0)
            self.assertIn("Coppie", out.with_suffix(".md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
