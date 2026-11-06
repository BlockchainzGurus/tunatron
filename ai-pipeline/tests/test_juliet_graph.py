"""Test di `juliet_graph`: archivio NIST, sorgenti `.c`, sink ed etichette. Gli zip si costruiscono al volo: nessuna rete."""
import hashlib
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import common
import juliet_graph
from juliet_graph import (
    JULIET_CWES,
    JULIET_SHA256,
    JULIET_URL,
    MACROS,
    extract_c_sources,
    fetch_juliet,
    flow_variant,
    sink_label,
    sink_targets,
    source_token,
    testcase_group,
    variant_kind,
)

KB = common.load_kb()
ZIP_NAME = JULIET_URL.rsplit("/", 1)[-1]
CMD = "CWE78_OS_Command_Injection__char_environment_system"

# Funzione `goodG2B` di un test case a file singolo: `SYSTEM(data);` sta alla riga 12 del file.
GOOD_G2B = """\
#include "std_testcase.h"

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
static void goodG2B()
{
    char * data;
    char data_buf[100] = "";
    data = data_buf;
    data[0] = 0;
    SYSTEM(data);
}

void CWE78_OS_Command_Injection__char_environment_system_01_good()
{
    goodG2B();
}

#endif
"""


def with_call(call: str) -> str:
    """`GOOD_G2B` con la chiamata alla riga 12 sostituita da `call`."""
    return GOOD_G2B.replace("SYSTEM(data);", call)


def make_zip(path: Path, members: dict[str, bytes | str]) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return path


def tree(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


class TempDirCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)


class FetchTests(TempDirCase):
    def test_constants(self):
        self.assertEqual(JULIET_CWES, ("CWE-120", "CWE-78", "CWE-190", "CWE-416"))
        self.assertEqual(ZIP_NAME, "2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3.zip")
        self.assertEqual(JULIET_SHA256, "ada9d7e1c323d283446df3f55bdee0d00bda1fed786785fe98764d58688f38eb")

    def test_fetch_juliet_rejects_wrong_sha256(self):
        (self.tmp / ZIP_NAME).write_bytes(b"non e' lo zip di Juliet")
        with mock.patch("urllib.request.urlopen", side_effect=AssertionError("niente rete")):
            with self.assertRaises(ValueError):
                fetch_juliet(self.tmp)
        self.assertTrue((self.tmp / ZIP_NAME).exists(), "il file locale sbagliato non va cancellato")

    def test_fetch_juliet_returns_existing_file_without_downloading(self):
        data = b"zip finto ma con lo sha256 giusto"
        (self.tmp / ZIP_NAME).write_bytes(data)
        with mock.patch.object(juliet_graph, "JULIET_SHA256", hashlib.sha256(data).hexdigest()), \
                mock.patch("urllib.request.urlopen", side_effect=AssertionError("niente rete")):
            self.assertEqual(fetch_juliet(self.tmp), self.tmp / ZIP_NAME)

    def test_fetch_juliet_downloads_when_missing_and_checks_sha256(self):
        data = b"contenuto scaricato"
        dest = self.tmp / "nuova" / "cartella"
        with mock.patch.object(juliet_graph, "JULIET_SHA256", hashlib.sha256(data).hexdigest()), \
                mock.patch("urllib.request.urlopen", return_value=io.BytesIO(data)) as urlopen:
            path = fetch_juliet(dest)
        self.assertEqual(path, dest / ZIP_NAME)
        self.assertEqual(path.read_bytes(), data)
        self.assertEqual(urlopen.call_args.args[0], JULIET_URL)
        self.assertEqual(tree(dest), {ZIP_NAME}, "il file temporaneo deve sparire dopo il rename")

    def test_fetch_juliet_discards_download_with_wrong_sha256(self):
        dest = self.tmp / "dest"
        with mock.patch("urllib.request.urlopen", return_value=io.BytesIO(b"corrotto")):
            with self.assertRaises(ValueError):
                fetch_juliet(dest)
        self.assertEqual(tree(dest), set(), "uno zip scaricato male non deve restare col nome definitivo")


class ExtractTests(TempDirCase):
    def test_extract_writes_only_selected_cwes_and_c_files(self):
        archive = make_zip(self.tmp / "j.zip", {
            f"C/testcases/CWE78_OS_Command_Injection/s01/{CMD}_01.c": "int a;\n",
            f"C/testcases/CWE78_OS_Command_Injection/s01/{CMD}_01.cpp": "int cpp;\n",
            "C/testcases/CWE78_OS_Command_Injection/s01/Makefile": "all:\n",
            "C/testcases/CWE121_Stack_Based_Buffer_Overflow/CWE121_Stack_Based_Buffer_Overflow__CWE131_loop_01.c": "int b;\n",
            "C/testcases/CWE415_Double_Free/s01/CWE415_Double_Free__malloc_free_char_01.c": "int c;\n",
            "C/testcases/CWE23_Relative_Path_Traversal/s01/CWE23_Relative_Path_Traversal__char_console_fopen_01.c": "int d;\n",
            "C/testcases/CWE401_Memory_Leak/s01/CWE401_Memory_Leak__malloc_01.c": "int e;\n",
            "C/testcasesupport/io.c": "int f;\n",
            "C/testcases/CWE78_OS_Command_Injection/readme.txt": "x\n",
        })
        out = self.tmp / "src"
        roots = extract_c_sources(archive, out, KB)
        self.assertEqual(roots, [
            out / "CWE121_Stack_Based_Buffer_Overflow",
            out / "CWE415_Double_Free",
            out / "CWE78_OS_Command_Injection",
        ])
        self.assertEqual(tree(out), {
            f"CWE78_OS_Command_Injection/s01/{CMD}_01.c",
            "CWE121_Stack_Based_Buffer_Overflow/CWE121_Stack_Based_Buffer_Overflow__CWE131_loop_01.c",
            "CWE415_Double_Free/s01/CWE415_Double_Free__malloc_free_char_01.c",
        })

    def test_extract_honours_cwes_argument(self):
        archive = make_zip(self.tmp / "j.zip", {
            f"C/testcases/CWE78_OS_Command_Injection/s01/{CMD}_01.c": "int a;\n",
            "C/testcases/CWE415_Double_Free/s01/CWE415_Double_Free__malloc_free_char_01.c": "int c;\n",
        })
        out = self.tmp / "src"
        roots = extract_c_sources(archive, out, KB, cwes=("CWE-416",))
        self.assertEqual(roots, [out / "CWE415_Double_Free"])
        self.assertEqual(tree(out), {"CWE415_Double_Free/s01/CWE415_Double_Free__malloc_free_char_01.c"})

    def test_extract_normalizes_crlf_and_invalid_bytes(self):
        name = f"C/testcases/CWE78_OS_Command_Injection/s01/{CMD}_01.c"
        archive = make_zip(self.tmp / "j.zip", {name: b"int a;\r\n/* \xff\xfe */\r\nint b;\rint c;\r\n"})
        out = self.tmp / "src"
        (root,) = extract_c_sources(archive, out, KB)
        raw = (root / "s01" / f"{CMD}_01.c").read_bytes()
        self.assertNotIn(b"\r", raw)
        text = raw.decode("utf-8")  # niente UnicodeDecodeError: i byte non validi sono già sostituiti
        self.assertEqual(text, "int a;\n/* �� */\nint b;\nint c;\n")

    def test_extract_refuses_members_outside_destination(self):
        archive = make_zip(self.tmp / "j.zip", {
            f"C/testcases/CWE78_OS_Command_Injection/s01/{CMD}_01.c": "int ok;\n",
            "C/testcases/CWE78_X/../../../evil.c": "int evil;\n",
        })
        out = self.tmp / "dentro" / "src"
        with self.assertRaises(ValueError):
            extract_c_sources(archive, out, KB)
        self.assertFalse((self.tmp / "evil.c").exists())
        self.assertFalse((self.tmp / "dentro" / "evil.c").exists())
        self.assertEqual(tree(out) if out.exists() else set(), set(), "il controllo va fatto prima di scrivere qualunque file")

    def test_extract_refuses_members_that_leave_their_cwe_directory(self):
        archive = make_zip(self.tmp / "j.zip", {"C/testcases/CWE78_X/../CWE190_Y/a.c": "int a;\n"})
        with self.assertRaises(ValueError):
            extract_c_sources(archive, self.tmp / "src", KB)


class TargetsCase(TempDirCase):
    def write(self, rel: str, text: str) -> Path:
        path = self.tmp / "root" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="")
        return path

    def targets(self, rel: str, text: str) -> list[dict]:
        self.write(rel, text)
        return sink_targets(self.tmp / "root", KB)


class SinkTargetTests(TargetsCase):
    def test_sink_targets_map_macros_and_give_absolute_lines(self):
        self.assertEqual(
            self.targets(f"s01/{CMD}_01.c", GOOD_G2B),
            [{"file": f"s01/{CMD}_01.c", "line": 12, "callee": "system", "cwe_id": "CWE-78"}],
        )

    def test_targets_follow_the_catalog_of_the_testcase_cwe(self):
        # `strcat` è un sink della CWE-120, non della 78: in un test case CWE-78 non conta.
        self.assertEqual(self.targets(f"s01/{CMD}_01.c", with_call("strcat(data, data);")), [])
        # Un test case CWE-121 (alias della 120) riconosce `snprintf` ma non `system`.
        name = "CWE121_Stack_Based_Buffer_Overflow__CWE805_char_declare_snprintf_01.c"
        self.assertEqual(self.targets(name, with_call("system(data);")), [])
        self.assertEqual(
            self.targets(name, with_call("SNPRINTF(data, 10, data);")),
            [{"file": name, "line": 12, "callee": "snprintf", "cwe_id": "CWE-120"}],
        )

    def test_only_bad_and_good_functions_are_scanned(self):
        source = """\
static void helper(char * data)
{
    system(data);
}

void CWE78_OS_Command_Injection__char_environment_system_54e_badSink(char * data)
{
    system(data);
}

static void goodB2G(char * data)
{
    system(data);
}

int main(int argc, char * argv[])
{
    system("ls");
}
"""
        self.assertEqual(
            [(t["line"], t["callee"]) for t in self.targets(f"s02/{CMD}_54e.c", source)],
            [(8, "system"), (13, "system")],
        )

    def test_lines_are_absolute_for_multiline_calls_and_comments(self):
        source = """\
/* intestazione
   su piu' righe */
#define X 1

void helper(void) { system("no"); }

/* commento
   prima della funzione */
static void goodG2B()
{
    /* system("nel commento"); */
    char * s = "system(\\"nella stringa\\")";
    SYSTEM
        (s);
}
"""
        self.assertEqual(
            [(t["line"], t["callee"]) for t in self.targets(f"s01/{CMD}_01.c", source)],
            [(13, "system")],
        )

    def test_targets_are_sorted_by_file_then_line(self):
        self.write(f"s02/{CMD}_02.c", with_call("SYSTEM(data); SYSTEM(data);"))
        self.write(f"s01/{CMD}_03.c", GOOD_G2B)
        self.write(f"s01/{CMD}_01.c", GOOD_G2B)
        got = sink_targets(self.tmp / "root", KB)
        self.assertEqual([(t["file"], t["line"]) for t in got], [
            (f"s01/{CMD}_01.c", 12),
            (f"s01/{CMD}_03.c", 12),
            (f"s02/{CMD}_02.c", 12),
            (f"s02/{CMD}_02.c", 12),
        ])

    def test_files_without_a_known_cwe_prefix_give_no_targets(self):
        self.assertEqual(self.targets("s01/helper_01.c", GOOD_G2B), [])

    def test_crlf_source_gives_same_lines_after_extraction(self):
        name = f"C/testcases/CWE78_OS_Command_Injection/s01/{CMD}_01.c"
        crlf = GOOD_G2B.replace("\n", "\r\n").replace("data_buf", "d\xff_buf").encode("latin-1")
        archive = make_zip(self.tmp / "j.zip", {name: crlf})
        (root,) = extract_c_sources(archive, self.tmp / "src", KB)
        self.assertEqual(
            sink_targets(root, KB),
            [{"file": f"s01/{CMD}_01.c", "line": 12, "callee": "system", "cwe_id": "CWE-78"}],
        )


class MacroTests(TargetsCase):
    """Ogni macro in maiuscolo di Juliet va riportata al nome vero della funzione che il catalogo conosce."""

    def macro_target(self, macro: str, file_name: str) -> dict:
        (target,) = self.targets(file_name, with_call(f"{macro}(data);"))
        return target

    def assert_macro(self, macro: str, file_name: str, callee: str, cwe_id: str):
        self.assertEqual(MACROS[macro], callee)
        target = self.macro_target(macro, file_name)
        self.assertEqual((target["callee"], target["cwe_id"], target["line"]), (callee, cwe_id, 12))

    def test_macro_system(self):
        self.assert_macro("SYSTEM", f"s01/{CMD}_01.c", "system", "CWE-78")

    def test_macro_execl(self):
        self.assert_macro("EXECL", "s01/CWE78_OS_Command_Injection__char_console_execl_01.c", "execl", "CWE-78")

    def test_macro_execlp(self):
        self.assert_macro("EXECLP", "s01/CWE78_OS_Command_Injection__char_console_execlp_01.c", "execlp", "CWE-78")

    def test_macro_execv(self):
        self.assert_macro("EXECV", "s01/CWE78_OS_Command_Injection__char_console_w32_execv_01.c", "execv", "CWE-78")

    def test_macro_execvp(self):
        self.assert_macro("EXECVP", "s01/CWE78_OS_Command_Injection__char_console_w32_execvp_01.c", "execvp", "CWE-78")

    def test_macro_popen(self):
        self.assert_macro("POPEN", "s01/CWE78_OS_Command_Injection__char_console_popen_01.c", "popen", "CWE-78")

    def test_macro_snprintf(self):
        self.assert_macro(
            "SNPRINTF", "CWE121_Stack_Based_Buffer_Overflow__CWE805_char_declare_snprintf_01.c", "snprintf", "CWE-120")


class NameTests(unittest.TestCase):
    def test_sink_label(self):
        self.assertEqual(sink_label(f"{CMD}_54e_badSink"), 1)
        self.assertEqual(sink_label(f"{CMD}_54e_goodG2BSink"), 0)
        self.assertIsNone(sink_label("main"))
        self.assertIsNone(sink_label("helper"))
        # nomi senza il prefisso del test case (funzioni `static`) e dispatcher globali
        self.assertEqual(sink_label("bad"), 1)
        self.assertEqual(sink_label("badSink"), 1)
        self.assertEqual(sink_label("goodG2B"), 0)
        self.assertEqual(sink_label("good1"), 0)
        self.assertEqual(sink_label(f"{CMD}_01_bad"), 1)
        self.assertEqual(sink_label(f"{CMD}_01_good"), 0)
        # una funzione che «contiene» bad/good senza cominciare così non è etichettata
        self.assertIsNone(sink_label("helperBad"))
        self.assertIsNone(sink_label("printLine"))

    def test_variant_kind(self):
        self.assertEqual(variant_kind(f"{CMD}_54e_goodG2BSink"), "g2b")
        self.assertEqual(variant_kind("goodG2B"), "g2b")
        self.assertEqual(variant_kind("goodG2B2"), "g2b")
        self.assertEqual(variant_kind(f"{CMD}_54e_goodB2GSink"), "b2g")
        self.assertEqual(variant_kind("goodB2G"), "b2g")
        self.assertEqual(variant_kind("goodB2G1"), "b2g")
        self.assertEqual(variant_kind(f"{CMD}_54e_badSink"), "bad")
        self.assertEqual(variant_kind(f"{CMD}_01_bad"), "bad")
        self.assertEqual(variant_kind("good1"), "good")
        self.assertEqual(variant_kind(f"{CMD}_01_good"), "good")
        self.assertIsNone(variant_kind("main"))
        self.assertIsNone(variant_kind("helper"))

    def test_group_variant_and_source(self):
        name = f"{CMD}_54e.c"
        self.assertEqual(testcase_group(name), f"{CMD}_54")
        self.assertEqual(flow_variant(name), "54")
        self.assertEqual(source_token(name), "environment")
        single = f"{CMD}_01.c"
        self.assertEqual(testcase_group(single), f"{CMD}_01")
        self.assertEqual(flow_variant(single), "01")
        # i file di una catena condividono gruppo, variante e sorgente
        for letter in "abcde":
            self.assertEqual(testcase_group(f"{CMD}_54{letter}.c"), f"{CMD}_54")

    def test_names_accept_a_relative_path(self):
        self.assertEqual(testcase_group(f"s02/{CMD}_54e.c"), f"{CMD}_54")
        self.assertEqual(flow_variant(f"s02/{CMD}_54e.c"), "54")
        self.assertEqual(source_token(f"s02/{CMD}_54e.c"), "environment")

    def test_source_token_covers_the_juliet_naming_schemes(self):
        cases = {
            "CWE78_OS_Command_Injection__wchar_t_connect_socket_w32_execvp_61a.c": "connect_socket",
            "CWE78_OS_Command_Injection__char_listen_socket_popen_65b.c": "listen_socket",
            "CWE78_OS_Command_Injection__wchar_t_file_w32spawnl_54d.c": "file",
            "CWE78_OS_Command_Injection__char_console_execl_12.c": "console",
            "CWE190_Integer_Overflow__int_fgets_add_01.c": "fgets",
            "CWE190_Integer_Overflow__unsigned_int_max_square_22a.c": "max",
            "CWE191_Integer_Underflow__int64_t_min_sub_01.c": "min",
            "CWE190_Integer_Overflow__short_rand_multiply_01.c": "rand",
            "CWE197_Numeric_Truncation_Error__int_fscanf_to_char_01.c": "fscanf",
            "CWE121_Stack_Based_Buffer_Overflow__CWE129_large_01.c": "large",
            "CWE127_Buffer_Underread__CWE839_negative_01.c": "negative",
            "CWE680_Integer_Overflow_to_Buffer_Overflow__malloc_fixed_01.c": "fixed",
        }
        for name, token in cases.items():
            with self.subTest(name=name):
                self.assertEqual(source_token(name), token)
        # nessuna sorgente esterna nel nome
        self.assertIsNone(source_token("CWE415_Double_Free__malloc_free_char_01.c"))
        self.assertIsNone(source_token("CWE121_Stack_Based_Buffer_Overflow__CWE805_char_declare_loop_01.c"))

    def test_group_and_variant_on_odd_names(self):
        self.assertEqual(testcase_group("CWE416_Use_After_Free__return_freed_ptr_01.c"), "CWE416_Use_After_Free__return_freed_ptr_01")
        self.assertEqual(flow_variant("CWE416_Use_After_Free__return_freed_ptr_01.c"), "01")
        # senza cifre finali non c'è variante e il gruppo è il nome senza estensione
        self.assertIsNone(flow_variant("helper.c"))
        self.assertEqual(testcase_group("helper.c"), "helper")


if __name__ == "__main__":
    unittest.main()
