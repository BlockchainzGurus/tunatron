"""Test di localize.py: dove sta il sink *vero* in una funzione (diff della fix, commenti FLAW di Juliet)."""
import unittest

import common
import localize

KB = common.load_kb()

VULN = """static int copy_name(char *dst, const char *src)
{
    char *p = malloc(16);
    strcpy(dst, src);
    return helper(dst);
}"""

FIXED = """static int copy_name(char *dst, const char *src)
{
    char *p = malloc(16);
    strncpy(dst, src, 15);
    return helper(dst);
}"""


class FindCallsTests(unittest.TestCase):
    def test_python_ignores_decorators_header_comments_and_strings(self):
        code = (
            "@app.route('/x')\n"
            "def view(request):\n"
            "    # os.system('x')\n"
            "    q = f\"SELECT {request.args['id']}\"\n"
            "    cur.execute(q)\n"
            "    s = '''eval(x)'''\n"
            "    return os.system(cmd)\n"
        )
        self.assertEqual(localize.find_calls(code, "python"), [("cur.execute", 5), ("os.system", 7)])

    def test_python_multiline_header_is_not_a_call(self):
        code = "def f(a,\n      b=g(1)):\n    run(a)\n"
        self.assertEqual(localize.find_calls(code, "python"), [("run", 3)])

    def test_c_default_language_is_unchanged(self):
        self.assertEqual(localize.find_calls("void f() {\n  strcpy(a, b);\n}"), [("strcpy", 2)])


class NameTests(unittest.TestCase):
    def test_python_and_c_names(self):
        self.assertEqual(localize.extract_function_name("@x.y(1)\nasync def handler(request):\n    pass", "python"), "handler")
        self.assertEqual(localize.extract_function_name("void Foo::bar(int n)\n{\n}", "cpp"), "bar")


class ChangedLinesTests(unittest.TestCase):
    def test_replaced_line_is_reported_on_the_old_side(self):
        got = localize.changed_lines(VULN, FIXED)
        self.assertEqual((got["changed"], got["boundary"]), ({4}, set()))

    def test_added_check_marks_the_lines_around_the_insertion_point(self):
        old = "a\nb\nc\nd"
        new = "a\nb\nCHECK\nc\nd"
        got = localize.changed_lines(old, new)
        self.assertEqual((got["changed"], got["boundary"]), (set(), {2, 3}))

    def test_whitespace_only_differences_are_ignored(self):
        got = localize.changed_lines("if (x) {\n    y();\n}", "if (x) {\n\ty();\n}")
        self.assertEqual((got["changed"], got["boundary"]), (set(), set()))

    def test_identical_and_missing_counterpart(self):
        self.assertEqual(localize.changed_lines("a", "a"), {"changed": set(), "boundary": set()})
        self.assertIsNone(localize.changed_lines("a", None))

    def test_first_changed_line(self):
        self.assertEqual(localize.first_changed_line(VULN, FIXED), 4)
        self.assertIsNone(localize.first_changed_line(VULN, VULN))
        self.assertIsNone(localize.first_changed_line(VULN, None))


class LocalizeSinkTests(unittest.TestCase):
    def test_sink_on_the_changed_line_beats_an_earlier_unrelated_call(self):
        s = localize.localize_sink(KB, VULN, "c", FIXED)
        self.assertEqual((s["entry"]["id"], s["callee"], s["line"], s["how"]), ("CWE-120", "strcpy", 4, "diff"))

    def test_the_fixed_side_asks_about_its_own_changed_line(self):
        s = localize.localize_sink(KB, FIXED, "c", VULN)
        self.assertEqual((s["entry"]["id"], s["callee"], s["line"]), ("CWE-120", "strncpy", 4))

    def test_sinks_on_unchanged_lines_are_not_localized(self):
        old = "void f() {\n  memcpy(a, b, n);\n  x = 1;\n}"
        new = "void f() {\n  memcpy(a, b, n);\n  x = 2;\n}"
        self.assertIsNone(localize.localize_sink(KB, old, "c", new))

    def test_check_added_before_the_call_localizes_the_call(self):
        old = "void f(char *d, char *s, int n) {\n  memcpy(d, s, n);\n}"
        new = "void f(char *d, char *s, int n) {\n  if (n > 16) return;\n  memcpy(d, s, n);\n}"
        s = localize.localize_sink(KB, old, "c", new)
        self.assertEqual((s["callee"], s["line"]), ("memcpy", 2))

    def test_preferred_cwe_filters_candidates(self):
        self.assertIsNone(localize.localize_sink(KB, VULN, "c", FIXED, preferred_cwe="CWE-78"))
        self.assertEqual(localize.localize_sink(KB, VULN, "c", FIXED, preferred_cwe="CWE-120")["callee"], "strcpy")

    def test_python_sink(self):
        old = "def q(cur, uid):\n    cur.execute(f\"SELECT * FROM t WHERE id={uid}\")\n"
        new = "def q(cur, uid):\n    cur.execute(\"SELECT * FROM t WHERE id=%s\", (uid,))\n"
        s = localize.localize_sink(KB, old, "python", new)
        self.assertEqual((s["entry"]["id"], s["callee"], s["line"]), ("CWE-89", "cur.execute", 2))

    def test_no_counterpart_means_no_localization(self):
        self.assertIsNone(localize.localize_sink(KB, VULN, "c", None))


JULIET_BAD = """void CWE690_NULL_Deref_From_Return__char_realloc_01_bad()
{
    char * data;
    data = NULL; /* Initialize data */
    /* POTENTIAL FLAW: Allocate memory without checking if the memory allocation function failed */
    data = (char *)realloc(data, 20*sizeof(char));
    /* FLAW: Initialize memory buffer without checking to see if the memory allocation function failed */
    strcpy(data, "Initialize");
    printLine(data);
    free(data);
}"""


class FlawHintTests(unittest.TestCase):
    def test_the_call_after_the_flaw_comment_is_the_sink(self):
        self.assertEqual(localize.flaw_sink(JULIET_BAD), {"callee": "strcpy", "line": 8})

    def test_potential_flaw_is_used_only_when_there_is_no_plain_flaw(self):
        code = "void f()\n{\n    /* POTENTIAL FLAW: x */\n    data = realloc(data, 5);\n    use(data);\n}"
        self.assertEqual(localize.flaw_sink(code), {"callee": "realloc", "line": 4})

    def test_no_flaw_comment(self):
        self.assertIsNone(localize.flaw_sink("void f()\n{\n    run();\n}"))


class FixHintTests(unittest.TestCase):
    def test_the_call_after_the_fix_comment_is_the_sink_of_the_good_variant(self):
        code = "static void goodG2B()\n{\n    char *data = malloc(10);\n    /* FIX: check first */\n    if (data) { free(data); }\n}"
        self.assertEqual(localize.fix_sink(code), {"callee": "free", "line": 5})

    def test_no_fix_comment(self):
        self.assertIsNone(localize.fix_sink("void f()\n{\n    run();\n}"))


JULIET_FILE = """/* Copyright
{ braces in a comment } */
#include "std_testcase.h"

#ifndef OMITBAD

void CWE121_bad()
{
    char buf[10];
    /* FLAW: too big */
    strcpy(buf, "this string { is } too long");
}

#endif /* OMITBAD */

static void goodG2B()
{
    char buf[10];
    /* FIX: ok */
    strncpy(buf, "ok", 9);
}

void good()
{
    goodG2B();
}

int main(int argc, char * argv[])
{
    return 0;
}
"""


class SplitFunctionsTests(unittest.TestCase):
    def test_functions_are_split_with_names_lines_and_code(self):
        funcs = localize.split_c_functions(JULIET_FILE)
        self.assertEqual([f["name"] for f in funcs], ["CWE121_bad", "goodG2B", "good", "main"])
        bad = funcs[0]
        self.assertEqual(bad["start_line"], 7)
        self.assertTrue(bad["code"].startswith("void CWE121_bad()\n{"))
        self.assertTrue(bad["code"].rstrip().endswith("}"))
        self.assertIn("this string { is } too long", bad["code"])
        self.assertEqual(funcs[1]["start_line"], 16)

    def test_structs_and_prototypes_are_not_functions(self):
        src = "struct s { int a; };\nint proto(int);\nint real(void)\n{\n  return 1;\n}\n"
        self.assertEqual([f["name"] for f in localize.split_c_functions(src)], ["real"])


PY_FILE = """import os

@app.route("/x")
def top(a):
    def inner():
        pass
    return a

class K:
    @staticmethod
    def method(self, b):
        run(b)

    async def amethod(self):
        await go()

x = 1
"""


class SplitPythonTests(unittest.TestCase):
    def test_functions_and_methods_are_split_but_nested_defs_stay_inside(self):
        funcs = localize.split_python_functions(PY_FILE)
        self.assertEqual([f["name"] for f in funcs], ["top", "method", "amethod"])
        top = funcs[0]
        self.assertEqual(top["start_line"], 3, "il decoratore fa parte della funzione")
        self.assertTrue(top["code"].startswith('@app.route("/x")\ndef top(a):'))
        self.assertIn("def inner():", top["code"])
        self.assertTrue(top["code"].rstrip().endswith("return a"))
        self.assertTrue(funcs[1]["code"].startswith("    @staticmethod\n    def method"))
        self.assertNotIn("x = 1", funcs[2]["code"])


class ScrubTests(unittest.TestCase):
    def test_label_comments_are_removed_but_line_numbers_are_preserved(self):
        code = (
            "def f(c, u):\n"
            "    # GOOD -- parameters\n"
            "    c.execute(q, u)\n"
            "    # a normal explanation stays\n"
            "    c.execute('x' % u) # $ Alert\n"
            "    c.execute('y' % u)  # BAD: formatting\n"
        )
        got = localize.scrub_labels(code, "#")
        lines = got.split("\n")
        self.assertEqual(len(lines), len(code.split("\n")))
        self.assertEqual(lines[1], "")
        self.assertEqual(lines[2], "    c.execute(q, u)")
        self.assertEqual(lines[3], "    # a normal explanation stays")
        self.assertEqual(lines[4], "    c.execute('x' % u)")
        self.assertEqual(lines[5], "    c.execute('y' % u)")

    def test_cpp_comment_style(self):
        self.assertEqual(localize.scrub_labels("strcpy(a, b); // $ Alert\nx(); // GOOD", "//"), "strcpy(a, b);\nx();")

    def test_annotated_lines(self):
        code = "a()\nb() # $ Alert\nc() # $ MISSING: Alert\nd() # $ SPURIOUS: Alert\ne() # $ Source"
        got = localize.annotated_lines(code, "#")
        self.assertEqual(got, {"alert": {2}, "ambiguous": {3, 4}})


if __name__ == "__main__":
    unittest.main()
