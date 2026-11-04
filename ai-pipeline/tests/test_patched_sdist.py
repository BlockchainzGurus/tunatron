"""Test di patched_sdist.py: sorgenti dei kernel Mamba da PyPI con -std=c++17 portato a c++20 (nessuna rete)."""
import hashlib
import io
import tarfile
import tempfile
import unittest
from pathlib import Path

import patched_sdist

SETUP = 'cc_flag.append("-std=c++17")\nextra = ["-O3", "-std=c++17"]\n'


def make_sdist(directory: Path, setup_text: str = SETUP, top: str = "pkg-1.0", extra_members: dict | None = None) -> tuple[Path, str]:
    archive = directory / f"{top}.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        for name, text in {f"{top}/setup.py": setup_text, f"{top}/pkg/__init__.py": "", **(extra_members or {})}.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return archive, hashlib.sha256(archive.read_bytes()).hexdigest()


class PatchTests(unittest.TestCase):
    def test_every_cxx17_flag_becomes_cxx20_and_the_count_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "setup.py").write_text(SETUP)
            self.assertEqual(patched_sdist.patch_cxx20(Path(d)), 2)
            text = (Path(d) / "setup.py").read_text()
        self.assertNotIn("c++17", text)
        self.assertEqual(text.count("-std=c++20"), 2)

    def test_second_run_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "setup.py").write_text(SETUP)
            patched_sdist.patch_cxx20(Path(d))
            self.assertEqual(patched_sdist.patch_cxx20(Path(d)), 0)

    def test_unknown_layout_is_an_error_not_a_silent_skip(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "setup.py").write_text("print('niente flag')\n")
            with self.assertRaisesRegex(RuntimeError, "c\\+\\+17"):
                patched_sdist.patch_cxx20(Path(d))


class ExtractTests(unittest.TestCase):
    def test_top_level_directory_is_stripped_so_the_destination_is_the_source_root(self):
        with tempfile.TemporaryDirectory() as d:
            archive, _ = make_sdist(Path(d))
            dest = Path(d) / "src"
            patched_sdist.extract_sdist(archive, dest)
            self.assertTrue((dest / "setup.py").exists())
            self.assertTrue((dest / "pkg" / "__init__.py").exists())
            self.assertFalse((dest / "pkg-1.0").exists())

    def test_members_escaping_the_destination_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            archive, _ = make_sdist(Path(d), extra_members={"pkg-1.0/../../evil.txt": "x"})
            dest = Path(d) / "a" / "src"
            with self.assertRaises(RuntimeError):
                patched_sdist.extract_sdist(archive, dest)
            self.assertFalse((Path(d) / "evil.txt").exists())
            self.assertFalse((Path(d) / "a" / "evil.txt").exists())


class FetchTests(unittest.TestCase):
    def fetch(self, d: Path, sha=None, dest_name="src"):
        archive, real_sha = make_sdist(d)
        lookup = lambda name, version: (f"https://example.invalid/{name}-{version}.tar.gz", f"{name}-{version}.tar.gz", sha or real_sha)
        download = lambda url, path: Path(path).write_bytes(archive.read_bytes())
        return patched_sdist.fetch_sdist("pkg", "1.0", d / dest_name, lookup=lookup, download=download)

    def test_fetch_extracts_and_returns_the_source_root(self):
        with tempfile.TemporaryDirectory() as d:
            root = self.fetch(Path(d))
            self.assertEqual(root, Path(d) / "src")
            self.assertIn("-std=c++17", (root / "setup.py").read_text())  # fetch non patcha: lo fa patch_cxx20

    def test_wrong_sha256_is_refused_before_anything_is_extracted(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError, "sha256"):
                self.fetch(Path(d), sha="0" * 64)
            self.assertFalse((Path(d) / "src" / "setup.py").exists())

    def test_previous_contents_of_the_destination_do_not_survive(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "src").mkdir()
            (Path(d) / "src" / "vecchio.o").write_text("oggetto compilato con un altro torch")
            self.fetch(Path(d))
            self.assertFalse((Path(d) / "src" / "vecchio.o").exists())

    def test_fetch_and_patch_returns_a_ready_to_build_tree(self):
        with tempfile.TemporaryDirectory() as d:
            archive, sha = make_sdist(Path(d))
            root, n = patched_sdist.fetch_and_patch(
                "pkg", "1.0", Path(d) / "src",
                lookup=lambda name, version: ("u", "f.tar.gz", sha), download=lambda url, path: Path(path).write_bytes(archive.read_bytes()))
            self.assertEqual(n, 2)
            self.assertNotIn("c++17", (root / "setup.py").read_text())


if __name__ == "__main__":
    unittest.main()
