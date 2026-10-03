"""2D runs must decompose with the core count the GUI is launching."""
import os
import tempfile
import unittest

from main_new import _sync_2d_decompose_subdomains


_DICT = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      decomposeParDict;
}

numberOfSubdomains 1;

method          scotch;
"""


class DecomposeSyncTests(unittest.TestCase):
    def test_requested_cores_replace_initialize_time_count(self):
        with tempfile.TemporaryDirectory() as root:
            system = os.path.join(root, "system")
            os.makedirs(system)
            path = os.path.join(system, "decomposeParDict")
            with open(path, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(_DICT)
            _sync_2d_decompose_subdomains(root, 2)
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
            self.assertIn("numberOfSubdomains 2;", text)
            self.assertNotIn("numberOfSubdomains 1;", text)
            self.assertIn("method          scotch;", text)

    def test_matching_count_is_left_unchanged(self):
        with tempfile.TemporaryDirectory() as root:
            system = os.path.join(root, "system")
            os.makedirs(system)
            path = os.path.join(system, "decomposeParDict")
            with open(path, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(_DICT.replace("1;", "4;"))
            with open(path, encoding="utf-8") as handle:
                before = handle.read()
            _sync_2d_decompose_subdomains(root, 4)
            with open(path, encoding="utf-8") as handle:
                after = handle.read()
            self.assertEqual(before, after)

    def test_missing_dictionary_is_ignored(self):
        with tempfile.TemporaryDirectory() as root:
            _sync_2d_decompose_subdomains(root, 3)

    def test_non_positive_cores_become_one(self):
        with tempfile.TemporaryDirectory() as root:
            system = os.path.join(root, "system")
            os.makedirs(system)
            path = os.path.join(system, "decomposeParDict")
            with open(path, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(_DICT.replace("1;", "8;"))
            _sync_2d_decompose_subdomains(root, 0)
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
            self.assertIn("numberOfSubdomains 1;", text)


if __name__ == "__main__":
    unittest.main()
