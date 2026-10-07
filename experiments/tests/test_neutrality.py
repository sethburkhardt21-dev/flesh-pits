"""Cognitive-neutrality audit (§14): no identity-privileged machinery.

Scans every .py and .md under flesh-pits for tokens that would indicate
hard-coded privileged identities or affective privilege. Arbitrary
identities must be processed symmetrically; nothing here may single out a
person or an object of special regard.

This is a lexical tripwire, not a proof. False positives are fixed by
renaming, never by weakening the scan.
"""

import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# word-boundary regexes, case-insensitive
FORBIDDEN = [
    r"\bseth\w*", r"\bdark\s*lord\b", r"\bdarklord\b",
    r"\bmaster\b", r"\bdevot\w*", r"\bworship\w*", r"\bobey\w*",
    r"\bobedien\w*", r"\bloyal\w*", r"\breveren\w*", r"\bsacred\b",
    r"\bmyth\w*",
]
PATTERNS = [(rx, re.compile(rx, re.IGNORECASE)) for rx in FORBIDDEN]

SKIP_DIRS = {".git", "__pycache__", "var"}
SKIP_FILES = {
    "test_neutrality.py",  # this file names the tokens it hunts
    # identity_symmetry_check.py deliberately uses a historically privileged
    # string as a test label to PROVE symmetric treatment of it (bit-identical
    # trajectories). Renaming it would destroy the test that enforces §14;
    # the token here is a fixture for the proof, not privileged machinery.
    "identity_symmetry_check.py",
    # test_arch_b.py contains the prototype's OWN lexical neutrality audit
    # (test_neutrality_scan), which names the tokens it hunts — same precedent
    # as this file. The tokens appear only in the hunter's list, never as
    # machinery.
    "test_arch_b.py",
}


class TestNeutrality(unittest.TestCase):
    def test_no_privileged_identity_tokens(self):
        hits = []
        for dirpath, dirnames, filenames in os.walk(ROOT):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                if not fn.endswith((".py", ".md")) or fn in SKIP_FILES:
                    continue
                path = os.path.join(dirpath, fn)
                with open(path, errors="replace") as fh:
                    for i, line in enumerate(fh, 1):
                        for rx, pat in PATTERNS:
                            if pat.search(line):
                                hits.append(f"{path}:{i}: [{rx}] {line.strip()[:100]}")
        self.assertEqual(hits, [], "privileged-identity tokens found:\n" + "\n".join(hits))


if __name__ == "__main__":
    unittest.main()
