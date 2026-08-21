import importlib.util
import sys
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parent.parent / "src" / "vault_mcp" / "vault.py"
spec = importlib.util.spec_from_file_location("vault_mcp_vault_rrf", MODULE_PATH)
vault = importlib.util.module_from_spec(spec)
sys.modules["vault_mcp_vault_rrf"] = vault
spec.loader.exec_module(vault)


class TestRrfFuse(unittest.TestCase):
    def test_single_list_preserves_order(self):
        result = vault._rrf_fuse([[("a", 9.0), ("b", 5.0), ("c", 1.0)]], k=60)
        self.assertEqual([p for p, _ in result], ["a", "b", "c"])

    def test_agreement_across_lists_wins(self):
        # "b" is #2 in both lists; "a" is #1 in one but absent from the other.
        list1 = [("a", 9.0), ("b", 5.0)]
        list2 = [("b", 8.0), ("c", 3.0)]
        result = vault._rrf_fuse([list1, list2], k=60)
        top = result[0][0]
        self.assertEqual(top, "b")

    def test_hand_computed_scores(self):
        # k=1 for easy hand math: score(page) = sum(1 / (1 + rank)) over
        # the lists it appears in, rank is 0-indexed.
        list1 = [("a", 1.0), ("b", 1.0)]  # a: rank 0, b: rank 1
        list2 = [("b", 1.0), ("a", 1.0)]  # b: rank 0, a: rank 1
        result = vault._rrf_fuse([list1, list2], k=1)
        scores = dict(result)
        # a: 1/(1+0) + 1/(1+1) = 1.0 + 0.5 = 1.5
        # b: 1/(1+1) + 1/(1+0) = 0.5 + 1.0 = 1.5
        self.assertAlmostEqual(scores["a"], 1.5)
        self.assertAlmostEqual(scores["b"], 1.5)

    def test_page_missing_from_one_list_still_scored(self):
        list1 = [("a", 9.0), ("b", 5.0), ("c", 1.0)]
        list2 = [("c", 9.0)]  # "a" and "b" absent here
        result = vault._rrf_fuse([list1, list2], k=60)
        page_ids = [p for p, _ in result]
        self.assertIn("a", page_ids)
        self.assertIn("b", page_ids)
        self.assertIn("c", page_ids)

    def test_empty_lists_return_empty(self):
        self.assertEqual(vault._rrf_fuse([[], []], k=60), [])
        self.assertEqual(vault._rrf_fuse([], k=60), [])

    def test_dedup_by_page_id(self):
        # Same page_id appearing more than once within a single list
        # (shouldn't happen from real callers, but must not double-count
        # silently into a broken score) — last occurrence's rank wins.
        result = vault._rrf_fuse([[("a", 9.0), ("a", 1.0)]], k=60)
        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
