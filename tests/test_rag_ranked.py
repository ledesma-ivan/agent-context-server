import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

MODULE_PATH = Path(__file__).parent.parent / "src" / "vault_mcp" / "rag.py"
spec = importlib.util.spec_from_file_location("vault_mcp_rag", MODULE_PATH)
rag = importlib.util.module_from_spec(spec)
sys.modules["vault_mcp_rag"] = rag
spec.loader.exec_module(rag)


class TestSearchSemanticRanked(unittest.TestCase):
    def test_returns_similarity_scores_higher_is_better(self):
        fake_collection = MagicMock()
        fake_collection.count.return_value = 2
        fake_collection.query.return_value = {
            "ids": [["a.md", "b.md"]],
            "distances": [[0.1, 0.4]],
        }
        with patch.object(rag, "get_collection", return_value=fake_collection), \
             patch.object(rag, "_embed", return_value=[[0.0]]):
            result = rag.search_semantic_ranked("query", n_results=2)

        self.assertEqual(result, [("a.md", 0.9), ("b.md", 0.6)])

    def test_empty_collection_returns_empty_list(self):
        fake_collection = MagicMock()
        fake_collection.count.return_value = 0
        with patch.object(rag, "get_collection", return_value=fake_collection):
            result = rag.search_semantic_ranked("query", n_results=5)
        self.assertEqual(result, [])


class TestIndexVaultCallsFts(unittest.TestCase):
    def test_index_vault_dry_run_includes_fts_report(self):
        fake_collection = MagicMock()
        fake_collection.get.return_value = {"ids": [], "metadatas": []}
        with patch.object(rag, "get_collection", return_value=fake_collection), \
             patch("vault_mcp_rag.VAULT_ROOT") as fake_root:
            fake_root.__truediv__.return_value.rglob.return_value = []
            with patch.object(rag.fts, "index_fts", return_value="[dry_run FTS5] fake report") as mock_fts:
                report = rag.index_vault(dry_run=True)
        mock_fts.assert_called_once_with(dry_run=True)
        self.assertIn("fake report", report)


if __name__ == "__main__":
    unittest.main()
