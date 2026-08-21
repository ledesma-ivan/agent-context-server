import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).parent.parent / "src" / "vault_mcp" / "vault.py"
spec = importlib.util.spec_from_file_location("vault_mcp_vault_hybrid", MODULE_PATH)
vault = importlib.util.module_from_spec(spec)
sys.modules["vault_mcp_vault_hybrid"] = vault
spec.loader.exec_module(vault)


class TestSearchHybrid(unittest.TestCase):
    def test_no_results_anywhere(self):
        with patch("vault_mcp.fts.search_wiki_ranked", return_value=[]), \
             patch("vault_mcp.rag.search_semantic_ranked", return_value=[]):
            result = vault.search_hybrid("query nada")
        self.assertEqual(result, "Sin resultados para 'query nada'")

    def test_fuses_bm25_and_vector_and_graph(self):
        with patch(
            "vault_mcp.fts.search_wiki_ranked",
            return_value=[("a.md", 5.0), ("b.md", 2.0)],
        ), patch(
            "vault_mcp.rag.search_semantic_ranked",
            return_value=[("b.md", 0.9), ("c.md", 0.5)],
        ), patch.object(vault, "get_related_pages") as mock_related:
            # get_related_pages is called once per stage-1 anchor page; return
            # a page not present in the BM25/vector results, to prove graph
            # results make it into the final fusion. Real format:
            # "- [[page]] (score) — breakdown", see vault.py:1473-1481.
            mock_related.return_value = "- [[d.md]] (3.0) — link directo"

            result = vault.search_hybrid("query", n_results=5)

        self.assertIn("d.md", result)
        self.assertIn("b.md", result)  # appears in both bm25 and vector, should rank high

    def test_only_one_signal_has_results(self):
        with patch(
            "vault_mcp.fts.search_wiki_ranked", return_value=[("a.md", 5.0)]
        ), patch(
            "vault_mcp.rag.search_semantic_ranked", return_value=[]
        ), patch.object(vault, "get_related_pages") as mock_related:
            mock_related.return_value = "- Ninguna señal de relación encontrada (página aislada)."

            result = vault.search_hybrid("solo bm25")

        self.assertIn("a.md", result)

    def test_anchor_page_that_does_not_resolve_does_not_crash(self):
        with patch(
            "vault_mcp.fts.search_wiki_ranked", return_value=[("orphan.md", 5.0)]
        ), patch(
            "vault_mcp.rag.search_semantic_ranked", return_value=[]
        ), patch.object(vault, "get_related_pages") as mock_related:
            # get_related_pages returns a warning string (its real behavior
            # for a nonexistent page), not an exception.
            mock_related.return_value = "⚠️ No existe ninguna página `orphan.md.md` en wiki/"

            result = vault.search_hybrid("orphan query")

        self.assertIn("orphan.md", result)


if __name__ == "__main__":
    unittest.main()
