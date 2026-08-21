import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parent.parent / "src" / "vault_mcp" / "fts.py"
spec = importlib.util.spec_from_file_location("vault_mcp_fts", MODULE_PATH)
fts = importlib.util.module_from_spec(spec)
sys.modules["vault_mcp_fts"] = fts
spec.loader.exec_module(fts)


class TestFtsIndexing(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.vault_root = Path(self.tmpdir.name)
        (self.vault_root / "wiki").mkdir()
        self.db_path = self.vault_root / "fts_index.db"

        # Point the module at our temp vault/db instead of the real ones.
        fts.VAULT_ROOT = self.vault_root
        fts.FTS_DB_PATH = self.db_path

        (self.vault_root / "wiki" / "gatos.md").write_text(
            "Los gatos duermen dieciseis horas por dia.", encoding="utf-8"
        )
        (self.vault_root / "wiki" / "perros.md").write_text(
            "Los perros necesitan paseos diarios y mucho ejercicio.", encoding="utf-8"
        )
        (self.vault_root / "wiki" / "ambos.md").write_text(
            "Gatos y perros pueden convivir si se los presenta con cuidado.", encoding="utf-8"
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_index_fts_dry_run_reports_without_writing(self):
        report = fts.index_fts(dry_run=True)
        self.assertIn("3 páginas nuevas/modificadas", report)
        self.assertFalse(self.db_path.exists())

    def test_index_fts_writes_and_is_idempotent(self):
        first = fts.index_fts(dry_run=False)
        self.assertIn("3 páginas nuevas/modificadas", first)
        self.assertTrue(self.db_path.exists())

        second = fts.index_fts(dry_run=False)
        self.assertIn("Nada para indexar", second)

    def test_index_fts_incremental_reindex(self):
        fts.index_fts(dry_run=False)
        (self.vault_root / "wiki" / "gatos.md").write_text(
            "Los gatos duermen dieciseis horas por dia y cazan de noche.", encoding="utf-8"
        )
        report = fts.index_fts(dry_run=False)
        self.assertIn("1 páginas nuevas/modificadas", report)

    def test_index_fts_deletes_stale(self):
        fts.index_fts(dry_run=False)
        (self.vault_root / "wiki" / "perros.md").unlink()
        report = fts.index_fts(dry_run=False)
        self.assertIn("1 obsoleta", report)

    def test_search_wiki_ranked_orders_by_relevance(self):
        fts.index_fts(dry_run=False)
        results = fts.search_wiki_ranked("gatos", max_results=10)
        page_ids = [r[0] for r in results]
        self.assertIn("gatos.md", page_ids)
        self.assertIn("ambos.md", page_ids)
        self.assertNotIn("perros.md", page_ids)
        # gatos.md mentions "gatos" more centrally/frequently than ambos.md
        self.assertEqual(page_ids[0], "gatos.md")

    def test_search_wiki_ranked_empty_index_returns_empty_list(self):
        results = fts.search_wiki_ranked("gatos", max_results=10)
        self.assertEqual(results, [])

    def test_search_wiki_public_format_and_empty_index_message(self):
        empty_msg = fts.search_wiki("gatos")
        self.assertIn("índice FTS5 está vacío", empty_msg)

        fts.index_fts(dry_run=False)
        out = fts.search_wiki("gatos")
        self.assertIn("# Resultados de búsqueda: 'gatos'", out)
        self.assertIn("## gatos.md", out)

    def test_search_wiki_no_results(self):
        fts.index_fts(dry_run=False)
        out = fts.search_wiki("elefantes")
        self.assertEqual(out, "Sin resultados para 'elefantes'")

    def test_search_wiki_ranked_handles_hyphenated_query_without_crash(self):
        fts.index_fts(dry_run=False)
        # Hyphens are FTS5 NOT-operator syntax if unescaped — must not raise.
        results = fts.search_wiki_ranked("llm-wiki gatos", max_results=10)
        self.assertIsInstance(results, list)


if __name__ == "__main__":
    unittest.main()
