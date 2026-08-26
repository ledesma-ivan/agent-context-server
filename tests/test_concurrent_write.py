import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parent.parent / "src" / "vault_mcp" / "vault.py"
spec = importlib.util.spec_from_file_location("vault_mcp_vault_concurrent", MODULE_PATH)
vault = importlib.util.module_from_spec(spec)
sys.modules["vault_mcp_vault_concurrent"] = vault
spec.loader.exec_module(vault)


HOT_TEMPLATE = """## Fallas abiertas y pendientes

### Decisiones pendientes
- [ ] Pendiente original sin tocar.
"""


class TestConcurrentWriteGuard(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.vault_root = Path(self.tmpdir.name)
        (self.vault_root / "wiki").mkdir()
        vault.VAULT_ROOT = self.vault_root
        vault.ARCHIVE_PATH = self.vault_root / "wiki" / "pendientes-archivo.md"
        self.hot_path = self.vault_root / "wiki" / "hot.md"
        self.hot_path.write_text(HOT_TEMPLATE, encoding="utf-8")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_add_pendiente_detects_concurrent_change(self):
        """Simula el escenario real que motivó este cambio: dos escritores
        (ej. una sesión interactiva y el heartbeat de Hermes) leen hot.md
        casi al mismo tiempo. add_pendiente es una sola llamada sincrónica,
        así que no hay forma de intercalar un segundo hilo real en el medio
        -- en cambio, se monkeypatchea _atomic_write_if_unchanged para que,
        justo antes de que add_pendiente llegue al punto de escribir (ya con
        su propia lectura inicial hecha), un "otro escritor" modifique
        hot.md en disco primero. Eso reproduce exactamente la condición que
        _atomic_write_if_unchanged está pensada para detectar: el contenido
        en disco ya no coincide con lo que add_pendiente leyó al empezar.
        """
        original_helper = vault._atomic_write_if_unchanged
        external_write_done = {"done": False}

        def helper_with_race(path, expected_text, new_text):
            if not external_write_done["done"] and path == self.hot_path:
                # Otro escritor (Hermes, un cron) modifica hot.md por fuera
                # justo antes de que esta llamada intente escribir.
                self.hot_path.write_text(
                    HOT_TEMPLATE + "\n- [ ] Agregado por el otro escritor.\n",
                    encoding="utf-8",
                )
                external_write_done["done"] = True
            return original_helper(path, expected_text, new_text)

        vault._atomic_write_if_unchanged = helper_with_race
        try:
            resultado = vault.add_pendiente("Pendiente nuevo desde esta sesión")
        finally:
            vault._atomic_write_if_unchanged = original_helper

        self.assertIn("cambió desde que se leyó", resultado)
        contenido_final = self.hot_path.read_text(encoding="utf-8")
        # La modificación del "otro escritor" sigue intacta -- no se perdió.
        self.assertIn("Agregado por el otro escritor.", contenido_final)
        # Y la nuestra NO se aplicó por encima (habría sido una escritura perdida).
        self.assertNotIn("Pendiente nuevo desde esta sesión", contenido_final)

    def test_add_pendiente_writes_normally_without_race(self):
        resultado = vault.add_pendiente("Comprar entradas para el recital de noviembre")
        self.assertIn("Agregado a hot.md", resultado)
        contenido = self.hot_path.read_text(encoding="utf-8")
        self.assertIn("Comprar entradas para el recital de noviembre", contenido)

    def test_atomic_write_if_unchanged_raises_on_mismatch(self):
        self.hot_path.write_text("contenido real en disco", encoding="utf-8")
        with self.assertRaises(vault.ConcurrentWriteError):
            vault._atomic_write_if_unchanged(
                self.hot_path, "lo que yo creía que había", "texto nuevo"
            )
        # El archivo no se tocó.
        self.assertEqual(self.hot_path.read_text(encoding="utf-8"), "contenido real en disco")

    def test_atomic_write_if_unchanged_writes_when_matching(self):
        current = self.hot_path.read_text(encoding="utf-8")
        vault._atomic_write_if_unchanged(self.hot_path, current, "texto nuevo")
        self.assertEqual(self.hot_path.read_text(encoding="utf-8"), "texto nuevo")
        # No queda archivo temporal huérfano.
        self.assertFalse((self.hot_path.with_name(self.hot_path.name + ".tmp")).exists())


if __name__ == "__main__":
    unittest.main()
