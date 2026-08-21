import importlib.util
import sys
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parent.parent / "src" / "vault_mcp" / "vault.py"
spec = importlib.util.spec_from_file_location("vault_mcp_vault", MODULE_PATH)
vault = importlib.util.module_from_spec(spec)
sys.modules["vault_mcp_vault"] = vault
spec.loader.exec_module(vault)


def _sesion(fecha: str, texto: str) -> str:
    return f"**Sesión del {fecha} ({texto})**\n- {texto}"


HOT_SIN_ULTIMA_SESION_RESTO = """## Cuellos de botella activos

- Ítem corto de cuello de botella.

## Hechos verificados

- Hecho verificado.

## Reglas generales

- Regla general.

## Fallas abiertas y pendientes

### Decisiones pendientes
- [ ] Pendiente de ejemplo.
"""


def _hot_con_ultima_sesion(entradas: list[str]) -> str:
    cuerpo = "\n\n".join(entradas)
    return f"## Última sesión\n{cuerpo}\n\n{HOT_SIN_ULTIMA_SESION_RESTO}"


class TestBudgetHotText(unittest.TestCase):
    def test_bajo_presupuesto_no_cambia_nada(self):
        texto = _hot_con_ultima_sesion([_sesion("1/1/2026", "corta")])
        resultado = vault._budget_hot_text(texto, budget=10_000)
        self.assertEqual(resultado, texto)

    def test_excede_solo_por_ultima_sesion_recorta_las_mas_viejas(self):
        entrada_vieja = _sesion("1/1/2026", "vieja " * 200)
        entrada_reciente = _sesion("20/8/2026", "reciente " * 5)
        texto = _hot_con_ultima_sesion([entrada_vieja, entrada_reciente])
        resto_size = len(texto) - len(
            vault._ultima_sesion_section(texto).group(0)
        )
        budget = resto_size + len(entrada_reciente) + 50
        resultado = vault._budget_hot_text(texto, budget=budget)
        self.assertIn(entrada_reciente.strip(), resultado)  # completa, sin recortar
        self.assertNotIn(entrada_vieja.strip(), resultado)  # bloque completo, no el resumen
        self.assertIn("entrada(s) más antigua(s) no incluidas", resultado)
        self.assertIn("wiki/log.md", resultado)
        self.assertIn("2026-01-01", resultado)  # resumen de la excluida sigue presente

    def test_resto_solo_ya_excede_presupuesto_cero_entradas_incluidas(self):
        entrada = _sesion("20/8/2026", "cualquier cosa " * 50)
        texto = _hot_con_ultima_sesion([entrada])
        resto_size = len(texto) - len(
            vault._ultima_sesion_section(texto).group(0)
        )
        budget = resto_size - 100  # menor al resto solo
        resultado = vault._budget_hot_text(texto, budget=budget)
        self.assertNotIn(entrada, resultado)  # bloque completo, no el resumen
        self.assertIn("entrada(s) más antigua(s) no incluidas", resultado)
        self.assertIn("2026-08-20", resultado)  # resumen sigue presente

    def test_secciones_fuera_de_ultima_sesion_nunca_se_tocan(self):
        entrada_vieja = _sesion("1/1/2026", "vieja " * 200)
        texto = _hot_con_ultima_sesion([entrada_vieja])
        resultado = vault._budget_hot_text(texto, budget=200)
        self.assertIn("## Cuellos de botella activos", resultado)
        self.assertIn("Ítem corto de cuello de botella.", resultado)
        self.assertIn("## Reglas generales", resultado)
        self.assertIn("### Decisiones pendientes", resultado)

    def test_ultima_sesion_vacia_no_rompe(self):
        texto = f"## Última sesión\n\n{HOT_SIN_ULTIMA_SESION_RESTO}"
        resultado = vault._budget_hot_text(texto, budget=10)
        self.assertEqual(resultado, texto)

    def test_ultima_sesion_ausente_no_rompe(self):
        texto = HOT_SIN_ULTIMA_SESION_RESTO
        resultado = vault._budget_hot_text(texto, budget=10)
        self.assertEqual(resultado, texto)


class TestGetPrioridadesUnaffected(unittest.TestCase):
    """get_prioridades() is the only internal caller of get_hot() (confirmed
    via `grep -n "get_hot\\b" src/vault_mcp/vault.py`). It extracts
    '## Cuellos de botella activos', which lives after 'Última sesión' in
    the file but is never touched by the budget trim — this must survive
    any of the 3 trimming scenarios above."""

    def _run_get_prioridades_on(self, texto: str) -> str:
        section_match = vault._cuellos_de_botella_section(texto)
        section = section_match.group(0).strip()
        return vault.re.sub(r"\n+---\s*$", "", section).strip()

    def test_cuellos_de_botella_completo_tras_recorte_agresivo(self):
        entrada_vieja = _sesion("1/1/2026", "vieja " * 500)
        texto = _hot_con_ultima_sesion([entrada_vieja])
        recortado = vault._budget_hot_text(texto, budget=200)
        original = self._run_get_prioridades_on(texto)
        despues = self._run_get_prioridades_on(recortado)
        self.assertEqual(original, despues)
        self.assertIn("Ítem corto de cuello de botella.", despues)


if __name__ == "__main__":
    unittest.main()
