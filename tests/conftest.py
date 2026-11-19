"""Configuración común de los tests."""

import pytest


@pytest.fixture(autouse=True)
def _sin_api_de_claude(monkeypatch):
    """Ningún test llama a la API real de Claude (cuesta plata y depende de la red).

    Con ANTHROPIC_API_KEY en el entorno, `build_assistant` armaría el asistente real y las
    simulaciones de los modos consultarían a Claude. Los tests del asistente usan clientes falsos.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
