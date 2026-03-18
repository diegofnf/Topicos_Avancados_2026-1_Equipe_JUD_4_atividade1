import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


def carregar_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def salvar_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    print(f"CSV salvo: {path}")


def timestamp_execucao() -> str:
    return datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat()


def extrair_json_bruto(texto: str) -> str | None:
    """
    Extrai o primeiro JSON válido do texto gerado pelo modelo.

    Estratégia:
    1. Tenta bloco markdown ```json ... ```
    2. Varre o texto caractere a caractere buscando objetos JSON válidos
       e retorna o último encontrado (mais completo)
    """
    texto = (texto or "").strip()

    blocos_md = re.findall(r"```json\s*(\{.*?\})\s*```", texto, flags=re.S | re.I)
    for candidato in reversed(blocos_md):
        try:
            json.loads(candidato)
            return candidato
        except Exception:
            pass

    decoder = json.JSONDecoder()
    candidatos_validos = []

    for i, ch in enumerate(texto):
        if ch != "{":
            continue
        try:
            obj, end = decoder.raw_decode(texto[i:])
            candidatos_validos.append(texto[i : i + end])
        except Exception:
            continue

    return candidatos_validos[-1] if candidatos_validos else None
