"""Funcoes utilitarias compartilhadas pelo projeto."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def salvar_csv(df: pd.DataFrame, caminho: Path) -> None:
    """Persiste um DataFrame em CSV UTF-8 com BOM."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminho, index=False, encoding="utf-8-sig")
    print(f"CSV salvo: {caminho}")


def normalizar_texto(texto: Any) -> str:
    """Remove excesso de espacos e padroniza valores nulos."""
    return re.sub(r"\s+", " ", str(texto or "")).strip()


def converter_para_float(valor: Any, padrao: float = 0.0) -> float:
    """Converte textos numericos com virgula decimal para float."""
    texto = normalizar_texto(valor).replace(",", ".")
    if not texto:
        return padrao
    try:
        return float(texto)
    except ValueError:
        return padrao


def carregar_json_seguro(valor: Any, padrao: Any) -> Any:
    """Interpreta JSON textual e devolve um valor padrao em caso de falha."""
    if valor is None:
        return padrao
    texto = str(valor).strip()
    if not texto:
        return padrao
    try:
        return json.loads(texto)
    except Exception:
        return padrao


def timestamp_execucao() -> str:
    """Gera o carimbo temporal da execucao em UTC."""
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def extrair_json_bruto(texto: str | None) -> str | None:
    """Recupera o ultimo objeto JSON completo presente em um texto."""
    if not texto:
        return None
    decodificador = json.JSONDecoder()
    candidatos_validos: list[str] = []
    for indice, caractere in enumerate(texto):
        if caractere != "{":
            continue
        try:
            # Busca o ultimo JSON bem-formado em saidas que podem conter texto extra do modelo.
            objeto, consumidos = decodificador.raw_decode(texto[indice:])
            if isinstance(objeto, dict):
                candidatos_validos.append(texto[indice:indice + consumidos])
        except Exception:
            continue
    return candidatos_validos[-1] if candidatos_validos else None


def segmentar_sentencas(texto: str, minimo_caracteres: int = 10) -> list[str]:
    """Segmenta sentencas preservando um fallback para respostas curtas."""
    texto_normalizado = normalizar_texto(texto)
    if not texto_normalizado:
        return []
    # A segmentacao permite comparar resposta e gabarito por partes menores na etapa semantica.
    partes = re.split(r"(?<=[.!?;])\s+", texto_normalizado)
    sentencas = [parte.strip() for parte in partes if len(parte.strip()) >= minimo_caracteres]
    # Se o texto nao tiver pontuacao util, a avaliacao segue com o bloco inteiro.
    return sentencas or [texto_normalizado]


def arredondar_numericos(df: pd.DataFrame, casas: int = 2) -> pd.DataFrame:
    """Arredonda todas as colunas numericas de um DataFrame."""
    saida = df.copy()
    for coluna in saida.columns:
        if pd.api.types.is_numeric_dtype(saida[coluna]):
            saida[coluna] = saida[coluna].round(casas)
    return saida
