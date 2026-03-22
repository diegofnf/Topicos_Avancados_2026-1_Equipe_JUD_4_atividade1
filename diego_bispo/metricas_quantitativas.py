from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.metrics import accuracy_score


COLUNAS_OBRIGATORIAS = {"resposta", "gabarito_oficial", "correto", "modelo"}


def _validar_colunas(df: pd.DataFrame, colunas: Iterable[str]) -> None:
    ausentes = sorted(set(colunas) - set(df.columns))
    if ausentes:
        raise ValueError(f"Colunas obrigatorias ausentes: {', '.join(ausentes)}")


def _serie_texto(df: pd.DataFrame, coluna: str) -> pd.Series:
    return df[coluna].fillna("N/A").astype(str).str.strip()


def calcular_acuracia(df: pd.DataFrame) -> float:
    """Calcula a acuracia global de respostas objetivas."""
    _validar_colunas(df, COLUNAS_OBRIGATORIAS)
    if df.empty:
        return 0.0
    return float(accuracy_score(_serie_texto(df, "gabarito_oficial"), _serie_texto(df, "resposta")))


def acuracia_por_modelo(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna a acuracia de cada modelo em um DataFrame."""
    _validar_colunas(df, COLUNAS_OBRIGATORIAS)
    if df.empty:
        return pd.DataFrame(columns=["modelo", "acuracia"])

    resultado = (
        df.groupby("modelo", dropna=False)
        .apply(lambda grupo: accuracy_score(_serie_texto(grupo, "gabarito_oficial"), _serie_texto(grupo, "resposta")))
        .reset_index(name="acuracia")
    )
    return resultado.sort_values("modelo").reset_index(drop=True)
