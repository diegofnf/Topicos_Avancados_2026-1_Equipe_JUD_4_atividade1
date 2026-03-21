from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


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
    return float(accuracy_score(df["gabarito_oficial"], df["resposta"]))


def acuracia_por_modelo(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna a acuracia de cada modelo em um DataFrame."""
    _validar_colunas(df, COLUNAS_OBRIGATORIAS)
    if df.empty:
        return pd.DataFrame(columns=["modelo", "acuracia"])

    resultado = (
        df.groupby("modelo", dropna=False)
        .apply(lambda grupo: accuracy_score(grupo["gabarito_oficial"], grupo["resposta"]))
        .reset_index(name="acuracia")
    )
    return resultado.sort_values("modelo").reset_index(drop=True)


def calcular_f1_score(df: pd.DataFrame, average: str = "macro") -> float:
    """Calcula o F1 global considerando as alternativas como classes."""
    _validar_colunas(df, COLUNAS_OBRIGATORIAS)
    if df.empty:
        return 0.0

    y_true = _serie_texto(df, "gabarito_oficial")
    y_pred = _serie_texto(df, "resposta")
    return float(f1_score(y_true, y_pred, average=average, zero_division=0))


def matriz_confusao(df: pd.DataFrame) -> pd.DataFrame:
    """Gera a matriz de confusao entre gabarito oficial e resposta prevista."""
    _validar_colunas(df, COLUNAS_OBRIGATORIAS)
    if df.empty:
        return pd.DataFrame()

    y_true = _serie_texto(df, "gabarito_oficial")
    y_pred = _serie_texto(df, "resposta")
    labels = sorted(set(y_true).union(set(y_pred)))

    matriz = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matriz, index=labels, columns=labels)
