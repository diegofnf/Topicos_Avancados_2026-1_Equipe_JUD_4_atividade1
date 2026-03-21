from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from config import MODELO_BERTSCORE, MODELO_EMBEDDINGS


COLUNA_NOTA = "nota"
COLUNAS_TEXTO_CANDIDATAS = ("resposta", "texto_resposta")
COLUNAS_MODELO_CANDIDATAS = ("modelo", "modelo_candidato")


def _validar_colunas(df: pd.DataFrame, colunas: Iterable[str]) -> None:
    ausentes = sorted(set(colunas) - set(df.columns))
    if ausentes:
        raise ValueError(f"Colunas obrigatorias ausentes: {', '.join(ausentes)}")


def _resolver_coluna(df: pd.DataFrame, candidatas: Sequence[str], descricao: str) -> str:
    for coluna in candidatas:
        if coluna in df.columns:
            return coluna
    raise ValueError(f"Nao foi encontrada uma coluna valida para {descricao}.")


def _normalizar_texto(valor: object) -> str:
    texto = "" if pd.isna(valor) else str(valor)
    return texto.strip()


@lru_cache(maxsize=2)
def _carregar_modelo_embedding(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("Instale 'sentence-transformers' para gerar embeddings.") from exc

    return SentenceTransformer(model_name)


def media_notas_por_modelo(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula a media de notas agrupada por modelo."""
    _validar_colunas(df, [COLUNA_NOTA])
    coluna_modelo = _resolver_coluna(df, COLUNAS_MODELO_CANDIDATAS, "modelo")
    if df.empty:
        return pd.DataFrame(columns=[coluna_modelo, "nota_media"])

    resultado = (
        df.groupby(coluna_modelo, dropna=False)[COLUNA_NOTA]
        .mean()
        .reset_index(name="nota_media")
    )
    return resultado.sort_values(coluna_modelo).reset_index(drop=True)


def variancia_notas(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula a variancia das notas por modelo."""
    _validar_colunas(df, [COLUNA_NOTA])
    coluna_modelo = _resolver_coluna(df, COLUNAS_MODELO_CANDIDATAS, "modelo")
    if df.empty:
        return pd.DataFrame(columns=[coluna_modelo, "variancia"])

    resultado = (
        df.groupby(coluna_modelo, dropna=False)[COLUNA_NOTA]
        .var(ddof=0)
        .fillna(0.0)
        .reset_index(name="variancia")
    )
    return resultado.sort_values(coluna_modelo).reset_index(drop=True)


def calcular_tamanho_medio(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula o tamanho medio das respostas em numero de palavras por modelo."""
    coluna_modelo = _resolver_coluna(df, COLUNAS_MODELO_CANDIDATAS, "modelo")
    coluna_texto = _resolver_coluna(df, COLUNAS_TEXTO_CANDIDATAS, "texto de resposta")
    if df.empty:
        return pd.DataFrame(columns=[coluna_modelo, "tamanho_medio"])

    base = df[[coluna_modelo, coluna_texto]].copy()
    base[coluna_texto] = base[coluna_texto].map(_normalizar_texto)
    base["qtd_palavras"] = base[coluna_texto].map(lambda texto: len(texto.split()) if texto else 0)

    resultado = (
        base.groupby(coluna_modelo, dropna=False)["qtd_palavras"]
        .mean()
        .reset_index(name="tamanho_medio")
    )
    return resultado.sort_values(coluna_modelo).reset_index(drop=True)


def calcular_flesch(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula a legibilidade media de Flesch por modelo."""
    try:
        import textstat
    except ImportError as exc:
        raise ImportError("Instale 'textstat' para calcular o indice de Flesch.") from exc

    coluna_modelo = _resolver_coluna(df, COLUNAS_MODELO_CANDIDATAS, "modelo")
    coluna_texto = _resolver_coluna(df, COLUNAS_TEXTO_CANDIDATAS, "texto de resposta")
    if df.empty:
        return pd.DataFrame(columns=[coluna_modelo, "flesch"])

    try:
        textstat.set_lang("pt")
    except Exception:
        pass

    base = df[[coluna_modelo, coluna_texto]].copy()
    base[coluna_texto] = base[coluna_texto].map(_normalizar_texto)

    def _pontuar(texto: str) -> float:
        if not texto:
            return np.nan
        try:
            return float(textstat.flesch_reading_ease(texto))
        except Exception:
            return np.nan

    base["flesch"] = base[coluna_texto].map(_pontuar)
    resultado = (
        base.groupby(coluna_modelo, dropna=False)["flesch"]
        .mean()
        .fillna(0.0)
        .reset_index()
    )
    return resultado.sort_values(coluna_modelo).reset_index(drop=True)


def calcular_bertscore(
    lista_respostas_a: Sequence[str],
    lista_respostas_b: Sequence[str],
    model_type: str = MODELO_BERTSCORE,
) -> tuple[float, list[float]]:
    """Calcula o BERTScore F1 medio e os scores individuais em lote."""
    if len(lista_respostas_a) != len(lista_respostas_b):
        raise ValueError("As listas para BERTScore devem ter o mesmo tamanho.")

    pares = [
        (_normalizar_texto(texto_a), _normalizar_texto(texto_b))
        for texto_a, texto_b in zip(lista_respostas_a, lista_respostas_b)
        if _normalizar_texto(texto_a) and _normalizar_texto(texto_b)
    ]
    if not pares:
        return 0.0

    try:
        from bert_score import score
    except ImportError as exc:
        raise ImportError("Instale 'bert-score' para calcular o BERTScore.") from exc

    candidatos = [candidato for candidato, _ in pares]
    referencias = [referencia for _, referencia in pares]
    _, _, f1 = score(candidatos, referencias, lang="pt", model_type=model_type, verbose=False)
    scores_individuais = [float(valor) for valor in f1.tolist()]
    return float(f1.mean().item()), scores_individuais


def gerar_embeddings(lista_textos: Sequence[str], model_name: str = MODELO_EMBEDDINGS) -> np.ndarray:
    """Gera embeddings para uma lista de textos usando sentence-transformers."""
    textos_validos = [_normalizar_texto(texto) for texto in lista_textos if _normalizar_texto(texto)]
    if not textos_validos:
        return np.empty((0, 0), dtype=float)

    modelo = _carregar_modelo_embedding(model_name)
    embeddings = modelo.encode(textos_validos, convert_to_numpy=True, normalize_embeddings=True)
    return np.asarray(embeddings, dtype=float)


def calcular_bertscore_por_modelo(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula a concordancia media por modelo via BERTScore entre respostas da mesma questao."""
    coluna_modelo = _resolver_coluna(df, COLUNAS_MODELO_CANDIDATAS, "modelo")
    coluna_texto = _resolver_coluna(df, COLUNAS_TEXTO_CANDIDATAS, "texto de resposta")
    _validar_colunas(df, ["question_id", coluna_modelo, coluna_texto])
    if df.empty:
        return pd.DataFrame(columns=[coluna_modelo, "bertscore_concordancia"])

    pares_candidatos = []
    pares_referencias = []
    metadados_pares = []
    base = df[["question_id", coluna_modelo, coluna_texto]].copy()
    base[coluna_texto] = base[coluna_texto].map(_normalizar_texto)

    for question_id, grupo in base.groupby("question_id", dropna=False):
        grupo = grupo[grupo[coluna_texto] != ""]
        if len(grupo) < 2:
            continue

        itens = list(grupo[[coluna_modelo, coluna_texto]].itertuples(index=False, name=None))
        for i, (modelo_a, texto_a) in enumerate(itens):
            for modelo_b, texto_b in itens[i + 1 :]:
                pares_candidatos.append(texto_a)
                pares_referencias.append(texto_b)
                metadados_pares.append((question_id, modelo_a, modelo_b))

    if not metadados_pares:
        return pd.DataFrame(columns=[coluna_modelo, "bertscore_concordancia"])

    _, scores_individuais = calcular_bertscore(pares_candidatos, pares_referencias)

    registros = []
    for (question_id, modelo_a, modelo_b), score_ab in zip(metadados_pares, scores_individuais):
        registros.append(
            {"question_id": question_id, coluna_modelo: modelo_a, "bertscore_concordancia": score_ab}
        )
        registros.append(
            {"question_id": question_id, coluna_modelo: modelo_b, "bertscore_concordancia": score_ab}
        )

    if not registros:
        return pd.DataFrame(columns=[coluna_modelo, "bertscore_concordancia"])

    return (
        pd.DataFrame(registros)
        .groupby(coluna_modelo, dropna=False)["bertscore_concordancia"]
        .mean()
        .reset_index()
        .sort_values(coluna_modelo)
        .reset_index(drop=True)
    )


def calcular_matriz_similaridade(embeddings: np.ndarray) -> np.ndarray:
    """Calcula a matriz de similaridade cosseno a partir dos embeddings."""
    matriz = np.asarray(embeddings, dtype=float)
    if matriz.size == 0:
        return np.empty((0, 0), dtype=float)
    if matriz.ndim != 2:
        raise ValueError("Os embeddings devem estar em uma matriz 2D.")

    normas = np.linalg.norm(matriz, axis=1, keepdims=True)
    normas[normas == 0] = 1.0
    matriz_normalizada = matriz / normas
    return matriz_normalizada @ matriz_normalizada.T
