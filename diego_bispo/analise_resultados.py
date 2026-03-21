from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from config import (
    AVALIACAO_DISCURSIVAS_CSV,
    BENCHMARK_DISCURSIVAS_CSV,
    BENCHMARK_OBJETIVAS_CSV,
    HEATMAP_DISCURSIVAS_PNG,
    RESPOSTAS_DISCURSIVAS_CSV,
    RESPOSTAS_OBJETIVAS_CSV,
    SIMILARIDADE_DISCURSIVAS_CSV,
)
from data_utils import carregar_csv, salvar_csv
from metricas_quantitativas import (
    _serie_texto,
    acuracia_por_modelo,
    calcular_acuracia,
    calcular_f1_score,
    matriz_confusao,
)
from metricas_qualitativas import (
    calcular_bertscore_por_modelo,
    calcular_flesch,
    calcular_matriz_similaridade,
    calcular_tamanho_medio,
    media_notas_por_modelo,
    variancia_notas,
)
from visualizacoes import plotar_heatmap_similaridade


def _coluna_modelo_discursiva(df: pd.DataFrame) -> str:
    if "modelo_candidato" in df.columns:
        return "modelo_candidato"
    if "modelo" in df.columns:
        return "modelo"
    raise ValueError("Nao foi encontrada coluna de modelo para as discursivas.")


def _renomear_modelo(df: pd.DataFrame, coluna_modelo: str) -> pd.DataFrame:
    if coluna_modelo == "modelo":
        return df
    return df.rename(columns={coluna_modelo: "modelo"})


def gerar_benchmark_objetivas(df_objetivas: pd.DataFrame) -> pd.DataFrame:
    """Gera benchmark consolidado das questoes objetivas por modelo."""
    if df_objetivas.empty:
        return pd.DataFrame(columns=["modelo", "acuracia", "f1"])

    base = acuracia_por_modelo(df_objetivas)
    f1_por_modelo = (
        df_objetivas.groupby("modelo", dropna=False)
        .apply(
            lambda grupo: f1_score(
                _serie_texto(grupo, "gabarito_oficial"),
                _serie_texto(grupo, "resposta"),
                average="macro",
                zero_division=0,
            )
        )
        .reset_index(name="f1")
    )
    return (
        base.merge(f1_por_modelo, on="modelo", how="left")
        .sort_values("modelo")
        .reset_index(drop=True)
    )


def gerar_benchmark_discursivas(df_discursivas: pd.DataFrame) -> pd.DataFrame:
    """Gera benchmark qualitativo das respostas discursivas por modelo."""
    coluna_modelo = _coluna_modelo_discursiva(df_discursivas)
    base = _renomear_modelo(media_notas_por_modelo(df_discursivas), coluna_modelo)
    variancia = _renomear_modelo(variancia_notas(df_discursivas), coluna_modelo)
    tamanho = _renomear_modelo(calcular_tamanho_medio(df_discursivas), coluna_modelo)
    flesch = _renomear_modelo(calcular_flesch(df_discursivas), coluna_modelo)
    bertscore_concordancia = _renomear_modelo(calcular_bertscore_por_modelo(df_discursivas), coluna_modelo)

    return (
        base.merge(variancia, on="modelo", how="outer")
        .merge(tamanho, on="modelo", how="outer")
        .merge(flesch, on="modelo", how="outer")
        .merge(bertscore_concordancia, on="modelo", how="outer")
        .sort_values("modelo")
        .reset_index(drop=True)
    )


def gerar_matriz_similaridade_discursivas(df_respostas_discursivas: pd.DataFrame) -> pd.DataFrame:
    """Gera uma matriz de similaridade entre modelos a partir da media dos embeddings."""
    from metricas_qualitativas import gerar_embeddings

    if df_respostas_discursivas.empty:
        return pd.DataFrame()

    if "modelo" not in df_respostas_discursivas.columns or "resposta" not in df_respostas_discursivas.columns:
        raise ValueError("O DataFrame de respostas discursivas deve conter as colunas 'modelo' e 'resposta'.")

    modelos = []
    vetores_medios = []

    for modelo, grupo in df_respostas_discursivas.groupby("modelo", dropna=False):
        textos = grupo["resposta"].fillna("").astype(str).tolist()
        embeddings = gerar_embeddings(textos)
        if embeddings.size == 0:
            continue
        modelos.append(str(modelo))
        vetores_medios.append(embeddings.mean(axis=0))

    if not vetores_medios:
        return pd.DataFrame()

    matriz = calcular_matriz_similaridade(np.vstack(vetores_medios))
    return pd.DataFrame(matriz, index=modelos, columns=modelos)


def executar_analise(
    respostas_objetivas_path: str | Path = RESPOSTAS_OBJETIVAS_CSV,
    avaliacao_discursivas_path: str | Path = AVALIACAO_DISCURSIVAS_CSV,
    respostas_discursivas_path: str | Path = RESPOSTAS_DISCURSIVAS_CSV,
    benchmark_objetivas_path: str | Path = BENCHMARK_OBJETIVAS_CSV,
    benchmark_discursivas_path: str | Path = BENCHMARK_DISCURSIVAS_CSV,
    similaridade_discursivas_path: str | Path = SIMILARIDADE_DISCURSIVAS_CSV,
    heatmap_discursivas_path: str | Path = HEATMAP_DISCURSIVAS_PNG,
) -> dict[str, pd.DataFrame | float]:
    """Executa a analise completa de resultados e salva os artefatos gerados."""
    df_objetivas = carregar_csv(Path(respostas_objetivas_path))
    df_discursivas = carregar_csv(Path(avaliacao_discursivas_path))
    df_respostas_discursivas = carregar_csv(Path(respostas_discursivas_path))

    acuracia_global = calcular_acuracia(df_objetivas)
    f1_global = calcular_f1_score(df_objetivas)
    df_benchmark_obj = gerar_benchmark_objetivas(df_objetivas)
    df_benchmark_disc = gerar_benchmark_discursivas(df_discursivas)
    df_benchmark_final = (
        df_benchmark_obj.merge(df_benchmark_disc, on="modelo", how="outer")
        .sort_values("modelo")
        .reset_index(drop=True)
    )
    df_matriz_confusao = matriz_confusao(df_objetivas)
    df_similaridade = gerar_matriz_similaridade_discursivas(df_respostas_discursivas)

    salvar_csv(df_benchmark_obj, Path(benchmark_objetivas_path))
    salvar_csv(df_benchmark_final, Path(benchmark_discursivas_path))

    if not df_similaridade.empty:
        matriz_serializavel = df_similaridade.copy()
        matriz_serializavel.insert(0, "modelo", matriz_serializavel.index)
        salvar_csv(matriz_serializavel.reset_index(drop=True), Path(similaridade_discursivas_path))
        figura, _ = plotar_heatmap_similaridade(
            df_similaridade.values,
            labels=df_similaridade.index.tolist(),
            output_path=heatmap_discursivas_path,
        )
        try:
            import matplotlib.pyplot as plt

            plt.close(figura)
        except Exception:
            pass
    else:
        salvar_csv(pd.DataFrame(), Path(similaridade_discursivas_path))

    return {
        "acuracia_global": acuracia_global,
        "f1_global": f1_global,
        "benchmark_objetivas": df_benchmark_obj,
        "benchmark_final": df_benchmark_final,
        "matriz_confusao": df_matriz_confusao,
        "similaridade_discursivas": df_similaridade,
    }
