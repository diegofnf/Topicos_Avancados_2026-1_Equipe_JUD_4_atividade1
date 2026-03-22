from __future__ import annotations

from pathlib import Path

import pandas as pd

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
    acuracia_por_modelo,
    calcular_acuracia,
)
from metricas_qualitativas import (
    calcular_matriz_bertscore,
    media_notas_por_modelo,
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
        return pd.DataFrame(columns=["modelo", "acuracia"])
    return acuracia_por_modelo(df_objetivas)


def gerar_benchmark_discursivas(df_discursivas: pd.DataFrame) -> pd.DataFrame:
    """Gera benchmark qualitativo das respostas discursivas por modelo."""
    coluna_modelo = _coluna_modelo_discursiva(df_discursivas)
    return _renomear_modelo(media_notas_por_modelo(df_discursivas), coluna_modelo)


def gerar_matriz_bertscore_discursivas(df_respostas_discursivas: pd.DataFrame) -> pd.DataFrame:
    """Gera uma matriz par a par de BERTScore entre modelos."""
    return calcular_matriz_bertscore(df_respostas_discursivas)


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
    df_benchmark_obj = gerar_benchmark_objetivas(df_objetivas)
    df_benchmark_disc = gerar_benchmark_discursivas(df_discursivas)
    df_benchmark_final = (
        df_benchmark_obj.merge(df_benchmark_disc, on="modelo", how="outer")
        .sort_values("modelo")
        .reset_index(drop=True)
    )
    df_bertscore = gerar_matriz_bertscore_discursivas(df_respostas_discursivas)

    salvar_csv(df_benchmark_obj, Path(benchmark_objetivas_path))
    salvar_csv(df_benchmark_final, Path(benchmark_discursivas_path))

    if not df_bertscore.empty:
        matriz_serializavel = df_bertscore.copy()
        matriz_serializavel.insert(0, "modelo", matriz_serializavel.index)
        salvar_csv(matriz_serializavel.reset_index(drop=True), Path(similaridade_discursivas_path))
        figura, _ = plotar_heatmap_similaridade(
            df_bertscore.values,
            labels=df_bertscore.index.tolist(),
            output_path=heatmap_discursivas_path,
            titulo="BERTScore entre Modelos",
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
        "benchmark_objetivas": df_benchmark_obj,
        "benchmark_final": df_benchmark_final,
        "bertscore_discursivas": df_bertscore,
    }
