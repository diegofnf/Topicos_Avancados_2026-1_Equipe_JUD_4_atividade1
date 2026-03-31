"""Avaliacao das questoes objetivas."""

from __future__ import annotations

import pandas as pd

from configuracoes import CaminhosSaida
from utilitarios import salvar_csv


def avaliar_objetivas(df_respostas_objetivas: pd.DataFrame, caminhos_saida: CaminhosSaida) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula o benchmark das questoes objetivas por modelo."""
    detalhe = df_respostas_objetivas.copy()
    detalhe["correto"] = detalhe["correto"].astype(bool)
    resumo = (
        detalhe.groupby("modelo", dropna=False)
        .agg(
            acuracia_objetivas=("correto", "mean"),
            acertos_objetivas=("correto", "sum"),
            total_objetivas=("correto", "size"),
        )
        .reset_index()
        .sort_values(["acuracia_objetivas", "acertos_objetivas", "modelo"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    salvar_csv(resumo, caminhos_saida.benchmark_objetivas_csv)
    return detalhe, resumo
