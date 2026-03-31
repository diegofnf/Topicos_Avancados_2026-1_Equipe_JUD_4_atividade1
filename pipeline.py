"""Funcoes de alto nivel para orquestrar a execucao do notebook."""

from __future__ import annotations

import pandas as pd

from configuracoes import ConfiguracaoExecucao


def preparar_conjunto_dados(configuracao: ConfiguracaoExecucao):
    """Executa a etapa de leitura e preparacao dos dados."""
    from carregar_dados import preparar_dados

    return preparar_dados(
        caminho_curadorias=configuracao.curadorias_csv,
        caminhos_saida=configuracao.caminhos_saida,
        limite_objetivas=configuracao.limite_objetivas,
        limite_discursivas=configuracao.limite_discursivas,
    )


def baixar_modelos_requeridos(configuracao: ConfiguracaoExecucao) -> None:
    """Baixa os modelos necessarios para a execucao."""
    from modelos import baixar_modelos

    baixar_modelos(configuracao)


def gerar_ou_carregar_respostas(
    configuracao: ConfiguracaoExecucao,
    df_objetivas: pd.DataFrame,
    df_discursivas: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gera respostas novas ou reaproveita respostas ja persistidas."""
    from geracao_respostas import gerar_respostas

    caminhos = configuracao.caminhos_saida
    if configuracao.reprocessar_respostas or not (caminhos.respostas_objetivas_csv.exists() and caminhos.respostas_discursivas_csv.exists()):
        return gerar_respostas(df_objetivas, df_discursivas, configuracao)

    print("Reaproveitando respostas ja salvas.")
    return (
        pd.read_csv(caminhos.respostas_objetivas_csv),
        pd.read_csv(caminhos.respostas_discursivas_csv),
    )


def executar_avaliacao_objetiva(configuracao: ConfiguracaoExecucao, df_respostas_objetivas: pd.DataFrame):
    """Executa a avaliacao das questoes objetivas."""
    from avaliacao_objetiva import avaliar_objetivas

    return avaliar_objetivas(df_respostas_objetivas, configuracao.caminhos_saida)


def executar_avaliacao_discursiva(configuracao: ConfiguracaoExecucao, df_respostas_discursivas: pd.DataFrame):
    """Executa a avaliacao discursiva estruturada."""
    from avaliacao_discursiva import avaliar_discursivas_estruturadas
    from modelos import BackendEmbeddingsJuridico, carregar_pipeline_nli

    backend_embeddings = BackendEmbeddingsJuridico(
        configuracao.modelo_sbert,
        prefixo="passage",
        max_length=configuracao.max_length_sbert,
    )
    pipeline_nli = carregar_pipeline_nli(configuracao.modelo_nli)
    return avaliar_discursivas_estruturadas(
        df_respostas_discursivas,
        backend_embeddings,
        pipeline_nli,
        configuracao.caminhos_saida,
    )


def executar_consolidado(
    configuracao: ConfiguracaoExecucao,
    df_objetivas_detalhe: pd.DataFrame,
    benchmark_objetivas: pd.DataFrame,
    df_discursivas_detalhe: pd.DataFrame,
    benchmark_discursivas: pd.DataFrame,
    df_composicao_discursiva: pd.DataFrame,
):
    """Gera os relatorios consolidados finais."""
    from relatorios import gerar_relatorios_consolidados

    return gerar_relatorios_consolidados(
        df_objetivas_detalhe,
        benchmark_objetivas,
        df_discursivas_detalhe,
        benchmark_discursivas,
        df_composicao_discursiva,
        configuracao.caminhos_saida,
    )
