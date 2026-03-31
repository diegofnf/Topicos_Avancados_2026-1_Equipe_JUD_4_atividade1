"""Leitura e preparo dos dados de curadoria."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .configuracoes import CaminhosSaida
from .utilitarios import carregar_json_seguro, converter_para_float, normalizar_texto, salvar_csv


def montar_texto_discursivo(linha: pd.Series) -> str:
    """Monta o texto completo de uma questao discursiva com seus subitens."""
    enunciado = normalizar_texto(linha["Questão"])
    perguntas = carregar_json_seguro(linha.get("Perguntas (J1)", "[]"), [])
    if not isinstance(perguntas, list) or not perguntas:
        return enunciado

    blocos: list[str] = []
    for indice, item in enumerate(perguntas, start=1):
        texto = normalizar_texto(item.get("texto") if isinstance(item, dict) else item)
        if texto:
            blocos.append(f"{indice}. {texto}")

    if not blocos:
        return enunciado
    return enunciado + "\n\nPerguntas:\n" + "\n\n".join(blocos)


def _normalizar_alternativas(valor: str) -> dict[str, str]:
    """Padroniza alternativas de questoes objetivas."""
    alternativas = carregar_json_seguro(valor, {})
    if isinstance(alternativas, dict):
        return {str(chave).upper(): normalizar_texto(texto) for chave, texto in alternativas.items()}
    return {}


def _normalizar_gabarito_discursivo(valor: str) -> dict:
    """Padroniza o JSON do gabarito discursivo."""
    dados = carregar_json_seguro(valor, {})
    return dados if isinstance(dados, dict) else {}


def preparar_dados(
    caminho_curadorias: Path,
    caminhos_saida: CaminhosSaida,
    limite_objetivas: int | None = None,
    limite_discursivas: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carrega o CSV principal e gera subconjuntos prontos para execucao."""
    df_curadoria = pd.read_csv(caminho_curadorias, sep=";", dtype=str, keep_default_na=False)
    df_curadoria["Pontuação Total (J1)"] = df_curadoria["Pontuação Total (J1)"].apply(
        lambda valor: converter_para_float(valor, 0.0)
    )

    df_objetivas = df_curadoria[df_curadoria["Tipo Questão"].str.upper() == "OBJETIVA"].copy()
    df_discursivas = df_curadoria[df_curadoria["Tipo Questão"].str.upper() != "OBJETIVA"].copy()

    df_objetivas["alternativas"] = df_objetivas["Alternativas (J2)"].apply(_normalizar_alternativas)
    df_objetivas["gabarito_oficial"] = df_objetivas["Gabarito"].astype(str).str.strip().str.upper()
    df_objetivas["texto_questao"] = df_objetivas["Questão"].astype(str).str.strip()
    df_objetivas["peso_questao"] = 1.0
    df_objetivas["nivel_dificuldade"] = df_objetivas["Dificuldade Nível"].replace("", "nao_informado")
    df_objetivas["disciplina"] = df_objetivas["Especialidade Disciplina"].replace("", "nao_informado")
    df_objetivas["tema"] = df_objetivas["Especialidade Tema"].replace("", "nao_informado")

    df_discursivas["texto_questao"] = df_discursivas.apply(montar_texto_discursivo, axis=1)
    df_discursivas["gabarito_estruturado"] = df_discursivas["Gabarito"].apply(_normalizar_gabarito_discursivo)
    df_discursivas["gabarito_completo"] = df_discursivas["gabarito_estruturado"].apply(
        lambda valor: normalizar_texto(valor.get("gabarito_completo", "")) if isinstance(valor, dict) else ""
    )
    df_discursivas["criterios_correcao"] = df_discursivas["gabarito_estruturado"].apply(
        lambda valor: valor.get("criterios", []) if isinstance(valor, dict) else []
    )
    df_discursivas["pontuacao_total"] = df_discursivas["Pontuação Total (J1)"].apply(
        lambda valor: converter_para_float(valor, 0.0)
    )
    df_discursivas["nivel_dificuldade"] = df_discursivas["Dificuldade Nível"].replace("", "nao_informado")
    df_discursivas["disciplina"] = df_discursivas["Especialidade Disciplina"].replace("", "nao_informado")
    df_discursivas["tema"] = df_discursivas["Especialidade Tema"].replace("", "nao_informado")

    if limite_objetivas:
        df_objetivas = df_objetivas.head(limite_objetivas).copy()
    if limite_discursivas:
        df_discursivas = df_discursivas.head(limite_discursivas).copy()

    salvar_csv(
        df_objetivas[
            [
                "ID da Questão",
                "Tipo Questão",
                "Dataset",
                "Prompt System",
                "texto_questao",
                "alternativas",
                "gabarito_oficial",
                "nivel_dificuldade",
                "disciplina",
                "tema",
                "peso_questao",
                "Número Questão Sequencial",
                "Número Questão Exame",
                "ID Exame",
                "Ano Exame",
            ]
        ].assign(alternativas=lambda df: df["alternativas"].apply(json.dumps)),
        caminhos_saida.questoes_objetivas_csv,
    )

    salvar_csv(
        df_discursivas[
            [
                "ID da Questão",
                "Tipo Questão",
                "Dataset",
                "Prompt System",
                "texto_questao",
                "gabarito_completo",
                "criterios_correcao",
                "pontuacao_total",
                "nivel_dificuldade",
                "disciplina",
                "tema",
                "Número Questão Sequencial",
                "Número Questão Exame",
                "ID Exame",
                "Ano Exame",
            ]
        ].assign(criterios_correcao=lambda df: df["criterios_correcao"].apply(json.dumps)),
        caminhos_saida.questoes_discursivas_csv,
    )

    return (
        df_curadoria.reset_index(drop=True),
        df_objetivas.reset_index(drop=True),
        df_discursivas.reset_index(drop=True),
    )
