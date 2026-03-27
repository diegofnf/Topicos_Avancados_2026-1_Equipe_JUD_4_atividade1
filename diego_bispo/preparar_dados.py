from __future__ import annotations

import json
import re

import pandas as pd
from datasets import load_dataset

from config import (
    DISC_SLICE_END,
    DISC_SLICE_START,
    OBJ_SLICE_END,
    OBJ_SLICE_START,
    QUESTOES_DISCURSIVAS_CSV,
    QUESTOES_OBJETIVAS_CSV,
)
from data_utils import (
    formatar_area_especialidade_j1,
    formatar_area_especialidade_j2,
    normalizar_values,
    salvar_csv,
    serializar_json_campo,
    somar_values,
)


def _extrair_texto_guideline(registro: pd.Series | dict | list | str) -> str:
    """
    Extrai o texto bruto do guideline suportando tanto o schema antigo com
    `turns` na raiz quanto o schema atual com `choices[0]['turns']`.
    """
    if isinstance(registro, str):
        return registro.strip()

    if isinstance(registro, list):
        return str(registro[0]).strip() if registro else ""

    if isinstance(registro, (pd.Series, dict)):
        turns = registro.get("turns")
        if isinstance(turns, list) and turns:
            return str(turns[0]).strip()

        choices = registro.get("choices")
        if isinstance(choices, list) and choices:
            primeiro = choices[0]
            if isinstance(primeiro, dict):
                turns_choice = primeiro.get("turns")
                if isinstance(turns_choice, list) and turns_choice:
                    return str(turns_choice[0]).strip()
            return str(primeiro).strip()

    return ""


def _normalizar_tabela_guideline(tabela_raw: str) -> str:
    """
    Normaliza tabelas markdown que às vezes chegam inline em uma única linha.
    """
    texto = str(tabela_raw or "").strip()
    if not texto:
        return ""

    # Alguns guidelines vêm com as linhas da tabela coladas, usando `| |`
    # entre uma linha e outra. Isso recompõe as quebras de linha esperadas.
    texto = re.sub(r"\|\s+\|", "|\n|", texto)
    texto = re.sub(r"\n{2,}", "\n", texto)
    return texto.strip()


def _extrair_partes_guideline(registro: pd.Series | dict | list | str) -> tuple[str, str]:
    """
    Recebe um registro de guidelines e devolve
    (narrativo, tabela_raw) separados pelo literal 'Distribuição dos Pontos'.
    """
    texto = _extrair_texto_guideline(registro)
    partes = re.split(r"distribuiç[aã]o dos pontos", texto, maxsplit=1, flags=re.IGNORECASE)
    narrativo = partes[0].strip()
    tabela_raw = _normalizar_tabela_guideline(partes[1]) if len(partes) > 1 else ""
    return narrativo, tabela_raw


def parse_itens_gabarito(tabela_raw: str) -> list[dict]:
    """
    Converte a tabela markdown de pontuação em uma lista de itens estruturados.

    Cada item tem:
    - texto
    - peso_maximo
    - secao
    """
    tabela_raw = _normalizar_tabela_guideline(tabela_raw)
    linhas = [linha.strip() for linha in tabela_raw.split("\n") if linha.strip().startswith("|")]
    linhas = [linha for linha in linhas if not re.match(r"^\|[-:\s|]+\|$", linha)]

    itens: list[dict] = []
    secao_atual: str | None = None

    for linha in linhas:
        colunas = [coluna.strip() for coluna in linha.strip("|").split("|")]
        if len(colunas) < 2:
            continue

        texto = colunas[0].strip()
        pontuacao_raw = colunas[1].strip()

        if texto == "ITEM":
            continue

        valores = re.findall(r"\d+,\d+", pontuacao_raw)
        peso = max(float(valor.replace(",", ".")) for valor in valores) if valores else None

        eh_numerado = bool(re.match(r"^\d+[\.\d]*\.?\s", texto))
        eh_secao = (peso is None) and (not eh_numerado)

        if eh_secao:
            secao_atual = texto
            continue

        itens.append(
            {
                "texto": texto,
                "peso_maximo": peso,
                "secao": secao_atual,
            }
        )

    return itens


def _montar_narrativo(registro: pd.Series | dict | list | str) -> str:
    return _extrair_partes_guideline(registro)[0]


def _montar_itens_json(registro: pd.Series | dict | list | str) -> str:
    _, tabela_raw = _extrair_partes_guideline(registro)
    itens = parse_itens_gabarito(tabela_raw)
    return json.dumps(itens, ensure_ascii=False)


def montar_texto_questao(row: pd.Series) -> str:
    """Concatena enunciado e subitens da questão discursiva."""
    enunciado = str(row["statement"]).strip()
    itens = row.get("turns", [])
    if isinstance(itens, list):
        itens = [str(item).strip() for item in itens if str(item).strip()]
    else:
        itens = []

    if not itens:
        return enunciado

    itens_fmt = "\n\n".join(f"{numero}. {item}" for numero, item in enumerate(itens, 1))
    return f"{enunciado}\n\n{itens_fmt}"


def normalizar_choices(choices: dict) -> str:
    """Serializa alternativas das objetivas em JSON plano label -> texto."""
    return json.dumps(
        dict(zip(choices.get("label", []), choices.get("text", []))),
        ensure_ascii=False,
    )


def preparar_questoes_discursivas() -> pd.DataFrame:
    """Carrega, prepara e salva o CSV de questões discursivas."""
    dataset_disc = load_dataset("maritaca-ai/oab-bench", name="questions")
    df_disc_raw = (
        pd.DataFrame(dataset_disc["train"])
        .iloc[DISC_SLICE_START:DISC_SLICE_END]
        .reset_index(drop=True)
    )

    dataset_guide = load_dataset("maritaca-ai/oab-bench", name="guidelines")
    df_guide_raw = pd.DataFrame(dataset_guide["train"])

    df_guide_raw = df_guide_raw.assign(
        gabarito_narrativo=df_guide_raw.apply(_montar_narrativo, axis=1),
        gabarito_itens_json=df_guide_raw.apply(_montar_itens_json, axis=1),
    )

    df_disc = pd.DataFrame(
        {
            "question_id": df_disc_raw["question_id"],
            "dataset": "J1",
            "tipo_questao": "discursiva",
            "categoria_dataset": df_disc_raw["category"],
            "area_especialidade_dataset": df_disc_raw["category"].apply(formatar_area_especialidade_j1),
            "system_prompt": df_disc_raw["system"],
            "values_json": df_disc_raw["values"].apply(
                lambda values: serializar_json_campo(normalizar_values(values))
            ),
            "nota_maxima_total": df_disc_raw["values"].apply(somar_values),
            "texto_questao": df_disc_raw.apply(montar_texto_questao, axis=1),
        }
    )

    df_disc = df_disc.merge(
        df_guide_raw[["question_id", "gabarito_narrativo", "gabarito_itens_json"]],
        on="question_id",
        how="left",
    )

    salvar_csv(df_disc, QUESTOES_DISCURSIVAS_CSV)
    return df_disc


def preparar_questoes_objetivas() -> pd.DataFrame:
    """Carrega, prepara e salva o CSV de questões objetivas."""
    dataset_obj = load_dataset("eduagarcia/oab_exams")
    df_obj_raw = (
        pd.DataFrame(dataset_obj["train"])
        .iloc[OBJ_SLICE_START:OBJ_SLICE_END]
        .reset_index(drop=True)
    )

    df_obj = pd.DataFrame(
        {
            "question_id": df_obj_raw["id"],
            "dataset": "J2",
            "tipo_questao": "objetiva",
            "numero_questao": df_obj_raw["question_number"],
            "categoria_dataset": df_obj_raw["question_type"],
            "area_especialidade_dataset": df_obj_raw["question_type"].apply(formatar_area_especialidade_j2),
            "texto_questao": df_obj_raw["question"],
            "alternativas_json": df_obj_raw["choices"].apply(normalizar_choices),
            "gabarito_oficial": df_obj_raw["answerKey"],
        }
    )

    salvar_csv(df_obj, QUESTOES_OBJETIVAS_CSV)
    return df_obj


def preparar_dados() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Executa o preparo completo das questões discursivas e objetivas."""
    df_disc = preparar_questoes_discursivas()
    df_obj = preparar_questoes_objetivas()
    return df_disc, df_obj
