from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from config import (
    CURADORIA_DISCURSIVAS_CSV,
    CURADORIA_OBJETIVAS_CSV,
    CURADORIAS_EXTERNAS_CSV,
    DISC_SLICE_END,
    DISC_SLICE_START,
    OBJ_SLICE_END,
    OBJ_SLICE_START,
    QUESTOES_DISCURSIVAS_CSV,
    QUESTOES_OBJETIVAS_CSV,
    USAR_CURADORIA_EXTERNA,
)
from data_utils import (
    formatar_area_especialidade_j1,
    formatar_area_especialidade_j2,
    normalizar_values,
    salvar_csv,
    serializar_json_campo,
    somar_values,
    timestamp_execucao,
)


def _slug_disciplina(valor: str) -> str:
    texto = str(valor or "").strip()
    if not texto:
        return "Nao identificado"
    return texto


def _parse_json_field(valor, default):
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return default
    if isinstance(valor, (dict, list)):
        return valor
    texto = str(valor).strip()
    if not texto:
        return default
    try:
        return json.loads(texto)
    except Exception:
        return default


def _montar_texto_discursiva_curada(item: dict) -> str:
    enunciado = str(item.get("Questão", "")).strip()
    perguntas = _parse_json_field(item.get("Perguntas (J1)"), [])
    if not perguntas:
        return enunciado

    itens_fmt = []
    for numero, pergunta in enumerate(perguntas, start=1):
        if isinstance(pergunta, dict):
            texto = str(pergunta.get("texto", "")).strip()
        else:
            texto = str(pergunta).strip()
        if texto:
            itens_fmt.append(f"{numero}. {texto}")

    if not itens_fmt:
        return enunciado
    return f"{enunciado}\n\n" + "\n\n".join(itens_fmt)


def _extrair_values_discursiva(item: dict) -> list[float]:
    perguntas = _parse_json_field(item.get("Perguntas (J1)"), [])
    valores = []
    for pergunta in perguntas:
        if isinstance(pergunta, dict) and pergunta.get("pontuacao_pergunta") is not None:
            try:
                valores.append(float(pergunta["pontuacao_pergunta"]))
            except (TypeError, ValueError):
                continue

    if valores:
        return normalizar_values(valores)

    total = item.get("Pontuação Total (J1)")
    try:
        return normalizar_values([float(total)])
    except (TypeError, ValueError):
        return []


def _formatar_legislacao_base(item: dict) -> str:
    norma = str(item.get("Legislação Norma", "")).strip()
    artigos = _parse_json_field(item.get("Legislação Artigos"), [])
    principais = [
        str(artigo.get("artigo", "")).strip()
        for artigo in artigos
        if isinstance(artigo, dict) and str(artigo.get("artigo", "")).strip()
    ]
    principais = list(dict.fromkeys(principais))

    if norma and principais:
        return f"{norma}: {', '.join(principais[:4])}"
    if norma:
        return norma
    return "Não identificado com segurança"


def _justificativa_dificuldade(item: dict) -> str:
    criterios = _parse_json_field(item.get("Dificuldade Critérios"), [])
    criterios_limpos = [str(criterio).strip() for criterio in criterios if str(criterio).strip()]
    if criterios_limpos:
        return "; ".join(criterios_limpos[:4])
    nivel = str(item.get("Dificuldade Nível", "")).strip()
    return nivel or "Classificação manual"


def _timestamp_classificacao(item: dict) -> str:
    return str(item.get("Data de Classificação") or timestamp_execucao())


def _alternativas_json(alternativas: dict) -> str:
    ordem = ["A", "B", "C", "D"]
    normalizado = {chave: alternativas.get(chave, "") for chave in ordem if chave in alternativas}
    for chave, valor in alternativas.items():
        if chave not in normalizado:
            normalizado[chave] = valor
    return json.dumps(normalizado, ensure_ascii=False)


def _gabarito_discursiva(item: dict) -> dict:
    return _parse_json_field(item.get("Gabarito"), {})


def _alternativas_objetiva(item: dict) -> dict:
    return _parse_json_field(item.get("Alternativas (J2)"), {})


def _preparar_curadoria_externa_discursivas(df_curado: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    registros_questoes = []
    registros_curadoria = []

    for item in df_curado.to_dict("records"):
        area = _slug_disciplina(item.get("Especialidade Disciplina"))
        values = _extrair_values_discursiva(item)
        texto_questao = _montar_texto_discursiva_curada(item)
        gabarito = _gabarito_discursiva(item)

        registros_questoes.append(
            {
                "question_id": item.get("ID da Questão"),
                "dataset": "J1",
                "tipo_questao": "discursiva",
                "categoria_dataset": area,
                "area_especialidade_dataset": area,
                "system_prompt": item.get("Prompt System"),
                "values_json": serializar_json_campo(values),
                "nota_maxima_total": somar_values(values),
                "texto_questao": texto_questao,
                "gabarito_narrativo": gabarito.get("gabarito_completo"),
                "gabarito_itens_json": serializar_json_campo(gabarito.get("itens") or []),
            }
        )

        registros_curadoria.append(
            {
                "question_id": item.get("ID da Questão"),
                "dataset": "J1",
                "tipo_questao": "discursiva",
                "modelo_curador": (item.get("Curador") or "Curadoria externa"),
                "texto_questao": texto_questao,
                "nivel_dificuldade_equipe": item.get("Dificuldade Nível"),
                "justificativa_dificuldade": _justificativa_dificuldade(item),
                "area_especialidade_equipe": area,
                "legislacao_base": _formatar_legislacao_base(item),
                "confianca": "alta",
                "saida_bruta": json.dumps(item, ensure_ascii=False),
                "json_parse_ok": True,
                "timestamp_execucao": _timestamp_classificacao(item),
            }
        )

    return pd.DataFrame(registros_questoes), pd.DataFrame(registros_curadoria)


def _preparar_curadoria_externa_objetivas(df_curado: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    registros_questoes = []
    registros_curadoria = []

    for item in df_curado.to_dict("records"):
        area = _slug_disciplina(item.get("Especialidade Disciplina"))
        texto_questao = str(item.get("Questão", "")).strip()

        registros_questoes.append(
            {
                "question_id": item.get("ID da Questão"),
                "dataset": "J2",
                "tipo_questao": "objetiva",
                "numero_questao": item.get("Número Questão Exame"),
                "categoria_dataset": area,
                "area_especialidade_dataset": area,
                "texto_questao": texto_questao,
                "alternativas_json": _alternativas_json(_alternativas_objetiva(item)),
                "gabarito_oficial": item.get("Gabarito"),
            }
        )

        registros_curadoria.append(
            {
                "question_id": item.get("ID da Questão"),
                "dataset": "J2",
                "tipo_questao": "objetiva",
                "modelo_curador": (item.get("Curador") or "Curadoria externa"),
                "texto_questao": texto_questao,
                "nivel_dificuldade_equipe": item.get("Dificuldade Nível"),
                "justificativa_dificuldade": _justificativa_dificuldade(item),
                "area_especialidade_equipe": area,
                "legislacao_base": _formatar_legislacao_base(item),
                "confianca": "alta",
                "saida_bruta": json.dumps(item, ensure_ascii=False),
                "json_parse_ok": True,
                "timestamp_execucao": _timestamp_classificacao(item),
            }
        )

    return pd.DataFrame(registros_questoes), pd.DataFrame(registros_curadoria)


def preparar_dados_curadoria_externa() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_curado = pd.read_csv(Path(CURADORIAS_EXTERNAS_CSV), sep=";", encoding="utf-8-sig")
    df_curado_j1 = df_curado.loc[df_curado["Dataset"] == "J1"].reset_index(drop=True)
    df_curado_j2 = df_curado.loc[df_curado["Dataset"] == "J2"].reset_index(drop=True)

    df_disc, df_cur_disc = _preparar_curadoria_externa_discursivas(df_curado_j1)
    df_obj, df_cur_obj = _preparar_curadoria_externa_objetivas(df_curado_j2)

    salvar_csv(df_disc, QUESTOES_DISCURSIVAS_CSV)
    salvar_csv(df_obj, QUESTOES_OBJETIVAS_CSV)
    salvar_csv(df_cur_disc, CURADORIA_DISCURSIVAS_CSV)
    salvar_csv(df_cur_obj, CURADORIA_OBJETIVAS_CSV)
    return df_disc, df_obj


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


def preparar_questoes_discursivas_hf() -> pd.DataFrame:
    dataset_disc = load_dataset("maritaca-ai/oab-bench", name="questions")
    df_disc_raw = (
        pd.DataFrame(dataset_disc["train"])
        .iloc[DISC_SLICE_START:DISC_SLICE_END]
        .reset_index(drop=True)
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

    salvar_csv(df_disc, QUESTOES_DISCURSIVAS_CSV)
    return df_disc


def preparar_questoes_objetivas_hf() -> pd.DataFrame:
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


def preparar_dados_hf() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_disc = preparar_questoes_discursivas_hf()
    df_obj = preparar_questoes_objetivas_hf()
    return df_disc, df_obj


def preparar_dados() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Executa o preparo completo das questões discursivas e objetivas."""
    if USAR_CURADORIA_EXTERNA:
        return preparar_dados_curadoria_externa()
    return preparar_dados_hf()
