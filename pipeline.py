import json
import re

from config import MODELO_JUIZ, MODELO_CURADOR
from data_utils import extrair_json_bruto, timestamp_execucao
from model_utils import gerar_texto
from prompts import (
    PROMPT_CANDIDATO_DISCURSIVA,
    PROMPT_CANDIDATO_OBJETIVA,
    PROMPT_CURADORIA,
    PROMPT_JUIZ_AVALIACAO,
)


def gerar_resposta_discursiva(model, tokenizer, row: dict, nome_modelo: str) -> dict:
    prompt = PROMPT_CANDIDATO_DISCURSIVA.format(questao=row["texto_questao"])
    resposta = gerar_texto(
        model,
        tokenizer,
        prompt,
        sample=True,
        max_tokens=400,
        temperature=0.7,
    )
    return {
        "question_id": row["question_id"],
        "dataset": row["dataset"],
        "tipo_questao": row["tipo_questao"],
        "modelo": nome_modelo,
        "texto_questao": row["texto_questao"],
        "resposta": resposta,
        "timestamp_execucao": timestamp_execucao(),
    }


def gerar_resposta_objetiva(model, tokenizer, row: dict, nome_modelo: str) -> dict:
    alternativas = json.loads(row["alternativas_json"])
    prompt = PROMPT_CANDIDATO_OBJETIVA.format(
        questao=row["texto_questao"],
        A=alternativas.get("A", ""),
        B=alternativas.get("B", ""),
        C=alternativas.get("C", ""),
        D=alternativas.get("D", ""),
    )
    resposta = gerar_texto(model, tokenizer, prompt, sample=False, max_tokens=10)
    match = re.search(r"\b[A-D]\b", resposta.upper())
    letra = match.group() if match else "N/A"
    return {
        "question_id": row["question_id"],
        "dataset": row["dataset"],
        "tipo_questao": row["tipo_questao"],
        "modelo": nome_modelo,
        "texto_questao": row["texto_questao"],
        "resposta": letra,
        "gabarito_oficial": row["gabarito_oficial"],
        "correto": letra == row["gabarito_oficial"],
        "timestamp_execucao": timestamp_execucao(),
    }


def gerar_curadoria(model, tokenizer, row: dict, tipo_questao: str) -> dict:
    prompt = PROMPT_CURADORIA.format(
        tipo_questao=tipo_questao,
        questao=row["texto_questao"],
    )
    saida = gerar_texto(model, tokenizer, prompt, sample=False, max_tokens=300)

    json_bruto = extrair_json_bruto(saida)
    try:
        curadoria = json.loads(json_bruto) if json_bruto else {}
    except Exception:
        curadoria = {}

    return {
        "question_id": row["question_id"],
        "dataset": row["dataset"],
        "tipo_questao": row["tipo_questao"],
        "modelo_curador": MODELO_CURADOR,
        "texto_questao": row["texto_questao"],
        "nivel_dificuldade_equipe": curadoria.get("nivel_dificuldade"),
        "justificativa_dificuldade": curadoria.get("justificativa_dificuldade"),
        "area_especialidade_equipe": curadoria.get("area_especialidade"),
        "justificativa_area": curadoria.get("justificativa_area"),
        "dominio_juridico": curadoria.get("dominio_juridico"),
        "legislacao_base": curadoria.get("legislacao_base"),
        "confianca": curadoria.get("confianca"),
        "timestamp_execucao": timestamp_execucao(),
        "saida_bruta": json_bruto if json_bruto else saida,
        "json_parse_ok": bool(curadoria),
    }


def gerar_avaliacao_discursiva(
    model, tokenizer, questao_row: dict, resposta_row: dict, curadoria_row: dict
) -> dict:
    prompt = PROMPT_JUIZ_AVALIACAO.format(
        questao=questao_row["texto_questao"],
        resposta=resposta_row["resposta"],
        nivel_dificuldade=curadoria_row.get("nivel_dificuldade_equipe", "Não informado"),
        dominio_juridico=curadoria_row.get("dominio_juridico", "Não informado"),
        area_especialidade=curadoria_row.get("area_especialidade_equipe", "Não informado"),
        legislacao_base=curadoria_row.get("legislacao_base", "Não informado"),
    )
    saida = gerar_texto(model, tokenizer, prompt, sample=False, max_tokens=300)

    json_bruto = extrair_json_bruto(saida)
    try:
        avaliacao = json.loads(json_bruto) if json_bruto else {}
    except Exception:
        avaliacao = {}

    try:
        nota = float(str(avaliacao.get("nota", "")).replace(",", "."))
        if not (0 <= nota <= 10):
            nota = None
    except (ValueError, TypeError):
        nota = None

    return {
        "question_id": resposta_row["question_id"],
        "dataset": resposta_row["dataset"],
        "tipo_questao": resposta_row["tipo_questao"],
        "modelo_candidato": resposta_row["modelo"],
        "modelo_juiz": MODELO_JUIZ,
        "texto_questao": questao_row["texto_questao"],
        "resposta": resposta_row["resposta"],
        "avaliacao": avaliacao.get("avaliacao"),
        "nota": nota,
        "json_parse_ok": bool(avaliacao),
        "saida_bruta": json_bruto if json_bruto else saida,
        "timestamp_execucao": timestamp_execucao(),
    }
