import json

from config import MODELO_JUIZ, MODELO_CURADOR
from data_utils import extrair_json_bruto, timestamp_execucao
from model_utils import gerar_texto
from prompts import (
    PROMPT_CANDIDATO_DISCURSIVA,
    PROMPT_CANDIDATO_OBJETIVA,
    PROMPT_CURADORIA,
    PROMPT_JUIZ_AVALIACAO,
)


def _parse_float(valor):
    try:
        return float(str(valor).replace(",", "."))
    except (TypeError, ValueError):
        return None


def _carregar_values(values_json) -> list[float]:
    try:
        values = json.loads(values_json) if values_json else []
    except Exception:
        values = []

    notas = []
    for value in values:
        nota = _parse_float(value)
        if nota is not None:
            notas.append(nota)
    return notas


def _formatar_pontuacao_questao(values: list[float]) -> str:
    if len(values) >= 2:
        return (
            f"- nota_fundamentacao_coerencia: 0 até {values[0]:.2f}\n"
            f"- nota_aderencia_completude: 0 até {values[1]:.2f}\n"
            f"- nota_total: 0 até {sum(values):.2f}"
        )
    if len(values) == 1:
        return f"- nota_total: 0 até {values[0]:.2f}"
    return "- nota_total: não informada"


def _normalizar_avaliacao(avaliacao: dict, values: list[float]) -> tuple[float | None, float | None, float | None, float | None]:
    if len(values) >= 2:
        nota_fc = _parse_float(avaliacao.get("nota_fundamentacao_coerencia"))
        nota_ac = _parse_float(avaliacao.get("nota_aderencia_completude"))

        if nota_fc is not None:
            nota_fc = max(0.0, min(nota_fc, values[0]))
        if nota_ac is not None:
            nota_ac = max(0.0, min(nota_ac, values[1]))

        if nota_fc is None and nota_ac is None:
            nota_total = None
        else:
            nota_total = round((nota_fc or 0.0) + (nota_ac or 0.0), 4)

        return nota_fc, nota_ac, nota_total, sum(values[:2])

    if len(values) == 1:
        nota_total = _parse_float(avaliacao.get("nota_total"))
        if nota_total is not None:
            nota_total = max(0.0, min(nota_total, values[0]))
        return None, None, nota_total, values[0]

    nota_total = _parse_float(avaliacao.get("nota_total"))
    return None, None, nota_total, None


def gerar_resposta_discursiva(model, tokenizer, row: dict, nome_modelo: str) -> dict:
    prompt = PROMPT_CANDIDATO_DISCURSIVA.format(questao=row["texto_questao"])
    # Código anterior:
    # resposta = gerar_texto(
    #     model,
    #     tokenizer,
    #     prompt,
    #     sample=True,
    #     max_tokens=400,
    #     temperature=0.7,
    # )
    # Artifício do prefill: queremos forçar que o modelo comece diretamente pelo JSON.
    prefill = "{"
    prompt = prompt + f"\n{prefill}"
    saida = prefill + gerar_texto(
        model,
        tokenizer,
        prompt,
        sample=True,
        max_tokens=400,
        temperature=0.7,
        system_prompt=row.get("system_prompt"),
    )
    json_bruto = extrair_json_bruto(saida)
    try:
        resposta_json = json.loads(json_bruto) if json_bruto else {}
    except Exception:
        resposta_json = {}

    resposta = str(resposta_json.get("resposta_discursiva", "")).strip()

    return {
        "question_id": row["question_id"],
        "dataset": row["dataset"],
        "tipo_questao": row["tipo_questao"],
        "modelo": nome_modelo,
        "area_especialidade_dataset": row.get("area_especialidade_dataset"),
        "system_prompt": row.get("system_prompt"),
        "values_json": row.get("values_json"),
        "nota_maxima_total": row.get("nota_maxima_total"),
        "texto_questao": row["texto_questao"],
        "resposta": resposta,
        "json_parse_ok": bool(resposta_json),
        "saida_bruta": json_bruto if json_bruto else saida,
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
    # Código anterior:
    # resposta = gerar_texto(model, tokenizer, prompt, sample=False, max_tokens=40)
    # Artifício do prefill: queremos forçar que o modelo comece diretamente pelo JSON.
    prefill = "{"
    prompt = prompt + f"\n{prefill}"
    saida = prefill + gerar_texto(model, tokenizer, prompt, sample=False, max_tokens=40)
    json_bruto = extrair_json_bruto(saida)
    try:
        resposta_json = json.loads(json_bruto) if json_bruto else {}
    except Exception:
        resposta_json = {}

    letra = str(resposta_json.get("resposta_objetiva", "")).strip().upper()
    if letra not in {"A", "B", "C", "D"}:
        letra = "N/A"

    return {
        "question_id": row["question_id"],
        "dataset": row["dataset"],
        "tipo_questao": row["tipo_questao"],
        "modelo": nome_modelo,
        "area_especialidade_dataset": row.get("area_especialidade_dataset"),
        "texto_questao": row["texto_questao"],
        "resposta": letra,
        "gabarito_oficial": row["gabarito_oficial"],
        "correto": letra == row["gabarito_oficial"],
        "json_parse_ok": bool(resposta_json),
        "saida_bruta": json_bruto if json_bruto else saida,
        "timestamp_execucao": timestamp_execucao(),
    }


def gerar_curadoria(model, tokenizer, row: dict, tipo_questao: str) -> dict:
    prompt = PROMPT_CURADORIA.format(
        tipo_questao=tipo_questao,
        questao=row["texto_questao"],
    )
    # Código anterior:
    # saida = gerar_texto(model, tokenizer, prompt, sample=False, max_tokens=300)
    # Artifício do prefill: queremos forçar que o modelo comece diretamente pelo JSON.
    prefill = "{"
    prompt = prompt + f"\n{prefill}"
    saida = prefill + gerar_texto(model, tokenizer, prompt, sample=False, max_tokens=180)

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
        "area_especialidade_equipe": row.get("area_especialidade_dataset"),
        "legislacao_base": curadoria.get("legislacao_base"),
        "confianca": curadoria.get("confianca"),        
        "saida_bruta": json_bruto if json_bruto else saida,
        "json_parse_ok": bool(curadoria),
        "timestamp_execucao": timestamp_execucao(),
    }


def gerar_avaliacao_discursiva(
    model, tokenizer, questao_row: dict, resposta_row: dict, curadoria_row: dict
) -> dict:
    values = _carregar_values(questao_row.get("values_json"))
    prompt = PROMPT_JUIZ_AVALIACAO.format(
        questao=questao_row["texto_questao"],
        resposta=resposta_row["resposta"],
        nivel_dificuldade=curadoria_row.get("nivel_dificuldade_equipe", "Não informado"),
        area_especialidade=questao_row.get("area_especialidade_dataset", "Não informado"),
        legislacao_base=curadoria_row.get("legislacao_base", "Não informado"),
        pontuacao_questao=_formatar_pontuacao_questao(values),
    )
    # Código anterior:
    # saida = gerar_texto(model, tokenizer, prompt, sample=False, max_tokens=450)
    # Artifício do prefill: queremos forçar que o modelo comece diretamente pelo JSON.
    prefill = "{"
    prompt = prompt + f"\n{prefill}"
    saida = prefill + gerar_texto(model, tokenizer, prompt, sample=False, max_tokens=400)

    json_bruto = extrair_json_bruto(saida)
    try:
        avaliacao = json.loads(json_bruto) if json_bruto else {}
    except Exception:
        avaliacao = {}

    nota_fc, nota_ac, nota_total, nota_maxima_total = _normalizar_avaliacao(avaliacao, values)

    return {
        "question_id": resposta_row["question_id"],
        "dataset": resposta_row["dataset"],
        "tipo_questao": resposta_row["tipo_questao"],
        "modelo_candidato": resposta_row["modelo"],
        "modelo_juiz": MODELO_JUIZ,
        "area_especialidade_dataset": questao_row.get("area_especialidade_dataset"),
        "texto_questao": questao_row["texto_questao"],
        "resposta": resposta_row["resposta"],
        "argumentacao": avaliacao.get("argumentacao"),
        "precisao": avaliacao.get("precisao"),
        "coesao_legal": avaliacao.get("coesao_legal"),
        "nota_fundamentacao_coerencia": nota_fc,
        "nota_aderencia_completude": nota_ac,
        "nota_maxima_total": nota_maxima_total,
        "avaliacao": avaliacao.get("avaliacao"),
        "nota": nota_total,
        "json_parse_ok": bool(avaliacao),
        "saida_bruta": json_bruto if json_bruto else saida,
        "timestamp_execucao": timestamp_execucao(),
    }
