from __future__ import annotations

import json
import re
import unicodedata
import warnings
from dataclasses import dataclass, field

import pandas as pd
from bert_score import score as bert_score_fn
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from tqdm.auto import tqdm

from config import (
    MODELO_BERTSCORE,
    MODELO_R_SBERT,
    PESO_R_BERT_F1,
    PESO_R_BLEU,
    PESO_R_KEYWORD,
    PESO_R_ROUGE_F1,
    PESO_R_SBERT,
)


def _log_status(message: str, verbose: bool = True) -> None:
    if verbose:
        print(message)


def normalize_text(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_text_ascii(text: object) -> str:
    texto = normalize_text(text).lower()
    texto = unicodedata.normalize("NFKD", texto)
    return "".join(ch for ch in texto if not unicodedata.combining(ch))


@dataclass
class MetricasQuestao:
    question_id: str
    modelo: str
    bleu: float = 0.0
    rouge1_precision: float = 0.0
    rouge1_recall: float = 0.0
    rouge1_f1: float = 0.0
    rouge2_precision: float = 0.0
    rouge2_recall: float = 0.0
    rouge2_f1: float = 0.0
    rougeL_precision: float = 0.0
    rougeL_recall: float = 0.0
    rougeL_f1: float = 0.0
    bert_precision: float = 0.0
    bert_recall: float = 0.0
    bert_f1: float = 0.0
    sbert_similaridade: float = 0.0
    keyword_score: float = 0.0
    keywords_encontrados: list[str] = field(default_factory=list)
    keywords_ausentes: list[str] = field(default_factory=list)
    score_composto: float = 0.0


def _parse_gabarito_itens(gabarito_itens_json: str | list | None) -> list[dict[str, object]]:
    if isinstance(gabarito_itens_json, str):
        try:
            dados = json.loads(gabarito_itens_json)
        except Exception:
            return []
    elif isinstance(gabarito_itens_json, list):
        dados = gabarito_itens_json
    else:
        return []

    return [item for item in dados if isinstance(item, dict)]


def _tokenize_simple(text: str) -> list[str]:
    return [token for token in re.split(r"\W+", normalize_text_ascii(text)) if len(token) >= 3]


def _extrair_keywords_legais(texto: str) -> list[str]:
    texto_ascii = normalize_text_ascii(texto)
    padroes = [
        r"art\.?\s*\d+[a-z0-9º°/-]*(?:,\s*(?:inciso|paragrafo|§)\s*[a-z0-9ivxlcdmº°-]+)?",
        r"lei\s*(?:n[ºo.]?\s*)?\d[\d./-]*",
        r"codigo\s+[a-z]+(?:\s+[a-z]+)?",
        r"constituicao\s+(?:federal|da republica)",
        r"cf/?88",
        r"sumula(?:\s+vinculante)?\s*(?:n[ºo.]?\s*)?\d+",
        r"mandado\s+de\s+seguranca",
        r"acao\s+[a-z]+(?:\s+[a-z]+){0,2}",
        r"tribunal\s+de\s+contas",
        r"ministerio\s+publico",
        r"ato\s+complexo",
        r"contraditorio",
        r"ampla\s+defesa",
        r"usucapiao",
        r"comodato",
        r"licitacao",
        r"pregao",
    ]
    encontrados: list[str] = []
    for padrao in padroes:
        encontrados.extend(re.findall(padrao, texto_ascii, flags=re.IGNORECASE))
    return list(dict.fromkeys(item.strip() for item in encontrados if item.strip()))


def _extrair_keywords_gabarito(
    gabarito_narrativo: str,
    gabarito_itens_json: str | list | None = None,
) -> list[str]:
    candidatos: list[str] = []
    candidatos.extend(_extrair_keywords_legais(gabarito_narrativo))

    for item in _parse_gabarito_itens(gabarito_itens_json):
        texto_item = normalize_text(item.get("texto", ""))
        if texto_item:
            candidatos.extend(_extrair_keywords_legais(texto_item))

            tokens_relevantes = [
                token
                for token in _tokenize_simple(texto_item)
                if token not in {"de", "da", "do", "das", "dos", "que", "para", "com"}
            ]
            if tokens_relevantes:
                candidatos.append(" ".join(tokens_relevantes[:4]))

    candidatos = [normalize_text_ascii(item) for item in candidatos if normalize_text(item)]
    candidatos = [item for item in candidatos if len(item) >= 4]
    return list(dict.fromkeys(candidatos))[:12]


def calcular_bleu(referencia: str, hipotese: str) -> float:
    ref_tok = referencia.lower().split()
    hyp_tok = hipotese.lower().split()
    if not ref_tok or not hyp_tok:
        return 0.0

    smoothing = SmoothingFunction().method4
    return float(
        sentence_bleu(
            [ref_tok],
            hyp_tok,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothing,
        )
    )


def calcular_rouge(referencia: str, hipotese: str) -> dict[str, object]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    return scorer.score(referencia, hipotese)


def calcular_bert_score(referencia: str, hipotese: str) -> tuple[float, float, float]:
    precision, recall, f1 = bert_score_fn(
        [hipotese],
        [referencia],
        lang="pt",
        model_type=MODELO_BERTSCORE,
        verbose=False,
    )
    return float(precision[0]), float(recall[0]), float(f1[0])


def calcular_sbert(referencia: str, hipotese: str, modelo: SentenceTransformer) -> float:
    emb_ref = modelo.encode(referencia, convert_to_tensor=True)
    emb_hyp = modelo.encode(hipotese, convert_to_tensor=True)
    return float(util.cos_sim(emb_ref, emb_hyp)[0][0])


def calcular_keyword_juridico(
    gabarito_narrativo: str,
    hipotese: str,
    gabarito_itens_json: str | list | None = None,
) -> tuple[float, list[str], list[str]]:
    keywords = _extrair_keywords_gabarito(gabarito_narrativo, gabarito_itens_json)
    if not keywords:
        return 0.0, [], []

    hipotese_ascii = normalize_text_ascii(hipotese)
    encontrados: list[str] = []
    ausentes: list[str] = []

    for keyword in keywords:
        if keyword in hipotese_ascii:
            encontrados.append(keyword)
        else:
            ausentes.append(keyword)

    score = len(encontrados) / len(keywords) if keywords else 0.0
    return float(score), encontrados, ausentes


def score_composto(metricas: MetricasQuestao) -> float:
    rouge_medio = (
        metricas.rouge1_f1 + metricas.rouge2_f1 + metricas.rougeL_f1
    ) / 3.0

    return float(
        (metricas.bleu * PESO_R_BLEU)
        + (rouge_medio * PESO_R_ROUGE_F1)
        + (metricas.bert_f1 * PESO_R_BERT_F1)
        + (metricas.sbert_similaridade * PESO_R_SBERT)
        + (metricas.keyword_score * PESO_R_KEYWORD)
    )


def _avaliar_resposta(
    question_id: str,
    modelo: str,
    resposta: str,
    gabarito_narrativo: str,
    gabarito_itens_json: str | list | None,
    sbert_model: SentenceTransformer,
) -> MetricasQuestao:
    resposta = normalize_text(resposta)
    gabarito_narrativo = normalize_text(gabarito_narrativo)

    metricas = MetricasQuestao(question_id=question_id, modelo=modelo)
    if not resposta or not gabarito_narrativo:
        return metricas

    metricas.bleu = round(calcular_bleu(gabarito_narrativo, resposta), 4)

    rouge = calcular_rouge(gabarito_narrativo, resposta)
    metricas.rouge1_precision = round(rouge["rouge1"].precision, 4)
    metricas.rouge1_recall = round(rouge["rouge1"].recall, 4)
    metricas.rouge1_f1 = round(rouge["rouge1"].fmeasure, 4)
    metricas.rouge2_precision = round(rouge["rouge2"].precision, 4)
    metricas.rouge2_recall = round(rouge["rouge2"].recall, 4)
    metricas.rouge2_f1 = round(rouge["rouge2"].fmeasure, 4)
    metricas.rougeL_precision = round(rouge["rougeL"].precision, 4)
    metricas.rougeL_recall = round(rouge["rougeL"].recall, 4)
    metricas.rougeL_f1 = round(rouge["rougeL"].fmeasure, 4)

    bert_p, bert_r, bert_f1 = calcular_bert_score(gabarito_narrativo, resposta)
    metricas.bert_precision = round(bert_p, 4)
    metricas.bert_recall = round(bert_r, 4)
    metricas.bert_f1 = round(bert_f1, 4)

    metricas.sbert_similaridade = round(
        calcular_sbert(gabarito_narrativo, resposta, sbert_model),
        4,
    )

    keyword_score, encontrados, ausentes = calcular_keyword_juridico(
        gabarito_narrativo,
        resposta,
        gabarito_itens_json=gabarito_itens_json,
    )
    metricas.keyword_score = round(keyword_score, 4)
    metricas.keywords_encontrados = encontrados
    metricas.keywords_ausentes = ausentes
    metricas.score_composto = round(score_composto(metricas), 4)
    return metricas


def evaluate_dataframe(
    df_respostas: pd.DataFrame,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    colunas_obrigatorias = {
        "question_id",
        "modelo",
        "resposta",
        "gabarito_narrativo",
    }
    ausentes = sorted(colunas_obrigatorias - set(df_respostas.columns))
    if ausentes:
        raise ValueError(f"Colunas obrigatorias ausentes: {', '.join(ausentes)}")

    _log_status(f"[Reference] Carregando SBERT: {MODELO_R_SBERT}", show_progress)
    sbert_model = SentenceTransformer(MODELO_R_SBERT)

    linhas: list[dict[str, object]] = []
    registros = df_respostas.to_dict("records")
    iterador = tqdm(
        registros,
        total=len(registros),
        desc="Reference por resposta",
        disable=not show_progress,
    )

    for row in iterador:
        gabarito = row.get("gabarito_narrativo")
        if pd.isna(gabarito) or not normalize_text(gabarito):
            warnings.warn(
                f"[Reference] Questao '{row.get('question_id')}' sem gabarito_narrativo. Linha ignorada.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        metricas = _avaliar_resposta(
            question_id=str(row.get("question_id", "")),
            modelo=str(row.get("modelo", "")),
            resposta=str(row.get("resposta", "")),
            gabarito_narrativo=str(gabarito),
            gabarito_itens_json=row.get("gabarito_itens_json"),
            sbert_model=sbert_model,
        )

        linhas.append(
            {
                "question_id": metricas.question_id,
                "modelo": metricas.modelo,
                "bleu": metricas.bleu,
                "rouge1_precision": metricas.rouge1_precision,
                "rouge1_recall": metricas.rouge1_recall,
                "rouge1_f1": metricas.rouge1_f1,
                "rouge2_precision": metricas.rouge2_precision,
                "rouge2_recall": metricas.rouge2_recall,
                "rouge2_f1": metricas.rouge2_f1,
                "rougeL_precision": metricas.rougeL_precision,
                "rougeL_recall": metricas.rougeL_recall,
                "rougeL_f1": metricas.rougeL_f1,
                "bert_precision": metricas.bert_precision,
                "bert_recall": metricas.bert_recall,
                "bert_f1": metricas.bert_f1,
                "sbert_similaridade": metricas.sbert_similaridade,
                "keyword_score": metricas.keyword_score,
                "keywords_encontrados": json.dumps(metricas.keywords_encontrados, ensure_ascii=False),
                "keywords_ausentes": json.dumps(metricas.keywords_ausentes, ensure_ascii=False),
                "final": metricas.score_composto,
            }
        )

    df_detalhe = pd.DataFrame(linhas)
    if df_detalhe.empty:
        return df_detalhe, pd.DataFrame(
            columns=[
                "modelo",
                "bleu",
                "rouge1_f1",
                "rouge2_f1",
                "rougeL_f1",
                "bert_f1",
                "sbert_similaridade",
                "keyword_score",
                "final",
            ]
        )

    df_detalhe["ranking"] = (
        df_detalhe.groupby("question_id")["final"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    df_resumo = (
        df_detalhe.groupby("modelo", dropna=False)[
            [
                "bleu",
                "rouge1_f1",
                "rouge2_f1",
                "rougeL_f1",
                "bert_f1",
                "sbert_similaridade",
                "keyword_score",
                "final",
            ]
        ]
        .mean()
        .reset_index()
        .sort_values("final", ascending=False)
        .reset_index(drop=True)
    )

    _log_status("[Reference] Avaliacao concluida.", show_progress)
    return df_detalhe, df_resumo
