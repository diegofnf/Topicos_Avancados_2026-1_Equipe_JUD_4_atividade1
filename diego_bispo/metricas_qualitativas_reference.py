from __future__ import annotations

import json
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from bert_score import score as bert_score_fn
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from tqdm.auto import tqdm

from config import (
    MODELO_BERTSCORE,
    MODELO_R_FACTSCORE,
    MODELO_R_SBERT,
    PESO_R_ARGUMENTACAO,
    PESO_R_ARG_ROUGE1,
    PESO_R_ARG_ROUGE2,
    PESO_R_ARG_ROUGEL,
    PESO_R_COESAO,
    PESO_R_FACTSCORE,
    PESO_R_PRECISAO,
    PESO_R_PRECISAO_BERT,
    PESO_R_PRECISAO_SBERT,
)
from metricas_qualitativas import NLIBackend, normalize_text, obter_proposicoes_resposta


def _log_status(message: str, verbose: bool = True) -> None:
    if verbose:
        print(message)


@dataclass
class MetricasQuestao:
    question_id: str
    modelo: str
    coesao: float = 0.0
    rouge1_f1: float = 0.0
    rouge2_f1: float = 0.0
    rougeL_f1: float = 0.0
    argumentacao: float = 0.0
    sbert_precisao: float = 0.0
    bert_f1: float = 0.0
    precisao: float = 0.0
    factscore: float = 0.0
    final: float = 0.0


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


def _carregar_sbert(verbose: bool = True) -> SentenceTransformer:
    _log_status(f"[Reference] Carregando SBERT: {MODELO_R_SBERT}", verbose)
    return SentenceTransformer(MODELO_R_SBERT)


def _carregar_nli(verbose: bool = True) -> NLIBackend:
    return NLIBackend(model_name=MODELO_R_FACTSCORE, verbose=verbose)


def _similaridade_sbert(texto_a: str, texto_b: str, modelo: SentenceTransformer) -> float:
    texto_a = normalize_text(texto_a)
    texto_b = normalize_text(texto_b)
    if not texto_a or not texto_b:
        return 0.0
    emb_a = modelo.encode(texto_a, convert_to_tensor=True)
    emb_b = modelo.encode(texto_b, convert_to_tensor=True)
    return float(util.cos_sim(emb_a, emb_b)[0][0])


def _score_coesao(resposta: str, sbert_model: SentenceTransformer) -> float:
    sentencas = obter_proposicoes_resposta(resposta)
    if not sentencas:
        return 0.0
    if len(sentencas) == 1:
        return 1.0

    embeddings = sbert_model.encode(sentencas, convert_to_tensor=True)
    scores = [
        float(util.cos_sim(embeddings[i], embeddings[i + 1])[0][0])
        for i in range(len(sentencas) - 1)
    ]
    return float(np.mean(scores)) if scores else 0.0


def _score_argumentacao(gabarito: str, resposta: str) -> tuple[float, float, float, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    scores = scorer.score(normalize_text(gabarito), normalize_text(resposta))
    rouge1_f1 = float(scores["rouge1"].fmeasure)
    rouge2_f1 = float(scores["rouge2"].fmeasure)
    rougeL_f1 = float(scores["rougeL"].fmeasure)
    argumentacao = (
        (PESO_R_ARG_ROUGE1 * rouge1_f1)
        + (PESO_R_ARG_ROUGE2 * rouge2_f1)
        + (PESO_R_ARG_ROUGEL * rougeL_f1)
    )
    return rouge1_f1, rouge2_f1, rougeL_f1, float(argumentacao)


def _score_bert(referencia: str, hipotese: str) -> float:
    try:
        precision, recall, f1 = bert_score_fn(
            [hipotese],
            [referencia],
            lang="pt",
            model_type=MODELO_BERTSCORE,
            verbose=False,
        )
    except KeyError:
        warnings.warn(
            f"[Reference] Modelo de BERTScore '{MODELO_BERTSCORE}' nao suportado. "
            "Usando fallback: 'bert-base-multilingual-cased'.",
            RuntimeWarning,
            stacklevel=2,
        )
        precision, recall, f1 = bert_score_fn(
            [hipotese],
            [referencia],
            lang="pt",
            model_type="bert-base-multilingual-cased",
            verbose=False,
        )
    return float(f1[0])


def _score_precisao(gabarito: str, resposta: str, sbert_model: SentenceTransformer) -> tuple[float, float, float]:
    sbert_score = _similaridade_sbert(gabarito, resposta, sbert_model)
    bert_f1 = _score_bert(gabarito, resposta)
    precisao = (PESO_R_PRECISAO_SBERT * sbert_score) + (PESO_R_PRECISAO_BERT * bert_f1)
    return float(sbert_score), float(bert_f1), float(precisao)


def _proposicoes_gabarito(gabarito: str, gabarito_itens_json: str | list | None) -> list[str]:
    proposicoes = obter_proposicoes_resposta(gabarito)
    if proposicoes:
        return proposicoes

    itens = _parse_gabarito_itens(gabarito_itens_json)
    props_itens = [normalize_text(item.get("texto", "")) for item in itens if normalize_text(item.get("texto", ""))]
    return props_itens or [normalize_text(gabarito)]


def _score_factscore(
    resposta: str,
    gabarito: str,
    gabarito_itens_json: str | list | None,
    nli_backend: NLIBackend,
) -> float:
    props_resposta = obter_proposicoes_resposta(resposta)
    props_gabarito = _proposicoes_gabarito(gabarito, gabarito_itens_json)

    if not props_resposta or not props_gabarito:
        return 0.0

    suporte_por_proposicao: list[float] = []
    for prop_resposta in props_resposta:
        pares = [(prop_resposta, prop_gabarito) for prop_gabarito in props_gabarito]
        scores = nli_backend.score_pairs(pares)
        if not scores:
            suporte_por_proposicao.append(0.0)
            continue
        melhor_suporte = max(scores)
        suporte_por_proposicao.append((melhor_suporte + 1.0) / 2.0)

    return float(np.mean(suporte_por_proposicao)) if suporte_por_proposicao else 0.0


def _avaliar_resposta(
    question_id: str,
    modelo: str,
    resposta: str,
    gabarito_narrativo: str,
    gabarito_itens_json: str | list | None,
    sbert_model: SentenceTransformer,
    nli_backend: NLIBackend,
) -> MetricasQuestao:
    resposta = normalize_text(resposta)
    gabarito_narrativo = normalize_text(gabarito_narrativo)

    metricas = MetricasQuestao(question_id=question_id, modelo=modelo)
    if not resposta or not gabarito_narrativo:
        return metricas

    metricas.coesao = round(_score_coesao(resposta, sbert_model), 4)
    (
        metricas.rouge1_f1,
        metricas.rouge2_f1,
        metricas.rougeL_f1,
        metricas.argumentacao,
    ) = [round(valor, 4) for valor in _score_argumentacao(gabarito_narrativo, resposta)]

    (
        metricas.sbert_precisao,
        metricas.bert_f1,
        metricas.precisao,
    ) = [round(valor, 4) for valor in _score_precisao(gabarito_narrativo, resposta, sbert_model)]

    metricas.factscore = round(
        _score_factscore(
            resposta,
            gabarito_narrativo,
            gabarito_itens_json,
            nli_backend,
        ),
        4,
    )

    metricas.final = round(
        (PESO_R_COESAO * metricas.coesao)
        + (PESO_R_ARGUMENTACAO * metricas.argumentacao)
        + (PESO_R_PRECISAO * metricas.precisao)
        + (PESO_R_FACTSCORE * metricas.factscore),
        4,
    )
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

    sbert_model = _carregar_sbert(show_progress)
    nli_backend = _carregar_nli(show_progress)

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
            nli_backend=nli_backend,
        )

        linhas.append(
            {
                "question_id": metricas.question_id,
                "modelo": metricas.modelo,
                "coesao": metricas.coesao,
                "rouge1_f1": metricas.rouge1_f1,
                "rouge2_f1": metricas.rouge2_f1,
                "rougeL_f1": metricas.rougeL_f1,
                "argumentacao": metricas.argumentacao,
                "sbert_precisao": metricas.sbert_precisao,
                "bert_f1": metricas.bert_f1,
                "precisao": metricas.precisao,
                "factscore": metricas.factscore,
                "final": metricas.final,
            }
        )

    df_detalhe = pd.DataFrame(linhas)
    if df_detalhe.empty:
        return df_detalhe, pd.DataFrame(
            columns=[
                "modelo",
                "coesao",
                "rouge1_f1",
                "rouge2_f1",
                "rougeL_f1",
                "argumentacao",
                "sbert_precisao",
                "bert_f1",
                "precisao",
                "factscore",
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
                "coesao",
                "rouge1_f1",
                "rouge2_f1",
                "rougeL_f1",
                "argumentacao",
                "sbert_precisao",
                "bert_f1",
                "precisao",
                "factscore",
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
