from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from config import (
    MODELO_R_SBERT,
    PESO_R_ARGUMENTACAO,
    PESO_R_COESAO,
    PESO_R_PRECISAO,
)
from metricas_qualitativas import EmbeddingBackend, _log_status, cosine_sim, normalize_text


@dataclass
class GabaritoItem:
    texto: str
    peso_maximo: float | None
    secao: str | None = None


@dataclass
class MetricasQuestao:
    question_id: str
    modelo: str
    argumentacao: float = 0.0
    precisao: float = 0.0
    coesao_legal: float = 0.0
    final: float = 0.0


def carregar_itens_gabarito(gabarito_itens_json: str | list | None) -> list[GabaritoItem]:
    if isinstance(gabarito_itens_json, str):
        try:
            dados = json.loads(gabarito_itens_json)
        except Exception:
            dados = []
    elif isinstance(gabarito_itens_json, list):
        dados = gabarito_itens_json
    else:
        dados = []

    return [
        GabaritoItem(
            texto=str(item.get("texto", "")),
            peso_maximo=item.get("peso_maximo"),
            secao=item.get("secao"),
        )
        for item in dados
        if isinstance(item, dict) and item.get("texto")
    ]


def _extrair_secoes_ordenadas(itens: list[GabaritoItem]) -> list[str]:
    vistas: set[str] = set()
    secoes: list[str] = []
    for item in itens:
        if item.secao and item.secao not in vistas:
            vistas.add(item.secao)
            secoes.append(item.secao)
    return secoes


def _carregar_backend_sbert(verbose: bool = True) -> EmbeddingBackend:
    return EmbeddingBackend(
        model_name=MODELO_R_SBERT,
        prefix="passage",
        verbose=verbose,
    )


def _similaridade(texto_a: str, texto_b: str, backend: EmbeddingBackend) -> float:
    texto_a = normalize_text(texto_a)
    texto_b = normalize_text(texto_b)
    if not texto_a or not texto_b:
        return 0.0

    embeddings = backend.get_embeddings([texto_a, texto_b])
    if texto_a not in embeddings or texto_b not in embeddings:
        return 0.0
    return cosine_sim(embeddings[texto_a], embeddings[texto_b])


def score_precisao(
    resposta: str,
    itens: list[GabaritoItem],
    backend: EmbeddingBackend,
) -> float:
    """
    Mede cobertura dos itens do gabarito via similaridade ponderada por peso.
    """
    resposta_norm = normalize_text(resposta)
    if not resposta_norm:
        return 0.0

    itens_com_peso = [item for item in itens if item.peso_maximo is not None]
    if not itens_com_peso:
        return 0.0

    textos_item = [normalize_text(item.texto) for item in itens_com_peso if normalize_text(item.texto)]
    if not textos_item:
        return 0.0

    embeddings = backend.get_embeddings([resposta_norm] + textos_item)
    if resposta_norm not in embeddings:
        return 0.0

    soma_ponderada = 0.0
    soma_pesos = 0.0

    for item, texto_item in zip(itens_com_peso, textos_item):
        if texto_item not in embeddings:
            continue
        sim = cosine_sim(embeddings[resposta_norm], embeddings[texto_item])
        soma_ponderada += sim * float(item.peso_maximo)
        soma_pesos += float(item.peso_maximo)

    return (soma_ponderada / soma_pesos) if soma_pesos else 0.0


def score_argumentacao(
    resposta: str,
    secoes: list[str],
    backend: EmbeddingBackend,
) -> float:
    """
    Mede presença da estrutura jurídica esperada via cobertura das seções.
    """
    resposta_norm = normalize_text(resposta)
    if not resposta_norm or not secoes:
        return 0.0

    secoes_norm = [normalize_text(secao) for secao in secoes if normalize_text(secao)]
    if not secoes_norm:
        return 0.0

    embeddings = backend.get_embeddings([resposta_norm] + secoes_norm)
    if resposta_norm not in embeddings:
        return 0.0

    scores = [
        cosine_sim(embeddings[resposta_norm], embeddings[secao])
        for secao in secoes_norm
        if secao in embeddings
    ]
    return float(np.mean(scores)) if scores else 0.0


def score_coesao_legal(
    resposta: str,
    gabarito_narrativo: str,
    backend: EmbeddingBackend,
) -> float:
    """
    Mede alinhamento global da resposta com o raciocínio jurídico do gabarito.
    """
    return _similaridade(resposta, gabarito_narrativo, backend)


def _avaliar_resposta(
    question_id: str,
    modelo: str,
    resposta: str,
    gabarito_narrativo: str,
    gabarito_itens_json: str | list | None,
    backend: EmbeddingBackend,
) -> MetricasQuestao:
    resposta = normalize_text(resposta)
    gabarito_narrativo = normalize_text(gabarito_narrativo)

    metricas = MetricasQuestao(question_id=question_id, modelo=modelo)
    if not resposta or not gabarito_narrativo:
        return metricas

    itens = carregar_itens_gabarito(gabarito_itens_json)
    secoes = _extrair_secoes_ordenadas(itens)

    metricas.argumentacao = round(score_argumentacao(resposta, secoes, backend), 4)
    metricas.precisao = round(score_precisao(resposta, itens, backend), 4)
    metricas.coesao_legal = round(score_coesao_legal(resposta, gabarito_narrativo, backend), 4)
    metricas.final = round(
        (PESO_R_ARGUMENTACAO * metricas.argumentacao)
        + (PESO_R_PRECISAO * metricas.precisao)
        + (PESO_R_COESAO * metricas.coesao_legal),
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
        "gabarito_itens_json",
    }
    ausentes = sorted(colunas_obrigatorias - set(df_respostas.columns))
    if ausentes:
        raise ValueError(f"Colunas obrigatorias ausentes: {', '.join(ausentes)}")

    _log_status("[Reference] Inicializando backend SBERT...", show_progress)
    backend = _carregar_backend_sbert(show_progress)

    linhas: list[dict[str, object]] = []
    registros = df_respostas.to_dict("records")
    iterador = tqdm(
        registros,
        total=len(registros),
        desc="Reference por resposta",
        disable=not show_progress,
    )

    for row in iterador:
        metricas = _avaliar_resposta(
            question_id=str(row["question_id"]),
            modelo=str(row["modelo"]),
            resposta=str(row.get("resposta") or ""),
            gabarito_narrativo=str(row.get("gabarito_narrativo") or ""),
            gabarito_itens_json=row.get("gabarito_itens_json"),
            backend=backend,
        )
        linhas.append(
            {
                "question_id": metricas.question_id,
                "modelo": metricas.modelo,
                "argumentacao": metricas.argumentacao,
                "precisao": metricas.precisao,
                "coesao_legal": metricas.coesao_legal,
                "final": metricas.final,
            }
        )

    df_detalhe = pd.DataFrame(linhas)
    if df_detalhe.empty:
        return df_detalhe, pd.DataFrame(
            columns=["modelo", "argumentacao", "precisao", "coesao_legal", "final"]
        )

    df_detalhe["ranking"] = (
        df_detalhe.groupby("question_id")["final"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    df_resumo = (
        df_detalhe.groupby("modelo", dropna=False)[
            ["argumentacao", "precisao", "coesao_legal", "final"]
        ]
        .mean()
        .reset_index()
        .sort_values("final", ascending=False)
        .reset_index(drop=True)
    )

    _log_status("[Reference] Avaliação concluída.", show_progress)
    return df_detalhe, df_resumo
