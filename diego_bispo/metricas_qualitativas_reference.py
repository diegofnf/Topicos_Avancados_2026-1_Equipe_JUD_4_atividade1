from __future__ import annotations

import json
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from config import (
    MODELO_RF_ARGUMENTACAO,
    MODELO_RF_COESAO,
    MODELO_RF_PRECISAO,
    MODELO_RF_PRECISAO_FALLBACK,
    PESO_R_ARG_COBERTURA,
    PESO_R_ARG_ORDEM,
    PESO_R_ARGUMENTACAO,
    PESO_R_COESAO,
    PESO_R_PRECISAO,
)
from metricas_qualitativas import (
    EmbeddingBackend,
    NLIBackend,
    _log_status,
    cosine_sim,
    normalize_text,
    obter_proposicoes_resposta,
    score_coesao,
)


@dataclass
class GabaritoItem:
    """
    Representa um critério de avaliação extraído da tabela de pontuação
    do gabarito oficial.
    """

    texto: str
    peso_maximo: float | None
    secao: str | None = None


def carregar_itens_gabarito(gabarito_itens_json: str | list) -> list[GabaritoItem]:
    """
    Deserializa a coluna `gabarito_itens_json` em uma lista de GabaritoItem.
    """
    if isinstance(gabarito_itens_json, str):
        dados = json.loads(gabarito_itens_json)
    else:
        dados = gabarito_itens_json or []

    return [
        GabaritoItem(
            texto=str(dado.get("texto", "")),
            peso_maximo=dado.get("peso_maximo"),
            secao=dado.get("secao"),
        )
        for dado in dados
        if dado.get("texto")
    ]


def _extrair_secoes_ordenadas(itens: list[GabaritoItem]) -> list[str]:
    """
    Extrai as seções únicas dos itens do gabarito preservando a ordem
    de primeira aparição.
    """
    vistas: set[str] = set()
    secoes: list[str] = []
    for item in itens:
        if item.secao and item.secao not in vistas:
            vistas.add(item.secao)
            secoes.append(item.secao)
    return secoes


def score_precisao(
    resposta: str,
    itens: list[GabaritoItem],
    backend: EmbeddingBackend,
    props: list[str] | None = None,
) -> float:
    """
    Mede o quanto a resposta candidata cobre os critérios do gabarito,
    ponderado pelo peso de cada item.
    """
    resposta_norm = normalize_text(resposta)
    if not resposta_norm:
        return 0.0

    itens_com_peso = [item for item in itens if item.peso_maximo is not None]
    if not itens_com_peso:
        return 0.0

    props = props if props is not None else obter_proposicoes_resposta(resposta_norm)
    if not props:
        props = [resposta_norm]

    textos_item = [normalize_text(item.texto) for item in itens_com_peso]
    todos_textos = list(dict.fromkeys(props + textos_item))
    embeddings = backend.get_embeddings(todos_textos)

    soma_ponderada = 0.0
    soma_pesos = 0.0

    for item, texto_item in zip(itens_com_peso, textos_item):
        if texto_item not in embeddings:
            continue

        sims = [
            cosine_sim(embeddings[prop], embeddings[texto_item])
            for prop in props
            if prop in embeddings
        ]
        if not sims:
            continue

        sim_max = float(max(sims))
        soma_ponderada += sim_max * item.peso_maximo
        soma_pesos += item.peso_maximo

    if soma_pesos == 0.0:
        return 0.0

    return soma_ponderada / soma_pesos


def score_argumentacao(
    resposta: str,
    secoes: list[str],
    backend: EmbeddingBackend,
    props: list[str] | None = None,
) -> float:
    """
    Mede cobertura e ordem das seções do gabarito na resposta candidata.
    """
    if not secoes:
        return 0.0

    resposta_norm = normalize_text(resposta)
    if not resposta_norm:
        return 0.0

    props = props if props is not None else obter_proposicoes_resposta(resposta_norm)
    if not props:
        props = [resposta_norm]

    secoes_norm = [normalize_text(secao) for secao in secoes]
    todos_textos = list(dict.fromkeys(props + secoes_norm))
    embeddings = backend.get_embeddings(todos_textos)

    coberturas: list[float] = []
    posicoes: list[int] = []

    for secao_norm in secoes_norm:
        if secao_norm not in embeddings:
            continue

        sims = [
            cosine_sim(embeddings[prop], embeddings[secao_norm])
            for prop in props
            if prop in embeddings
        ]
        if not sims:
            continue

        idx_max = int(np.argmax(sims))
        coberturas.append(sims[idx_max])
        posicoes.append(idx_max)

    if not coberturas:
        return 0.0

    score_cobertura = float(np.mean(coberturas))

    if len(posicoes) >= 2:
        pares_em_ordem = sum(
            1 for i in range(len(posicoes) - 1) if posicoes[i] <= posicoes[i + 1]
        )
        score_ordem = pares_em_ordem / (len(posicoes) - 1)
    else:
        score_ordem = 1.0

    return (PESO_R_ARG_COBERTURA * score_cobertura) + (PESO_R_ARG_ORDEM * score_ordem)


def _tentar_backend_precisao(verbose: bool = True) -> EmbeddingBackend:
    try:
        return EmbeddingBackend(MODELO_RF_PRECISAO, verbose=verbose)
    except Exception:
        warnings.warn(
            f"[Reference] Modelo de precisão '{MODELO_RF_PRECISAO}' indisponível. "
            f"Usando fallback: '{MODELO_RF_PRECISAO_FALLBACK}'.",
            RuntimeWarning,
            stacklevel=2,
        )
        return EmbeddingBackend(MODELO_RF_PRECISAO_FALLBACK, verbose=verbose)


def _evaluate_all_with_backends(
    answers: dict[str, str | dict[str, object]],
    itens_gabarito: list[GabaritoItem],
    backend_argumentacao: EmbeddingBackend,
    backend_precisao: EmbeddingBackend,
    backend_nli: NLIBackend,
) -> tuple[dict[str, dict[str, float]], list[dict[str, float | str]]]:
    """
    Avalia todas as respostas de uma questão contra o gabarito estruturado.
    """
    secoes = _extrair_secoes_ordenadas(itens_gabarito)
    resultados: dict[str, dict[str, float]] = {}

    for modelo, resposta_payload in answers.items():
        if isinstance(resposta_payload, dict):
            resposta = normalize_text(str(resposta_payload.get("resposta", "")))
            props = obter_proposicoes_resposta(
                resposta,
                resposta_payload.get("proposicoes_json"),
            )
        else:
            resposta = normalize_text(str(resposta_payload))
            props = obter_proposicoes_resposta(resposta)

        score_arg = score_argumentacao(resposta, secoes, backend_argumentacao, props=props)
        score_pre = score_precisao(resposta, itens_gabarito, backend_precisao, props=props)
        score_coe = score_coesao(props, backend_nli)

        score_final = (
            (PESO_R_ARGUMENTACAO * score_arg)
            + (PESO_R_PRECISAO * score_pre)
            + (PESO_R_COESAO * score_coe)
        )

        resultados[modelo] = {
            "argumentacao": float(score_arg),
            "precisao": float(score_pre),
            "coesao": float(score_coe),
            "final": float(score_final),
        }

    ranking = [
        {"modelo": modelo, **scores}
        for modelo, scores in sorted(
            resultados.items(),
            key=lambda item: item[1]["final"],
            reverse=True,
        )
    ]
    return resultados, ranking


def evaluate_all(
    answers: dict[str, str],
    gabarito_itens_json: str | list,
    verbose: bool = True,
) -> tuple[dict[str, dict[str, float]], list[dict[str, float | str]]]:
    """
    Avalia todas as respostas de uma mesma questão contra o gabarito oficial.
    """
    itens = carregar_itens_gabarito(gabarito_itens_json)
    backend_argumentacao = EmbeddingBackend(
        MODELO_RF_ARGUMENTACAO, prefix="passage", verbose=verbose
    )
    backend_precisao = _tentar_backend_precisao(verbose=verbose)
    backend_nli = NLIBackend(model_name=MODELO_RF_COESAO, verbose=verbose)

    return _evaluate_all_with_backends(
        answers,
        itens,
        backend_argumentacao=backend_argumentacao,
        backend_precisao=backend_precisao,
        backend_nli=backend_nli,
    )


def evaluate_dataframe(
    df_respostas: pd.DataFrame,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Avalia um DataFrame de respostas discursivas agrupando por questão.
    """
    colunas_obrigatorias = {"question_id", "modelo", "resposta", "gabarito_itens_json"}
    ausentes = sorted(colunas_obrigatorias - set(df_respostas.columns))
    if ausentes:
        raise ValueError(f"Colunas obrigatórias ausentes: {', '.join(ausentes)}")

    _log_status("[Reference] Inicializando backends de avaliação...", show_progress)
    backend_argumentacao = EmbeddingBackend(
        MODELO_RF_ARGUMENTACAO, prefix="passage", verbose=show_progress
    )
    backend_precisao = _tentar_backend_precisao(verbose=show_progress)
    backend_nli = NLIBackend(model_name=MODELO_RF_COESAO, verbose=show_progress)

    linhas_detalhe: list[dict[str, object]] = []
    grupos = list(df_respostas.groupby("question_id", dropna=False))

    iterador = tqdm(
        grupos,
        total=len(grupos),
        desc="Reference por questão",
        disable=not show_progress,
    )

    for question_id, grupo in iterador:
        gabarito_raw = next(
            (valor for valor in grupo["gabarito_itens_json"] if pd.notna(valor) and valor),
            None,
        )
        if not gabarito_raw:
            _log_status(
                f"[Reference] Questão '{question_id}' sem gabarito_itens_json — ignorada.",
                show_progress,
            )
            continue

        itens = carregar_itens_gabarito(gabarito_raw)
        secoes = _extrair_secoes_ordenadas(itens)
        n_itens = sum(1 for item in itens if item.peso_maximo is not None)

        answers = {
            str(row["modelo"]): {
                "resposta": str(row["resposta"] or ""),
                "proposicoes_json": row.get("proposicoes_json"),
            }
            for _, row in grupo.iterrows()
        }

        _, ranking = _evaluate_all_with_backends(
            answers,
            itens,
            backend_argumentacao=backend_argumentacao,
            backend_precisao=backend_precisao,
            backend_nli=backend_nli,
        )

        for posicao, item in enumerate(ranking, start=1):
            linhas_detalhe.append(
                {
                    "question_id": question_id,
                    "modelo": item["modelo"],
                    "argumentacao": item["argumentacao"],
                    "precisao": item["precisao"],
                    "coesao": item["coesao"],
                    "final": item["final"],
                    "ranking": posicao,
                    "n_secoes_gabarito": len(secoes),
                    "n_itens_gabarito": n_itens,
                }
            )

    df_detalhe = pd.DataFrame(linhas_detalhe)
    if df_detalhe.empty:
        return df_detalhe, pd.DataFrame(
            columns=["modelo", "argumentacao", "precisao", "coesao", "final"]
        )

    df_resumo = (
        df_detalhe.groupby("modelo", dropna=False)[
            ["argumentacao", "precisao", "coesao", "final"]
        ]
        .mean()
        .reset_index()
        .sort_values("final", ascending=False)
        .reset_index(drop=True)
    )

    _log_status("[Reference] Avaliação concluída.", show_progress)
    return df_detalhe, df_resumo
