import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from config import (
    MODELO_RF_ARGUMENTACAO,
    MODELO_RF_PRECISAO,
    MODELO_RF_PRECISAO_FALLBACK,
    PESO_RF_ARGUMENTACAO,
    PESO_RF_COESAO,
    PESO_RF_PRECISAO,
    PESO_RF_PRECISAO_PERGUNTA,
    PESO_RF_PRECISAO_PROPOSICIONAL,
)
from metricas_qualitativas import (
    EmbeddingBackend,
    NLIBackend,
    _log_status,
    _mean_normalized,
    cosine_sim,
    get_embeddings,
    normalize_text,
    obter_proposicoes_resposta,
    score_coesao,
)


def score_argumentacao(props: list[str], backend: EmbeddingBackend) -> float:
    if not props:
        return 0.0
    if len(props) == 1:
        return 1.0

    embeddings = get_embeddings(props, backend)

    fluidez = [
        cosine_sim(embeddings[props[i]], embeddings[props[i + 1]])
        for i in range(len(props) - 1)
    ]

    # Em vez de concatenar todas as proposições anteriores em um texto longo,
    # agregamos seus embeddings já normalizados. Isso preserva a ideia de
    # "construção do argumento" sem diluir tanto o sinal semântico.
    construcao = [
        cosine_sim(
            _mean_normalized([embeddings[props[j]] for j in range(i + 1)]),
            embeddings[props[i + 1]],
        )
        for i in range(len(props) - 1)
    ]

    score_fluidez = float(np.mean(fluidez)) if fluidez else 0.0
    score_const = float(np.mean(construcao)) if construcao else 0.0
    return 0.5 * score_fluidez + 0.5 * score_const


def score_precisao(
    props_a: list[str],
    respostas_referencia: list[list[str]],
    question: str,
    answer_text: str,
    backend: EmbeddingBackend,
) -> float:
    """
    Mede dois sinais complementares:
    1. alinhamento proposicional entre respostas concorrentes
    2. relevância da resposta completa para a pergunta

    O score final mistura ambos, então "precisão" aqui não significa apenas
    concordância entre modelos; também incorpora aderência ao enunciado.
    """
    if not props_a:
        return 0.0

    question_norm = normalize_text(question)
    answer_norm = normalize_text(answer_text)
    props_norm = [normalize_text(prop) for prop in props_a]
    referencias_norm = [[normalize_text(prop) for prop in resposta if normalize_text(prop)] for resposta in respostas_referencia]

    todos_textos = list(props_norm) + [question_norm, answer_norm]
    for resposta in referencias_norm:
        todos_textos.extend(resposta)

    embeddings = get_embeddings(todos_textos, backend)
    scores_prop = []

    for prop in props_norm:
        matches = []
        for resposta in referencias_norm:
            if not resposta:
                continue
            sims = [cosine_sim(embeddings[prop], embeddings[outra_prop]) for outra_prop in resposta]
            matches.append(max(sims))
        scores_prop.append(float(np.mean(matches)) if matches else 0.0)

    score_prop = float(np.mean(scores_prop)) if scores_prop else 0.0
    score_q = cosine_sim(embeddings[answer_norm], embeddings[question_norm])
    return (PESO_RF_PRECISAO_PROPOSICIONAL * score_prop) + (PESO_RF_PRECISAO_PERGUNTA * score_q)
def _tentar_backend_precisao(verbose: bool = True) -> EmbeddingBackend:
    try:
        return EmbeddingBackend(MODELO_RF_PRECISAO, verbose=verbose)
    except Exception:
        _log_status(
            f"[Reference-Free] Fallback de precisão ativado: {MODELO_RF_PRECISAO_FALLBACK}",
            verbose,
        )
        return EmbeddingBackend(MODELO_RF_PRECISAO_FALLBACK, verbose=verbose)


def _evaluate_all_with_backends(
    answers: dict[str, str | dict[str, object]],
    question: str,
    backend_argumentacao: EmbeddingBackend,
    backend_precisao: EmbeddingBackend,
    backend_nli: NLIBackend,
) -> tuple[dict[str, dict[str, float]], list[dict[str, float | str]]]:
    respostas_limpas: dict[str, str] = {}
    proposicoes: dict[str, list[str]] = {}

    for modelo, payload in answers.items():
        if isinstance(payload, dict):
            resposta = normalize_text(str(payload.get("resposta", "")))
            props = obter_proposicoes_resposta(resposta, payload.get("proposicoes_json"))
        else:
            resposta = normalize_text(str(payload))
            props = obter_proposicoes_resposta(resposta)

        respostas_limpas[modelo] = resposta
        proposicoes[modelo] = props

    resultados: dict[str, dict[str, float]] = {}
    for modelo, props in proposicoes.items():
        outras_props = [proposicoes[outro] for outro in proposicoes if outro != modelo]
        score_arg = score_argumentacao(props, backend_argumentacao)
        score_pre = score_precisao(props, outras_props, question, respostas_limpas[modelo], backend_precisao)
        score_coe = score_coesao(props, backend_nli)
        score_final = (
            (PESO_RF_ARGUMENTACAO * score_arg)
            + (PESO_RF_PRECISAO * score_pre)
            + (PESO_RF_COESAO * score_coe)
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
    question: str,
    verbose: bool = True,
) -> tuple[dict[str, dict[str, float]], list[dict[str, float | str]]]:
    """
    Avalia todas as respostas de uma mesma questão e retorna scores por modelo e ranking final.
    """
    backend_argumentacao = EmbeddingBackend(MODELO_RF_ARGUMENTACAO, prefix="passage", verbose=verbose)
    backend_precisao = _tentar_backend_precisao(verbose=verbose)
    backend_nli = NLIBackend(verbose=verbose)
    return _evaluate_all_with_backends(
        answers,
        question,
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

    Espera colunas:
    - question_id
    - texto_questao
    - modelo
    - resposta
    """
    colunas_obrigatorias = {"question_id", "texto_questao", "modelo", "resposta"}
    ausentes = sorted(colunas_obrigatorias - set(df_respostas.columns))
    if ausentes:
        raise ValueError(f"Colunas obrigatórias ausentes: {', '.join(ausentes)}")

    _log_status("[Reference-Free] Inicializando backends de avaliação...", show_progress)
    backend_argumentacao = EmbeddingBackend(MODELO_RF_ARGUMENTACAO, prefix="passage", verbose=show_progress)
    backend_precisao = _tentar_backend_precisao(verbose=show_progress)
    backend_nli = NLIBackend(verbose=show_progress)

    linhas_detalhe: list[dict[str, object]] = []
    grupos = list(df_respostas.groupby("question_id", dropna=False))
    iterador = tqdm(
        grupos,
        total=len(grupos),
        desc="Reference-Free por questão",
        disable=not show_progress,
    )
    for question_id, grupo in iterador:
        answers = {
            str(row["modelo"]): {
                "resposta": str(row["resposta"] or ""),
                "proposicoes_json": row.get("proposicoes_json"),
            }
            for _, row in grupo.iterrows()
        }
        question = str(grupo["texto_questao"].iloc[0] or "")
        _, ranking = _evaluate_all_with_backends(
            answers,
            question,
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
                }
            )

    df_detalhe = pd.DataFrame(linhas_detalhe)
    if df_detalhe.empty:
        return df_detalhe, pd.DataFrame(columns=["modelo", "argumentacao", "precisao", "coesao", "final"])

    df_resumo = (
        df_detalhe.groupby("modelo", dropna=False)[["argumentacao", "precisao", "coesao", "final"]]
        .mean()
        .reset_index()
        .sort_values("final", ascending=False)
        .reset_index(drop=True)
    )
    _log_status("[Reference-Free] Avaliação concluída.", show_progress)
    return df_detalhe, df_resumo
