from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from config import (
    MODELO_RF_ARGUMENTACAO,
    MODELO_RF_COESAO,
    MODELO_RF_PRECISAO,
    MODELO_RF_PRECISAO_FALLBACK,
    PESO_RF_ARGUMENTACAO,
    PESO_RF_COESAO,
    PESO_RF_PRECISAO,
    PESO_RF_PRECISAO_PERGUNTA,
    PESO_RF_PRECISAO_PROPOSICIONAL,
)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def decompose_into_propositions(text: str) -> list[str]:
    """
    Decompõe um texto em proposições usando heurística simples.

    A implementação foi mantida propositalmente simples para ser estável no Colab
    e facilitar futura troca por um decompositor baseado em LLM.
    """
    texto = normalize_text(text)
    if not texto:
        return []
    # Evita quebrar abreviações jurídicas e honoríficas muito comuns.
    texto = re.sub(
        r"\b(art|arts|n[ºo]|nr|sr|sra|dr|dra|prof|profa|inc|incs)\.",
        lambda match: match.group(0).replace(".", "<DOT>"),
        texto,
        flags=re.IGNORECASE,
    )
    partes = re.split(r"[.!?;\n]+", texto)
    partes = [parte.replace("<DOT>", ".") for parte in partes]
    return [parte.strip(" -\t\r") for parte in partes if parte.strip(" -\t\r")]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Calcula similaridade de cosseno assumindo vetores já normalizados em L2."""
    assert np.isfinite(a).all() and np.isfinite(b).all(), "Embeddings inválidos para cosine_sim."
    return float(np.dot(a, b))


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _log_status(message: str, verbose: bool = True) -> None:
    if verbose:
        print(message)


def _mean_normalized(vectors: list[np.ndarray]) -> np.ndarray:
    media = np.mean(vectors, axis=0)
    norma = np.linalg.norm(media)
    if norma == 0:
        return media
    return media / norma


@dataclass
class EmbeddingBackend:
    model_name: str
    prefix: str | None = None
    batch_size: int = 8
    verbose: bool = True

    def __post_init__(self) -> None:
        _log_status(f"[Reference-Free] Carregando embeddings: {self.model_name}", self.verbose)
        self.device = _device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        self._cache: dict[str, np.ndarray] = {}

    def _normalize_text(self, text: str) -> str:
        texto = normalize_text(text)
        if self.prefix:
            return f"{self.prefix}: {texto}"
        return texto

    def _mean_pool(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        soma = torch.sum(hidden_state * mask, dim=1)
        contagem = torch.clamp(mask.sum(dim=1), min=1e-9)
        return soma / contagem

    def get_embeddings(self, texts: Iterable[str]) -> dict[str, np.ndarray]:
        textos_brutos = [normalize_text(text) for text in texts]
        textos_unicos = [texto for texto in dict.fromkeys(textos_brutos) if texto]
        faltantes = [texto for texto in textos_unicos if texto not in self._cache]

        for inicio in range(0, len(faltantes), self.batch_size):
            lote = faltantes[inicio : inicio + self.batch_size]
            lote_modelo = [self._normalize_text(texto) for texto in lote]
            inputs = self.tokenizer(
                lote_modelo,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = self._mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            for texto, embedding in zip(lote, embeddings):
                self._cache[texto] = embedding.detach().cpu().numpy()

        return {texto: self._cache[texto] for texto in textos_unicos if texto in self._cache}


@dataclass
class NLIBackend:
    model_name: str = MODELO_RF_COESAO
    batch_size: int = 8
    verbose: bool = True

    def __post_init__(self) -> None:
        _log_status(f"[Reference-Free] Carregando NLI: {self.model_name}", self.verbose)
        self.device = _device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        self.id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}

    def _mapear_labels(self, probabilidades: np.ndarray) -> tuple[float, float]:
        entailment = 0.0
        contradiction = 0.0

        for idx, prob in enumerate(probabilidades):
            label = self.id2label.get(idx, "")
            if "entail" in label:
                entailment = float(prob)
            elif "contra" in label:
                contradiction = float(prob)

        return entailment, contradiction

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []

        scores: list[float] = []
        for inicio in range(0, len(pairs), self.batch_size):
            lote = pairs[inicio : inicio + self.batch_size]
            textos_a = [a for a, _ in lote]
            textos_b = [b for _, b in lote]

            inputs = self.tokenizer(
                textos_a,
                textos_b,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

            for prob in probs:
                entailment, contradiction = self._mapear_labels(prob)
                scores.append(entailment - contradiction)

        return scores


def get_embeddings(texts: Iterable[str], backend: EmbeddingBackend) -> dict[str, np.ndarray]:
    return backend.get_embeddings(texts)


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


def score_coesao(props: list[str], backend: NLIBackend) -> float:
    if not props:
        return 0.0
    if len(props) == 1:
        return 1.0

    pairs = [(props[i], props[i + 1]) for i in range(len(props) - 1)]

    if len(props) >= 3:
        pairs.append((props[0], props[2]))
    if len(props) >= 2:
        pair_global = (props[0], props[-1])
        if pair_global not in pairs:
            pairs.append(pair_global)

    scores = backend.score_pairs(pairs)
    return float(np.mean(scores)) if scores else 0.0


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
    answers: dict[str, str],
    question: str,
    backend_argumentacao: EmbeddingBackend,
    backend_precisao: EmbeddingBackend,
    backend_nli: NLIBackend,
) -> tuple[dict[str, dict[str, float]], list[dict[str, float | str]]]:
    respostas_limpas = {modelo: normalize_text(texto) for modelo, texto in answers.items()}
    proposicoes = {modelo: decompose_into_propositions(texto) for modelo, texto in respostas_limpas.items()}

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
            str(row["modelo"]): str(row["resposta"] or "")
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
