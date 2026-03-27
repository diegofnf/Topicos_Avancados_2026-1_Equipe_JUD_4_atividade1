from __future__ import annotations

import importlib
import json
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from config import MAX_LENGTH_RF_EMBEDDINGS, MAX_LENGTH_RF_NLI, MODELO_BERTSCORE, MODELO_RF_COESAO


COLUNA_NOTA = "nota"
COLUNAS_TEXTO_CANDIDATAS = ("resposta", "texto_resposta")
COLUNAS_MODELO_CANDIDATAS = ("modelo", "modelo_candidato")


def _validar_colunas(df: pd.DataFrame, colunas: Iterable[str]) -> None:
    ausentes = sorted(set(colunas) - set(df.columns))
    if ausentes:
        raise ValueError(f"Colunas obrigatorias ausentes: {', '.join(ausentes)}")


def _resolver_coluna(df: pd.DataFrame, candidatas: Sequence[str], descricao: str) -> str:
    for coluna in candidatas:
        if coluna in df.columns:
            return coluna
    raise ValueError(f"Nao foi encontrada uma coluna valida para {descricao}.")


def _normalizar_texto(valor: object) -> str:
    texto = "" if pd.isna(valor) else str(valor)
    return texto.strip()


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
    texto = re.sub(
        r"\b(art|arts|n[ºo]|nr|sr|sra|dr|dra|prof|profa|inc|incs)\.",
        lambda match: match.group(0).replace(".", "<DOT>"),
        texto,
        flags=re.IGNORECASE,
    )
    partes = re.split(r"[.!?;\n]+", texto)
    partes = [parte.replace("<DOT>", ".") for parte in partes]
    return [parte.strip(" -\t\r") for parte in partes if parte.strip(" -\t\r")]


def carregar_proposicoes_json(proposicoes_json: str | list | None) -> list[str]:
    """Extrai apenas os textos das proposições a partir de JSON serializado ou lista."""
    if isinstance(proposicoes_json, str):
        try:
            dados = json.loads(proposicoes_json)
        except Exception:
            dados = []
    elif isinstance(proposicoes_json, list):
        dados = proposicoes_json
    else:
        dados = []

    proposicoes: list[str] = []
    for item in dados:
        if isinstance(item, dict):
            texto = normalize_text(item.get("proposicao", ""))
        else:
            texto = normalize_text(item)
        if texto:
            proposicoes.append(texto)
    return proposicoes


def obter_proposicoes_resposta(resposta: str, proposicoes_json: str | list | None = None) -> list[str]:
    """
    Usa `proposicoes_json` como fonte principal e cai para decomposição heurística
    apenas quando as proposições estruturadas não estiverem disponíveis.
    """
    proposicoes = carregar_proposicoes_json(proposicoes_json)
    if proposicoes:
        return proposicoes
    return decompose_into_propositions(resposta)


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
    max_length: int = MAX_LENGTH_RF_EMBEDDINGS

    def __post_init__(self) -> None:
        _log_status(f"[Qualitativas] Carregando embeddings: {self.model_name}", self.verbose)
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
                max_length=self.max_length,
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
    max_length: int = MAX_LENGTH_RF_NLI

    def __post_init__(self) -> None:
        _log_status(f"[Qualitativas] Carregando NLI: {self.model_name}", self.verbose)
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
                max_length=self.max_length,
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


def _ajustar_tokenizer_bertscore() -> None:
    """Forca um max_length valido para tokenizers de modelos customizados no BERTScore."""
    utils_module = importlib.import_module("bert_score.utils")
    score_module = importlib.import_module("bert_score.score")

    if getattr(utils_module, "_oab_tokenizer_patched", False):
        return

    original_get_tokenizer = utils_module.get_tokenizer

    def get_tokenizer_ajustado(model_type: str, use_fast_tokenizer: bool = False):
        try:
            tokenizer = original_get_tokenizer(model_type, use_fast_tokenizer=use_fast_tokenizer)
        except TypeError:
            tokenizer = original_get_tokenizer(model_type)
        max_length = getattr(tokenizer, "model_max_length", None)
        if max_length is None or max_length > 512:
            tokenizer.model_max_length = 512
        return tokenizer

    utils_module.get_tokenizer = get_tokenizer_ajustado
    score_module.get_tokenizer = get_tokenizer_ajustado
    utils_module._oab_tokenizer_patched = True


def media_notas_por_modelo(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula a media de notas agrupada por modelo."""
    _validar_colunas(df, [COLUNA_NOTA])
    coluna_modelo = _resolver_coluna(df, COLUNAS_MODELO_CANDIDATAS, "modelo")
    if df.empty:
        return pd.DataFrame(columns=[coluna_modelo, "nota_media"])

    resultado = (
        df.groupby(coluna_modelo, dropna=False)[COLUNA_NOTA]
        .mean()
        .reset_index(name="nota_media")
    )
    return resultado.sort_values(coluna_modelo).reset_index(drop=True)


def calcular_bertscore(
    lista_respostas_a: Sequence[str],
    lista_respostas_b: Sequence[str],
    model_type: str = MODELO_BERTSCORE,
    num_layers: int = 12,
) -> tuple[float, list[float]]:
    """Calcula o BERTScore F1 medio e os scores individuais em lote."""
    if len(lista_respostas_a) != len(lista_respostas_b):
        raise ValueError("As listas para BERTScore devem ter o mesmo tamanho.")

    pares = [
        (_normalizar_texto(texto_a), _normalizar_texto(texto_b))
        for texto_a, texto_b in zip(lista_respostas_a, lista_respostas_b)
        if _normalizar_texto(texto_a) and _normalizar_texto(texto_b)
    ]
    if not pares:
        return 0.0, []

    try:
        from bert_score import score
    except ImportError as exc:
        raise ImportError("Instale 'bert-score' para calcular o BERTScore.") from exc

    _ajustar_tokenizer_bertscore()

    candidatos = [candidato for candidato, _ in pares]
    referencias = [referencia for _, referencia in pares]
    _, _, f1 = score(
        candidatos,
        referencias,
        lang="pt",
        model_type=model_type,
        num_layers=num_layers,
        use_fast_tokenizer=False,
        verbose=False,
    )
    scores_individuais = [float(valor) for valor in f1.tolist()]
    return float(f1.mean().item()), scores_individuais


def calcular_matriz_bertscore(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula uma matriz par a par de BERTScore medio entre modelos."""
    coluna_modelo = _resolver_coluna(df, COLUNAS_MODELO_CANDIDATAS, "modelo")
    coluna_texto = _resolver_coluna(df, COLUNAS_TEXTO_CANDIDATAS, "texto de resposta")
    _validar_colunas(df, ["question_id", coluna_modelo, coluna_texto])
    if df.empty:
        return pd.DataFrame()

    base = df[["question_id", coluna_modelo, coluna_texto]].copy()
    base[coluna_texto] = base[coluna_texto].map(_normalizar_texto)
    modelos = sorted(base[coluna_modelo].dropna().astype(str).unique().tolist())
    matriz = pd.DataFrame(1.0, index=modelos, columns=modelos, dtype=float)

    for i, modelo_a in enumerate(modelos):
        for modelo_b in modelos[i + 1 :]:
            pares_a = []
            pares_b = []
            for _, grupo in base.groupby("question_id", dropna=False):
                resposta_a = grupo.loc[grupo[coluna_modelo].astype(str) == modelo_a, coluna_texto]
                resposta_b = grupo.loc[grupo[coluna_modelo].astype(str) == modelo_b, coluna_texto]
                if resposta_a.empty or resposta_b.empty:
                    continue
                texto_a = _normalizar_texto(resposta_a.iloc[0])
                texto_b = _normalizar_texto(resposta_b.iloc[0])
                if texto_a and texto_b:
                    pares_a.append(texto_a)
                    pares_b.append(texto_b)

            score_medio, _ = calcular_bertscore(pares_a, pares_b) if pares_a else (0.0, [])
            matriz.loc[modelo_a, modelo_b] = score_medio
            matriz.loc[modelo_b, modelo_a] = score_medio

    return matriz
