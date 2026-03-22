from __future__ import annotations

import importlib
from typing import Iterable, Sequence

import pandas as pd
from config import MODELO_BERTSCORE


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
