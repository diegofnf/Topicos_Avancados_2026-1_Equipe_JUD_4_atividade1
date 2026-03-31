"""Avaliacao discursiva estruturada com base em criterios do gabarito."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from configuracoes import CaminhosSaida
from utilitarios import carregar_json_seguro, normalizar_texto, salvar_csv, segmentar_sentencas


@dataclass
class ComponenteCriterio:
    """Representa um componente elementar de avaliacao."""

    tipo: str
    peso: float
    referencia: str = ""
    termos: list[str] = field(default_factory=list)


@dataclass
class CriterioCorrecao:
    """Representa um criterio do gabarito estruturado."""

    identificador: str
    secao: str
    peso_total: float
    componentes: list[ComponenteCriterio]


def normalizar_legislacao(texto: str) -> str:
    """Padroniza o texto para comparacao de referencias legais."""
    texto_normalizado = normalizar_texto(texto).lower()
    texto_normalizado = texto_normalizado.replace("artigo", "art")
    texto_normalizado = texto_normalizado.replace("nº", "")
    texto_normalizado = texto_normalizado.replace("°", "")
    return texto_normalizado


def verificar_legislacao(resposta: str, termos: list[str]) -> bool:
    """Verifica se algum termo legal esperado aparece na resposta."""
    texto = normalizar_legislacao(resposta)
    return any(normalizar_legislacao(termo) in texto for termo in termos)


def calcular_nli(resposta: str, referencia: str, pipeline_nli) -> float:
    """Calcula o score textual de entailment entre referencia e resposta."""
    if not resposta or not referencia or pipeline_nli is None:
        return 0.0
    try:
        resultado = pipeline_nli(f"{referencia} </s> {resposta[:1000]}")
        if isinstance(resultado, list):
            resultado = resultado[0]
        rotulo = str(resultado["label"]).upper()
        score = float(resultado["score"])
    except Exception:
        return 0.0

    if rotulo == "ENTAILMENT":
        return score
    if rotulo == "NEUTRAL":
        return 0.5 * score
    return 1.0 - score


def calcular_sbert(texto_1: str, texto_2: str, backend_embeddings) -> float:
    """Calcula a similaridade SBERTScore entre dois textos."""
    # A comparacao por sentencas reduz o efeito de respostas longas com apenas aderencia parcial.
    sentencas_resposta = segmentar_sentencas(texto_1) or [texto_1]
    sentencas_referencia = segmentar_sentencas(texto_2) or [texto_2]

    embeddings = backend_embeddings.obter_embeddings(sentencas_resposta + sentencas_referencia)
    matriz_resposta = [embeddings[sentenca] for sentenca in sentencas_resposta if sentenca in embeddings]
    matriz_referencia = [embeddings[sentenca] for sentenca in sentencas_referencia if sentenca in embeddings]

    if not matriz_resposta or not matriz_referencia:
        return 0.0

    vetor_resposta = np.array(matriz_resposta)
    vetor_referencia = np.array(matriz_referencia)

    vetor_resposta = vetor_resposta / np.linalg.norm(vetor_resposta, axis=1, keepdims=True)
    vetor_referencia = vetor_referencia / np.linalg.norm(vetor_referencia, axis=1, keepdims=True)
    matriz_similaridade = np.dot(vetor_resposta, vetor_referencia.T)

    # A logica segue a ideia de SBERTScore, equilibrando cobertura da resposta e cobertura do gabarito.
    precisao = matriz_similaridade.max(axis=1).mean()
    revocacao = matriz_similaridade.max(axis=0).mean()
    if precisao + revocacao == 0:
        return 0.0
    return float((2 * precisao * revocacao) / (precisao + revocacao))


def avaliar_semantico(resposta: str, referencia: str, backend_embeddings, pipeline_nli) -> float:
    """Combina similaridade semantica com NLI conforme o estudo de metricas."""
    score_sbert = calcular_sbert(resposta, referencia, backend_embeddings)
    score_nli = calcular_nli(resposta, referencia, pipeline_nli)
    # Contradicao forte derruba a nota semantica, ainda que exista alguma proximidade lexical.
    if score_nli < 0.2:
        return 0.0
    return (0.7 * score_sbert) + (0.3 * score_nli)


def carregar_criterios_correcao(criterios_brutos, gabarito_completo: str, pontuacao_total: float) -> list[CriterioCorrecao]:
    """Interpreta os criterios estruturados do gabarito."""
    if isinstance(criterios_brutos, str):
        criterios_brutos = carregar_json_seguro(criterios_brutos, [])
    if not isinstance(criterios_brutos, list):
        criterios_brutos = []

    criterios: list[CriterioCorrecao] = []
    for criterio in criterios_brutos:
        componentes = []
        for componente in criterio.get("componentes", []):
            # Cada componente ja chega com o tipo de verificacao e o peso definido no espelho.
            componentes.append(
                ComponenteCriterio(
                    tipo=normalizar_texto(componente.get("tipo")).lower(),
                    peso=float(componente.get("peso", 0.0) or 0.0),
                    referencia=normalizar_texto(componente.get("referencia", "")),
                    termos=[normalizar_texto(termo) for termo in componente.get("termos", []) if normalizar_texto(termo)],
                )
            )
        criterios.append(
            CriterioCorrecao(
                identificador=str(criterio.get("id", "")),
                secao=normalizar_texto(criterio.get("secao", "")) or "secao_nao_informada",
                peso_total=float(criterio.get("peso_total", 0.0) or 0.0),
                componentes=componentes,
            )
        )

    if criterios:
        return criterios

    # O fallback preserva a avaliacao quando a curadoria nao trouxer criterios detalhados.
    if gabarito_completo and pontuacao_total > 0:
        return [
            CriterioCorrecao(
                identificador="fallback",
                secao="avaliacao_global",
                peso_total=pontuacao_total,
                componentes=[
                    ComponenteCriterio(
                        tipo="semantico",
                        peso=pontuacao_total,
                        referencia=gabarito_completo,
                    )
                ],
            )
        ]

    return []


def avaliar_discursivas_estruturadas(
    df_respostas_discursivas: pd.DataFrame,
    backend_embeddings,
    pipeline_nli,
    caminhos_saida: CaminhosSaida,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Avalia cada resposta discursiva segundo o novo gabarito estruturado."""
    linhas_questao: list[dict] = []
    linhas_componente: list[dict] = []

    for _, linha in tqdm(df_respostas_discursivas.iterrows(), total=len(df_respostas_discursivas), desc="Avaliando discursivas"):
        pontuacao_total = float(linha.get("pontuacao_total", 0.0) or 0.0)
        criterios = carregar_criterios_correcao(
            linha.get("criterios_correcao", []),
            normalizar_texto(linha.get("gabarito_completo", "")),
            pontuacao_total,
        )

        nota_total = 0.0
        resposta = str(linha.get("resposta", ""))

        for criterio in criterios:
            nota_criterio = 0.0
            for indice, componente in enumerate(criterio.componentes, start=1):
                # Componentes semanticos e legislativos sao avaliados separadamente e depois somados.
                if componente.tipo == "semantico":
                    score = avaliar_semantico(resposta, componente.referencia, backend_embeddings, pipeline_nli)
                    nota = score * componente.peso
                    referencia = componente.referencia
                    termos_esperados = ""
                elif componente.tipo == "legislacao":
                    houve_match = verificar_legislacao(resposta, componente.termos)
                    score = 1.0 if houve_match else 0.0
                    nota = componente.peso if houve_match else 0.0
                    referencia = ""
                    termos_esperados = " | ".join(componente.termos)
                else:
                    score = 0.0
                    nota = 0.0
                    referencia = ""
                    termos_esperados = ""

                nota_criterio += nota
                # O detalhamento por componente alimenta a auditoria do espelho e os relatorios finais.
                linhas_componente.append(
                    {
                        "question_id": linha["question_id"],
                        "modelo": linha["modelo"],
                        "criterio_id": criterio.identificador,
                        "secao": criterio.secao,
                        "indice_componente": indice,
                        "tipo_componente": componente.tipo,
                        "peso_componente": componente.peso,
                        "peso_total_criterio": criterio.peso_total,
                        "score": round(score, 4),
                        "nota_obtida": round(nota, 4),
                        "referencia": referencia,
                        "termos_esperados": termos_esperados,
                    }
                )

            nota_total += nota_criterio

        # O aproveitamento normaliza a nota da questao pela pontuacao maxima prevista no espelho.
        aproveitamento = nota_total / pontuacao_total if pontuacao_total > 0 else 0.0
        linhas_questao.append(
            {
                "question_id": linha["question_id"],
                "modelo": linha["modelo"],
                "tipo_questao": linha.get("tipo_questao", ""),
                "nivel_dificuldade": linha.get("nivel_dificuldade", ""),
                "disciplina": linha.get("disciplina", ""),
                "tema": linha.get("tema", ""),
                "pontuacao_total": pontuacao_total,
                "nota_estimada": round(nota_total, 4),
                "aproveitamento": round(aproveitamento, 4),
            }
        )

    df_detalhe_questao = pd.DataFrame(linhas_questao)
    df_composicao = pd.DataFrame(linhas_componente)

    # O benchmark agrega o desempenho discursivo por modelo para comparacao direta entre candidatos.
    benchmark = (
        df_detalhe_questao.groupby("modelo", dropna=False)
        .agg(
            nota_total=("nota_estimada", "sum"),
            pontuacao_total_disc=("pontuacao_total", "sum"),
            aproveitamento_medio=("aproveitamento", "mean"),
            total_questoes=("question_id", "size"),
        )
        .reset_index()
    )
    benchmark["aproveitamento_geral"] = (
        benchmark["nota_total"] / benchmark["pontuacao_total_disc"].replace(0, np.nan)
    ).fillna(0.0)
    benchmark = benchmark.sort_values(
        ["aproveitamento_geral", "nota_total", "modelo"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    salvar_csv(df_detalhe_questao, caminhos_saida.avaliacao_discursivas_csv)
    salvar_csv(benchmark, caminhos_saida.benchmark_discursivas_csv)
    salvar_csv(df_composicao, caminhos_saida.composicao_discursivas_csv)
    return df_detalhe_questao, benchmark, df_composicao
