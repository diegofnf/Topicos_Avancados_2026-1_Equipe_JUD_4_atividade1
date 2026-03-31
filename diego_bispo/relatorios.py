"""Relatorios consolidados do benchmark."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import Image, Markdown, display
from matplotlib.backends.backend_pdf import PdfPages

from .configuracoes import CaminhosSaida
from .utilitarios import arredondar_numericos, salvar_csv


def vencedores_por_grupo(df: pd.DataFrame, coluna_grupo: str, coluna_score: str, tipo_avaliacao: str, criterio: str) -> pd.DataFrame:
    """Seleciona o melhor modelo em cada agrupamento."""
    base = df.copy().sort_values([coluna_grupo, coluna_score, "modelo"], ascending=[True, False, True])
    vencedores = base.drop_duplicates(subset=[coluna_grupo]).copy()
    vencedores.insert(0, "tipo_avaliacao", tipo_avaliacao)
    vencedores.insert(1, "criterio", criterio)
    return vencedores


def renderizar_pagina_tabela(pdf: PdfPages, titulo: str, df: pd.DataFrame, linhas_por_pagina: int = 18) -> None:
    """Renderiza uma tabela paginada em PDF."""
    if df.empty:
        figura, eixo = plt.subplots(figsize=(11.69, 8.27))
        eixo.axis("off")
        eixo.set_title(titulo, fontsize=14, fontweight="bold", loc="left")
        eixo.text(0.02, 0.9, "Sem dados.", fontsize=11, transform=eixo.transAxes)
        pdf.savefig(figura, bbox_inches="tight")
        plt.close(figura)
        return

    total = len(df)
    for inicio in range(0, total, linhas_por_pagina):
        recorte = arredondar_numericos(df.iloc[inicio:inicio + linhas_por_pagina], 2)
        altura = max(3.8, 1.4 + 0.34 * (len(recorte) + 2))
        figura, eixo = plt.subplots(figsize=(11.69, min(altura, 8.27)))
        eixo.axis("off")
        sufixo = "" if total <= linhas_por_pagina else f" (linhas {inicio + 1}-{min(inicio + linhas_por_pagina, total)})"
        eixo.set_title(titulo + sufixo, fontsize=13, fontweight="bold", loc="left")
        tabela = eixo.table(cellText=recorte.astype(str).values, colLabels=list(recorte.columns), loc="center", cellLoc="center")
        tabela.auto_set_font_size(False)
        tabela.set_fontsize(8.5)
        tabela.scale(1, 1.3)
        for (linha, _), celula in tabela.get_celld().items():
            if linha == 0:
                celula.set_text_props(weight="bold")
                celula.set_facecolor("#D9EAF7")
        plt.tight_layout()
        pdf.savefig(figura, bbox_inches="tight")
        plt.close(figura)


def gerar_relatorios_consolidados(
    df_objetivas_detalhe: pd.DataFrame,
    benchmark_objetivas: pd.DataFrame,
    df_discursivas_detalhe: pd.DataFrame,
    benchmark_discursivas: pd.DataFrame,
    df_composicao_discursiva: pd.DataFrame,
    caminhos_saida: CaminhosSaida,
) -> pd.DataFrame:
    """Gera os artefatos executivos e analiticos da execucao."""
    benchmark_consolidado = benchmark_objetivas.merge(benchmark_discursivas, on="modelo", how="outer").fillna(0.0)
    benchmark_consolidado["score_balanceado"] = (
        benchmark_consolidado["acuracia_objetivas"] + benchmark_consolidado["aproveitamento_geral"]
    ) / 2
    benchmark_consolidado["score_total_ponderado"] = (
        benchmark_consolidado["acertos_objetivas"] + benchmark_consolidado["nota_total"]
    ) / (
        benchmark_consolidado["total_objetivas"] + benchmark_consolidado["pontuacao_total_disc"]
    ).replace(0, float("nan"))
    benchmark_consolidado = benchmark_consolidado.fillna(0.0).sort_values(
        ["score_balanceado", "score_total_ponderado", "modelo"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    salvar_csv(benchmark_consolidado, caminhos_saida.benchmark_consolidado_csv)

    notas_discursivas = (
        df_discursivas_detalhe.pivot_table(index="question_id", columns="modelo", values="nota_estimada", aggfunc="first")
        .reset_index()
        .merge(
            df_discursivas_detalhe[["question_id", "pontuacao_total"]].drop_duplicates(),
            on="question_id",
            how="left",
        )
    )
    colunas_modelos = [coluna for coluna in notas_discursivas.columns if coluna not in {"question_id", "pontuacao_total"}]
    notas_discursivas = notas_discursivas[["question_id", "pontuacao_total", *colunas_modelos]].rename(
        columns={"pontuacao_total": "nota_questao"}
    )
    linha_media = {"question_id": "MEDIA", "nota_questao": notas_discursivas["nota_questao"].mean()}
    for coluna in colunas_modelos:
        linha_media[coluna] = notas_discursivas[coluna].mean()
    notas_discursivas = pd.concat([notas_discursivas, pd.DataFrame([linha_media])], ignore_index=True)
    salvar_csv(notas_discursivas, caminhos_saida.discursivas_notas_por_questao_csv)

    objetivas_resumo = benchmark_objetivas[["modelo", "acertos_objetivas", "total_objetivas", "acuracia_objetivas"]].copy()
    objetivas_resumo.insert(0, "total_questoes_objetivas", int(df_objetivas_detalhe["question_id"].nunique()))
    salvar_csv(objetivas_resumo, caminhos_saida.objetivas_resumo_csv)

    objetivas_por_dificuldade = (
        df_objetivas_detalhe.groupby(["nivel_dificuldade", "modelo"], dropna=False)
        .agg(score=("correto", "mean"), acertos=("correto", "sum"), total=("correto", "size"))
        .reset_index()
    )
    discursivas_por_dificuldade = (
        df_discursivas_detalhe.groupby(["nivel_dificuldade", "modelo"], dropna=False)
        .agg(score_num=("nota_estimada", "sum"), score_den=("pontuacao_total", "sum"))
        .reset_index()
    )
    discursivas_por_dificuldade["score"] = discursivas_por_dificuldade["score_num"] / discursivas_por_dificuldade["score_den"].replace(0, float("nan"))
    discursivas_por_dificuldade = discursivas_por_dificuldade.fillna(0.0)

    vencedores_dificuldade = pd.concat(
        [
            vencedores_por_grupo(objetivas_por_dificuldade, "nivel_dificuldade", "score", "objetiva", "nivel_dificuldade"),
            vencedores_por_grupo(discursivas_por_dificuldade, "nivel_dificuldade", "score", "discursiva", "nivel_dificuldade"),
        ],
        ignore_index=True,
    )
    salvar_csv(vencedores_dificuldade, caminhos_saida.vencedores_dificuldade_csv)

    objetivas_por_disciplina = (
        df_objetivas_detalhe.groupby(["disciplina", "modelo"], dropna=False)
        .agg(score=("correto", "mean"), acertos=("correto", "sum"), total=("correto", "size"))
        .reset_index()
    )
    discursivas_por_disciplina = (
        df_discursivas_detalhe.groupby(["disciplina", "modelo"], dropna=False)
        .agg(score_num=("nota_estimada", "sum"), score_den=("pontuacao_total", "sum"))
        .reset_index()
    )
    discursivas_por_disciplina["score"] = discursivas_por_disciplina["score_num"] / discursivas_por_disciplina["score_den"].replace(0, float("nan"))
    discursivas_por_disciplina = discursivas_por_disciplina.fillna(0.0)

    vencedores_disciplina = pd.concat(
        [
            vencedores_por_grupo(objetivas_por_disciplina, "disciplina", "score", "objetiva", "disciplina"),
            vencedores_por_grupo(discursivas_por_disciplina, "disciplina", "score", "discursiva", "disciplina"),
        ],
        ignore_index=True,
    )
    salvar_csv(vencedores_disciplina, caminhos_saida.vencedores_disciplina_csv)

    melhor_objetiva = benchmark_objetivas.iloc[0]
    melhor_discursiva = benchmark_discursivas.iloc[0]
    melhor_geral = benchmark_consolidado.iloc[0]
    display(
        Markdown(
            f"### Painel executivo\n"
            f"- **Melhor nas objetivas:** `{melhor_objetiva['modelo']}` com **{melhor_objetiva['acuracia_objetivas']:.2%}** de acuracia.\n"
            f"- **Melhor nas discursivas:** `{melhor_discursiva['modelo']}` com **{melhor_discursiva['aproveitamento_geral']:.2%}** de aproveitamento e **{melhor_discursiva['nota_total']:.2f} pontos**.\n"
            f"- **Melhor no consolidado:** `{melhor_geral['modelo']}` com **{melhor_geral['score_balanceado']:.2%}**."
        )
    )

    display(Markdown("### Secao 1 - Notas das discursivas por questao"))
    display(arredondar_numericos(notas_discursivas))
    display(Markdown("### Secao 2 - Composicao dos criterios discursivos"))
    display(arredondar_numericos(df_composicao_discursiva.head(20)))
    display(Markdown("### Secao 3 - Avaliacao das objetivas"))
    display(arredondar_numericos(objetivas_resumo))
    display(Markdown("### Secao 4 - Analise por dificuldade"))
    display(arredondar_numericos(vencedores_dificuldade))
    display(Markdown("### Secao 5 - Analise por especialidade"))
    display(arredondar_numericos(vencedores_disciplina))

    colunas_heatmap = ["acuracia_objetivas", "aproveitamento_geral", "score_balanceado", "score_total_ponderado"]
    heatmap_df = benchmark_consolidado.set_index("modelo")[colunas_heatmap]
    plt.figure(figsize=(10, 4))
    sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="YlOrRd", linewidths=0.5, cbar=True)
    plt.title("Painel consolidado de desempenho por modelo")
    plt.xlabel("Metrica")
    plt.ylabel("Modelo")
    plt.tight_layout()
    plt.savefig(caminhos_saida.heatmap_consolidado_png, dpi=200, bbox_inches="tight")
    plt.show()
    display(Image(filename=str(caminhos_saida.heatmap_consolidado_png)))

    with PdfPages(caminhos_saida.relatorio_executivo_pdf) as pdf:
        resumo_pdf = arredondar_numericos(
            benchmark_consolidado[["modelo", "acuracia_objetivas", "aproveitamento_geral", "score_balanceado", "score_total_ponderado"]]
        )
        renderizar_pagina_tabela(pdf, "Resumo executivo", resumo_pdf, linhas_por_pagina=12)
        renderizar_pagina_tabela(pdf, "Secao 1 - Notas discursivas", notas_discursivas, linhas_por_pagina=18)
        renderizar_pagina_tabela(pdf, "Secao 2 - Composicao dos criterios", df_composicao_discursiva, linhas_por_pagina=16)
        renderizar_pagina_tabela(pdf, "Secao 3 - Avaliacao objetivas", objetivas_resumo, linhas_por_pagina=10)
        renderizar_pagina_tabela(pdf, "Secao 4 - Por dificuldade", vencedores_dificuldade, linhas_por_pagina=16)
        renderizar_pagina_tabela(pdf, "Secao 5 - Por especialidade", vencedores_disciplina, linhas_por_pagina=16)

    print(f"Relatorio PDF salvo em: {caminhos_saida.relatorio_executivo_pdf}")
    return benchmark_consolidado
