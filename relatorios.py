"""Relatorios consolidados do benchmark."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import Image, Markdown, display
from matplotlib.backends.backend_pdf import PdfPages

from configuracoes import CaminhosSaida
from utilitarios import arredondar_numericos, salvar_csv


def normalizar_nivel_dificuldade(valor: str) -> str:
    """Padroniza os rotulos de dificuldade para exibicao."""
    texto = str(valor or "").strip().lower()
    mapa = {
        "media": "medio",
        "média": "medio",
        "medio": "medio",
        "médio": "medio",
        "facil": "facil",
        "fácil": "facil",
        "dificil": "dificil",
        "difícil": "dificil",
        "muito_dificil": "muito_dificil",
        "muito difícil": "muito_dificil",
        "muito dificil": "muito_dificil",
    }
    return mapa.get(texto, texto or "nao_informado")


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


def resumir_totais_por_grupo(
    df_base: pd.DataFrame,
    coluna_grupo: str,
    coluna_pontos_modelo: str,
    coluna_pontos_questao: str,
) -> pd.DataFrame:
    """Resume total de questoes, total da prova e total obtido por modelo em um agrupamento."""
    totais_questao = (
        df_base[["question_id", coluna_grupo, coluna_pontos_questao]]
        .drop_duplicates(subset=["question_id"])
        .groupby(coluna_grupo, dropna=False)
        .agg(total_questoes=("question_id", "nunique"), total_pontuacao_questoes=(coluna_pontos_questao, "sum"))
        .reset_index()
    )
    totais_modelo = (
        df_base.groupby([coluna_grupo, "modelo"], dropna=False)
        .agg(total_pontuacao_modelo=(coluna_pontos_modelo, "sum"))
        .reset_index()
        .pivot(index=coluna_grupo, columns="modelo", values="total_pontuacao_modelo")
        .fillna(0.0)
        .reset_index()
    )
    return totais_questao.merge(totais_modelo, on=coluna_grupo, how="left").fillna(0.0)


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
    benchmark_consolidado["media_discursivas"] = benchmark_consolidado["aproveitamento_geral"]
    benchmark_consolidado["media_geral"] = (
        benchmark_consolidado["acuracia_objetivas"] + benchmark_consolidado["media_discursivas"]
    ) / 2
    benchmark_consolidado = benchmark_consolidado.fillna(0.0).sort_values(
        ["media_geral", "acuracia_objetivas", "media_discursivas", "modelo"],
        ascending=[False, False, False, True],
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

    df_obj_dificuldade = df_objetivas_detalhe.copy()
    df_obj_dificuldade["valor_questao"] = 1.0
    df_obj_dificuldade["nivel_dificuldade"] = df_obj_dificuldade["nivel_dificuldade"].map(normalizar_nivel_dificuldade)
    resumo_objetivas_dificuldade = resumir_totais_por_grupo(
        df_obj_dificuldade,
        coluna_grupo="nivel_dificuldade",
        coluna_pontos_modelo="correto",
        coluna_pontos_questao="valor_questao",
    )
    resumo_objetivas_dificuldade.insert(0, "tipo_avaliacao", "objetiva")

    df_disc_dificuldade = df_discursivas_detalhe.copy()
    df_disc_dificuldade["nivel_dificuldade"] = df_disc_dificuldade["nivel_dificuldade"].map(normalizar_nivel_dificuldade)
    resumo_discursivas_dificuldade = resumir_totais_por_grupo(
        df_disc_dificuldade,
        coluna_grupo="nivel_dificuldade",
        coluna_pontos_modelo="nota_estimada",
        coluna_pontos_questao="pontuacao_total",
    )
    resumo_discursivas_dificuldade.insert(0, "tipo_avaliacao", "discursiva")

    resumo_dificuldade = pd.concat(
        [resumo_objetivas_dificuldade, resumo_discursivas_dificuldade],
        ignore_index=True,
    )
    ordem_dificuldade = {"facil": 0, "medio": 1, "dificil": 2, "muito_dificil": 3, "nao_informado": 4}
    resumo_dificuldade["_ordem"] = resumo_dificuldade["nivel_dificuldade"].map(ordem_dificuldade).fillna(99)
    resumo_dificuldade = resumo_dificuldade.sort_values(["tipo_avaliacao", "_ordem", "nivel_dificuldade"]).drop(columns="_ordem").reset_index(drop=True)
    salvar_csv(resumo_dificuldade, caminhos_saida.vencedores_dificuldade_csv)

    df_obj_disciplina = df_objetivas_detalhe.copy()
    df_obj_disciplina["valor_questao"] = 1.0
    resumo_objetivas_disciplina = resumir_totais_por_grupo(
        df_obj_disciplina,
        coluna_grupo="disciplina",
        coluna_pontos_modelo="correto",
        coluna_pontos_questao="valor_questao",
    )
    resumo_objetivas_disciplina.insert(0, "tipo_avaliacao", "objetiva")

    df_disc_disciplina = df_discursivas_detalhe.copy()
    resumo_discursivas_disciplina = resumir_totais_por_grupo(
        df_disc_disciplina,
        coluna_grupo="disciplina",
        coluna_pontos_modelo="nota_estimada",
        coluna_pontos_questao="pontuacao_total",
    )
    resumo_discursivas_disciplina.insert(0, "tipo_avaliacao", "discursiva")

    resumo_disciplina = pd.concat(
        [resumo_objetivas_disciplina, resumo_discursivas_disciplina],
        ignore_index=True,
    )
    resumo_disciplina = resumo_disciplina.sort_values(["tipo_avaliacao", "disciplina"]).reset_index(drop=True)
    salvar_csv(resumo_disciplina, caminhos_saida.vencedores_disciplina_csv)

    composicao_resumida = (
        df_composicao_discursiva.groupby(["question_id", "criterio_id", "secao", "tipo_componente", "modelo"], dropna=False)
        .agg(
            nota_criterio=("peso_total_criterio", "max"),
            nota_modelo=("nota_obtida", "sum"),
        )
        .reset_index()
    )
    primeiras_questoes = list(df_discursivas_detalhe["question_id"].drop_duplicates().head(2))
    composicao_resumida = composicao_resumida[composicao_resumida["question_id"].isin(primeiras_questoes)].reset_index(drop=True)
    composicao_resumida = (
        composicao_resumida.pivot_table(
            index=["question_id", "criterio_id", "secao", "tipo_componente", "nota_criterio"],
            columns="modelo",
            values="nota_modelo",
            aggfunc="first",
        )
        .reset_index()
        .fillna(0.0)
    )

    melhor_objetiva = benchmark_objetivas.iloc[0]
    melhor_discursiva = benchmark_discursivas.iloc[0]
    melhor_geral = benchmark_consolidado.iloc[0]
    display(
        Markdown(
            f"### Painel executivo\n"
            f"- **Melhor nas objetivas:** `{melhor_objetiva['modelo']}` com **{melhor_objetiva['acuracia_objetivas']:.2%}** de acuracia.\n"
            f"- **Melhor media das discursivas:** `{melhor_discursiva['modelo']}` com **{melhor_discursiva['aproveitamento_geral']:.2%}**.\n"
            f"- **Melhor media geral:** `{melhor_geral['modelo']}` com **{melhor_geral['media_geral']:.2%}**."
        )
    )

    display(Markdown("### Secao 1 - Notas das discursivas por questao"))
    display(arredondar_numericos(notas_discursivas))
    display(Markdown("### Secao 2 - Composicao dos criterios discursivos"))
    display(arredondar_numericos(composicao_resumida))
    display(Markdown("### Secao 3 - Avaliacao das objetivas"))
    display(arredondar_numericos(objetivas_resumo))
    display(Markdown("### Secao 4 - Analise por dificuldade"))
    display(arredondar_numericos(resumo_dificuldade))
    display(Markdown("### Secao 5 - Analise por especialidade"))
    display(arredondar_numericos(resumo_disciplina))

    colunas_heatmap = ["acuracia_objetivas", "media_discursivas", "media_geral"]
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
            benchmark_consolidado[["modelo", "acuracia_objetivas", "media_discursivas", "media_geral"]]
        )
        renderizar_pagina_tabela(pdf, "Resumo executivo", resumo_pdf, linhas_por_pagina=12)
        renderizar_pagina_tabela(pdf, "Secao 1 - Notas discursivas", notas_discursivas, linhas_por_pagina=18)
        renderizar_pagina_tabela(pdf, "Secao 2 - Composicao dos criterios", composicao_resumida, linhas_por_pagina=16)
        renderizar_pagina_tabela(pdf, "Secao 3 - Avaliacao objetivas", objetivas_resumo, linhas_por_pagina=10)
        renderizar_pagina_tabela(pdf, "Secao 4 - Por dificuldade", resumo_dificuldade, linhas_por_pagina=16)
        renderizar_pagina_tabela(pdf, "Secao 5 - Por especialidade", resumo_disciplina, linhas_por_pagina=16)

    print(f"Relatorio PDF salvo em: {caminhos_saida.relatorio_executivo_pdf}")
    return benchmark_consolidado
