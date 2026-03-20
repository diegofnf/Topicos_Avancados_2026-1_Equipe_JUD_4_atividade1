# OAB Pipeline — Benchmark de Modelos de Linguagem

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/diegofnf/Topicos_Avancados_2026-1_Equipe_JUD_4_atividade1/blob/main/diego_bispo/notebook.ipynb)

Avaliação de modelos open-source em questões discursivas e objetivas da OAB, utilizando uma arquitetura de três agentes: **candidato**, **curador** e **juiz**.

---

## Regras dos Datasets

### Discursivas (J1)

- A **área de especialidade** é definida diretamente pelo campo `category` do dataset.
- O modelo curador **não** infere mais a área de especialidade das questões discursivas.
- O campo `values` é salvo junto com cada questão e usado na etapa de avaliação do juiz.

### Objetivas (J2)

- A **área de especialidade** é definida diretamente pelo campo `question_type` do dataset.

---

## Arquitetura

```
config.py       → paths, modelos, constantes
prompts.py      → prompts dos três agentes
model_utils.py  → load, unload, geração de texto
data_utils.py   → IO de CSV, timestamp, parsing JSON
pipeline.py     → lógica de resposta, curadoria e avaliação
notebook.ipynb  → orquestração no Colab
```

---

## Pipeline

| Etapa | Agente | Entrada | Saída |
|---|---|---|---|
| Respostas discursivas | Candidato | questões | respostas_discursivas.csv |
| Respostas objetivas | Candidato | questões + alternativas | respostas_objetivas.csv |
| Curadoria | Curador | questões | curadoria_discursivas/objetivas.csv |
| Avaliação | Juiz | questões + respostas + curadoria + values | avaliacao_discursivas.csv |
| Similaridade discursiva | Embeddings | respostas_discursivas.csv | similaridade_discursivas.csv + heatmap |
| Resultados | — | avaliação + respostas | accuracy + benchmark |

### Avaliação das discursivas

O juiz agora usa o campo `values` da questão para limitar a pontuação.

- Quando `values = [0.65, 0.6]`, a correção é dividida em:
- `nota_fundamentacao_coerencia`: de `0` até `0.65`
- `nota_aderencia_completude`: de `0` até `0.6`
- `nota_total`: soma das duas notas parciais

Os critérios qualitativos usados pelo juiz são:

- argumentação
- precisão jurídica
- coesão legal

Quando a questão tiver apenas um valor, como em uma peça com `values = [5]`, esse valor passa a ser o teto da `nota_total`.

### Similaridade semântica entre modelos

As respostas discursivas também podem ser avaliadas com embeddings, comparando apenas respostas da mesma questão.

- O notebook gera uma tabela `similaridade_discursivas.csv` com a similaridade cosseno entre pares de modelos.
- Também gera um heatmap `heatmap_similaridade_discursivas.png` com a similaridade média entre os modelos.
- Essa análise mede proximidade semântica entre respostas, não correção jurídica.

---

## Limitações metodológicas

- A avaliação das discursivas é feita por um LLM juiz, e não por corretores humanos da OAB.
- O dataset de discursivas não fornece `commented answer`; por isso, a avaliação não compara a resposta do modelo com um espelho oficial comentado, e se baseia no enunciado, na pontuação da questão e na curadoria produzida pelo pipeline.
- O pipeline depende de saídas em JSON válido; quando um modelo quebra o formato, pode haver perda de resposta ou necessidade de fallback analítico.

---

## Como executar no Colab

1. Abra o `notebook.ipynb` no Google Colab
2. Configure o `HF_TOKEN` nas Secrets do Colab (`🔑 Secrets`)
3. Ative a GPU: `Ambiente de execução > Alterar tipo de hardware > T4`
4. Execute as células em ordem

---

## Datasets

- **Discursivas:** [`maritaca-ai/oab-bench`](https://huggingface.co/datasets/maritaca-ai/oab-bench)
- **Objetivas:** [`eduagarcia/oab_exams`](https://huggingface.co/datasets/eduagarcia/oab_exams)

---

## Modelos avaliados

| Nome | Modelo | Tamanho (4bit) |
|---|---|---|
| gemma2 | google/gemma-2-2b-it | ~5.25 GB |
| llama323b | meta-llama/Llama-3.2-3B-Instruct | ~12.9 GB |
| llama321b | meta-llama/Llama-3.2-1B-Instruct | ~4.95 GB |

**Curador/Juiz:** Qwen/Qwen3-4B-Instruct-2507
