# OAB Pipeline — Benchmark de Modelos de Linguagem

Avaliação de modelos open-source em questões discursivas e objetivas da OAB, utilizando uma arquitetura de três agentes: **candidato**, **curador** e **juiz**.

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
| Avaliação | Juiz | questões + respostas + curadoria | avaliacao_discursivas.csv |
| Resultados | — | avaliação + respostas | accuracy + benchmark |

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
