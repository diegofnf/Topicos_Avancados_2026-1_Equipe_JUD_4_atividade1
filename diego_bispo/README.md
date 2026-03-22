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
- Na análise quantitativa das questões objetivas, optou-se por utilizar apenas **acurácia**. Essa escolha foi feita porque, neste contexto, o objetivo principal é medir quantas questões o modelo acerta no total, e não investigar padrões de confusão entre letras específicas. Métricas como `F1 macro` e matriz de confusão foram consideradas excessivas para a proposta da atividade, já que as alternativas `A/B/C/D` não possuem valor semântico próprio fora do gabarito da questão.

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
| Resultados | — | avaliação + respostas | accuracy + benchmark quantitativo/qualitativo |

### Avaliação das discursivas

O juiz agora usa o campo `values` da questão para limitar a pontuação.

Metodologicamente, esta etapa segue a lógica do artigo de referência do `oab-bench`, que adota avaliação individual de respostas com um `LLM judge` em modo `single-answer` e `reference-guided grading`. Em vez de colocar as três respostas dos modelos no mesmo prompt, o pipeline avalia cada resposta discursiva separadamente, atribuindo notas e justificativas para argumentação, precisão jurídica e coesão legal. A comparação entre os três modelos é feita posteriormente, de forma agregada, a partir dessas avaliações individuais e de métricas adicionais de concordância semântica, como embeddings e `bertscore_concordancia`.

Para preservar a qualidade das respostas discursivas dos modelos candidatos, o pipeline deixou de exigir JSON nessa etapa. Na prática, os candidatos agora respondem em texto livre, em formato aberto, e o sistema captura diretamente a resposta final gerada. A exigência de JSON foi mantida apenas nas etapas em que ela se mostrou estável e útil, como nas respostas objetivas, na curadoria e na avaliação do juiz.

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

### BERTScore de concordância entre modelos

O benchmark qualitativo também calcula `bertscore_concordancia`, comparando respostas de modelos diferentes para a mesma questão em lote.

- Essa métrica mede concordância semântica entre modelos.
- Ela não deve ser interpretada como nota de qualidade jurídica absoluta.
- O backbone usado nessa etapa é o `RoBERTaLexPT-base`, um modelo jurídico em português cuja escolha é motivada pelo artigo de Garcia et al., que mostra ganhos consistentes de adaptação ao domínio jurídico lusófono no benchmark `PortuLex`.
- Já a etapa de embeddings para matriz de similaridade permanece com um modelo ajustado para `sentence similarity`, por ser mais apropriado para comparação vetorial direta entre respostas completas.

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
