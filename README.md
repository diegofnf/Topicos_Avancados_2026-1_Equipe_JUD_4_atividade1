[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/diegofnf/Topicos_Avancados_2026-1_Equipe_JUD_4_atividade1/blob/main/diego_bispo.ipynb)

# Benchmark OAB com Avaliacao Estruturada de Questoes Objetivas e Discursivas

## 1. Visao geral

Este projeto implementa um prototipo autonomo para benchmark de modelos de linguagem em questoes da prova da OAB. O fluxo parte de um unico arquivo de entrada, `curadorias.csv`, e executa as seguintes etapas:

1. preparar os dados curados;
2. carregar os modelos de geracao e de avaliacao;
3. gerar respostas para questoes objetivas e discursivas;
4. avaliar o desempenho quantitativo nas objetivas;
5. avaliar o desempenho discursivo com base no novo formato estruturado do gabarito;
6. consolidar os resultados em relatorios, tabelas e visualizacoes.

O projeto foi refatorado para separar responsabilidades em modulos Python e deixar o notebook principal apenas como camada de orquestracao.

## 2. Objetivo do benchmark

O objetivo central e comparar diferentes modelos de linguagem em tarefas juridicas da OAB, considerando dois eixos complementares:

- desempenho quantitativo nas questoes objetivas, por meio de acuracia;
- desempenho discursivo estruturado, por meio de aderencia semantica e atendimento a referencias legais previstas no espelho.

Essa abordagem permite avaliar nao apenas se o modelo acerta a alternativa correta, mas tambem se a resposta discursiva atende aos criterios juridicos relevantes do gabarito.

## 3. Base de dados

O projeto utiliza o arquivo `curadorias.csv` na raiz do repositorio como fonte unica de dados. O arquivo contem:

- identificadores das questoes;
- tipo da questao;
- prompt de sistema;
- enunciado;
- perguntas complementares das discursivas;
- alternativas das objetivas;
- gabarito;
- metadados de dificuldade, disciplina, tema e legislacao.

### 3.1 Estrutura do gabarito

As questoes objetivas utilizam um gabarito simples com a alternativa correta.

As questoes discursivas seguem a nova estrutura JSON no campo `Gabarito`, com destaque para:

- `gabarito_completo`: texto corrido com a resposta esperada;
- `criterios`: lista estruturada de criterios de correcao;
- `componentes`: subdivisoes internas de cada criterio.

Exemplo conceitual:

```json
{
  "gabarito_completo": "Resposta discursiva esperada...",
  "criterios": [
    {
      "id": "A",
      "secao": "Fundamentacao",
      "peso_total": 0.6,
      "componentes": [
        {
          "tipo": "semantico",
          "referencia": "Tese juridica esperada",
          "peso": 0.5
        },
        {
          "tipo": "legislacao",
          "termos": ["art 71", "sumula 3"],
          "peso": 0.1
        }
      ]
    }
  ]
}
```

Essa estrutura substitui a logica antiga baseada em itens menos aderentes ao formato atual do espelho.

## 4. Arquitetura do projeto

O projeto foi modularizado no pacote `diego_bispo/`, com separacao de responsabilidades.

### 4.1 Estrutura de diretorios

```text
.
|-- README.md
|-- curadorias.csv
|-- diego_bispo.ipynb
|-- artefatos_estudo/
|   |-- Metricas_Escolhidas.ipynb
|   |-- Parser_Gabarito.ipynb
|   `-- Teste_Métricas_Qualitativas.ipynb
|-- diego_bispo/
|   |-- __init__.py
|   |-- configuracoes.py
|   |-- prompts.py
|   |-- utilitarios.py
|   |-- carregar_dados.py
|   |-- modelos.py
|   |-- geracao_respostas.py
|   |-- avaliacao_objetiva.py
|   |-- avaliacao_discursiva.py
|   |-- relatorios.py
|   `-- pipeline.py
`-- artefatos_gerados/
```

### 4.2 Responsabilidade dos modulos

- `configuracoes.py`: concentra paths, nomes de modelos, cache, limites de execucao e variaveis de ambiente.
- `prompts.py`: armazena os prompts utilizados para geracao das respostas.
- `utilitarios.py`: reune funcoes auxiliares, como normalizacao textual, parsing de JSON e persistencia de CSV.
- `carregar_dados.py`: le o `curadorias.csv`, separa objetivas e discursivas, interpreta o novo gabarito e salva os dados preparados.
- `modelos.py`: baixa e carrega os modelos de geracao, embeddings juridicos e NLI.
- `geracao_respostas.py`: executa a inferencia dos modelos candidatos.
- `avaliacao_objetiva.py`: calcula a acuracia das objetivas.
- `avaliacao_discursiva.py`: aplica a nova avaliacao estruturada das discursivas.
- `relatorios.py`: consolida resultados, gera heatmap, tabelas e PDF.
- `pipeline.py`: fornece funcoes de alto nivel para o notebook principal.

## 5. Notebook principal

O arquivo `diego_bispo.ipynb` e a interface principal de execucao. Ele ficou propositalmente mais enxuto e organiza o fluxo em sete blocos:

1. configuracao do ambiente;
2. preparacao dos dados;
3. download dos modelos;
4. geracao ou reaproveitamento das respostas;
5. avaliacao das objetivas;
6. avaliacao discursiva estruturada;
7. consolidado executivo.

## 6. Modelos utilizados

### 6.1 Modelos candidatos para geracao

Os modelos atualmente definidos para responder as questoes sao:

- `google/gemma-2-2b-it`
- `meta-llama/Llama-3.2-3B-Instruct`
- `meta-llama/Llama-3.2-1B-Instruct`

Esses modelos sao usados tanto para questoes objetivas quanto para questoes discursivas.

### 6.2 Modelo de embeddings juridicos

Para comparacao semantica das respostas discursivas, o projeto utiliza:

- `stjiris/bert-large-portuguese-cased-legal-mlm-sts-v1.0`

Esse modelo fornece embeddings juridicos em portugues e sustenta a avaliacao de similaridade semantica entre a resposta do candidato e a referencia esperada no gabarito.

### 6.3 Modelo de inferencia textual NLI

Para verificar entailment, neutralidade e contradicao semantica, o projeto utiliza:

- `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`

Esse modelo e aplicado na etapa discursiva para complementar a similaridade semantica com uma nocao de consistencia inferencial.

## 7. Metricas utilizadas

## 7.1 Metrica quantitativa das objetivas

Nas questoes objetivas, a metrica principal e a acuracia:

```text
acuracia = total de acertos / total de questoes objetivas
```

Tambem sao registrados:

- `acertos_objetivas`;
- `total_objetivas`;
- `acuracia_objetivas`.

## 7.2 Metrica discursiva estruturada

As discursivas deixaram de usar a analise qualitativa anterior e passaram a seguir a nova estrutura do gabarito, com criterios e componentes.

Cada criterio pode possuir componentes de dois tipos:

- `semantico`;
- `legislacao`.

### 7.2.1 Componente semantico

O componente semantico combina duas fontes:

- similaridade SBERT juridica;
- score de NLI.

Formula adotada no projeto:

```text
score_semantico = 0.7 * score_sbert + 0.3 * score_nli
```

Regra de penalizacao:

```text
se score_nli < 0.2, entao score_semantico = 0
```

Essa regra reduz a nota quando a resposta se mostra contraditoria em relacao ao criterio esperado.

### 7.2.2 Componente de legislacao

O componente legislativo verifica se a resposta menciona os termos legais esperados, como artigos, sumulas e referencias normativas.

No estado atual, a verificacao e binaria:

```text
score_legislacao = 1, se houver correspondencia
score_legislacao = 0, caso contrario
```

### 7.2.3 Nota de cada componente

Para cada componente:

```text
nota_componente = score_componente * peso_componente
```

### 7.2.4 Nota de cada criterio

A nota do criterio e a soma das notas de seus componentes:

```text
nota_criterio = soma(nota_componente)
```

### 7.2.5 Nota final da discursiva

A nota final da questao discursiva e a soma dos criterios:

```text
nota_discursiva = soma(nota_criterio)
```

O aproveitamento da questao e calculado por:

```text
aproveitamento = nota_discursiva / pontuacao_total
```

### 7.2.6 Aproveitamento agregado por modelo

No benchmark discursivo, sao calculados:

- `nota_total`;
- `pontuacao_total_disc`;
- `aproveitamento_medio`;
- `aproveitamento_geral`.

Com destaque para:

```text
aproveitamento_geral = nota_total / pontuacao_total_disc
```

## 8. Regras de pontuacao

As regras de pontuacao seguem o espelho estruturado informado em `curadorias.csv`.

- Cada criterio possui `peso_total`.
- Cada componente possui seu proprio `peso`.
- A soma dos componentes de um criterio deve refletir o peso maximo esperado para aquele bloco.
- A nota final da resposta discursiva e a soma dos pontos efetivamente obtidos em cada componente.

Se uma questao discursiva vier sem `criterios`, o sistema aplica um fallback com base em `gabarito_completo`, tratando a resposta esperada como um unico criterio semantico global.

## 9. Consolidado executivo

A etapa final combina os resultados das objetivas e das discursivas.

Os principais indicadores consolidados sao:

- `score_balanceado`;
- `score_total_ponderado`.

Formulas:

```text
score_balanceado = (acuracia_objetivas + aproveitamento_geral) / 2
```

```text
score_total_ponderado =
  (acertos_objetivas + nota_total) /
  (total_objetivas + pontuacao_total_disc)
```

Esses indicadores ajudam a comparar os modelos sob duas perspectivas:

- equilibrio entre objetivas e discursivas;
- desempenho total ponderado pelo volume efetivo de pontos.

## 10. Artefatos gerados

O diretorio `artefatos_gerados/` recebe os principais produtos da execucao, como:

- questoes objetivas e discursivas preparadas;
- respostas geradas;
- benchmark das objetivas;
- benchmark das discursivas;
- composicao dos criterios discursivos;
- benchmark consolidado;
- vencedores por dificuldade;
- vencedores por disciplina;
- heatmap consolidado;
- relatorio executivo em PDF.

## 11. Ambiente de execucao

O projeto foi pensado para ambiente com GPU e token do Hugging Face.

Requisitos principais:

- Python 3;
- GPU habilitada;
- `HF_TOKEN` configurado no ambiente ou no Colab;
- bibliotecas instaladas no notebook.

Pacotes utilizados:

- `transformers`
- `accelerate`
- `bitsandbytes`
- `pandas`
- `numpy`
- `tqdm`
- `seaborn`
- `matplotlib`
- `huggingface_hub`
- `sentencepiece`
- `scipy`
- `sentence-transformers`

## 12. Papel da pasta artefatos_estudo

A pasta `artefatos_estudo/` foi preservada e nao participa diretamente da execucao do pipeline modular. Ela funciona como base de estudo e validacao metodologica, especialmente para:

- definicao da estrategia de metricas;
- entendimento do parser de gabarito;
- testes exploratorios anteriores.

Entre esses artefatos, `Metricas_Escolhidas.ipynb` serviu como referencia para a incorporacao da avaliacao semantica com SBERT e NLI no fluxo principal.

## 13. Principais decisoes de projeto

- modularizacao do pipeline para facilitar manutencao;
- centralizacao da configuracao em um unico modulo;
- reaproveitamento opcional de respostas ja geradas;
- substituicao da avaliacao qualitativa antiga por correcao estruturada baseada no espelho;
- manutencao do notebook como camada de demonstracao e execucao;
- preservacao integral da pasta `artefatos_estudo`.

## 14. Proximos aprimoramentos sugeridos

- incluir testes automatizados para parsing do gabarito;
- permitir configuracao externa dos modelos candidatos;
- enriquecer a verificacao legislativa com normalizacao mais robusta;
- adicionar validacoes de consistencia entre `peso_total` e soma dos pesos dos componentes;
- registrar logs de execucao por etapa;
- exportar relatorios adicionais por questao e por criterio.
