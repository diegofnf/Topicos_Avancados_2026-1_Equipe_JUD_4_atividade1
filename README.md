[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/diegofnf/Topicos_Avancados_2026-1_Equipe_JUD_4_atividade1/blob/main/diego_bispo.ipynb)

# Benchmark OAB com Avaliação Estruturada de Questões Objetivas e Discursivas

## 1. Visão geral

Este projeto implementa um protótipo autônomo para benchmark de modelos de linguagem em questões da prova da OAB. O fluxo parte de um único arquivo de entrada, `curadorias.csv`, e executa as seguintes etapas:

1. preparar os dados curados;
2. carregar os modelos de geração e de avaliação;
3. gerar respostas para questões objetivas e discursivas;
4. avaliar o desempenho quantitativo nas objetivas;
5. avaliar o desempenho discursivo com base no gabarito estruturado;
6. consolidar os resultados em relatórios, tabelas e visualizações.

O arquivo `curadorias.csv` é gerado na aplicação de curadorias do Grupo 4. Link da aplicação: [Curadorias Grupo 4](https://my-google-ai-studio-applet-60692236585.us-west1.run.app/).

O projeto foi refatorado para separar responsabilidades em módulos Python e deixar o notebook principal apenas como camada de orquestração.

## 2. Objetivo do benchmark

O objetivo central é comparar diferentes modelos de linguagem em tarefas jurídicas da OAB, considerando dois eixos complementares:

- desempenho quantitativo nas questões objetivas, por meio de acurácia;
- desempenho discursivo estruturado, por meio de aderência semântica e atendimento a referências legais previstas no espelho.

Essa abordagem permite avaliar não apenas se o modelo acerta a alternativa correta, mas também se a resposta discursiva atende aos critérios jurídicos relevantes do gabarito.

## 3. Base de dados

O projeto utiliza o arquivo `curadorias.csv` na raiz do repositório como fonte única de dados. O arquivo contém:

- identificadores das questões;
- tipo da questão;
- prompt de sistema;
- enunciado;
- perguntas complementares das discursivas;
- alternativas das objetivas;
- gabarito;
- metadados de dificuldade, disciplina, tema e legislação.

### 3.1 Estrutura do gabarito

As questões objetivas utilizam um gabarito simples com a alternativa correta.

As questões discursivas seguem uma estrutura JSON no campo `Gabarito`, com destaque para:

- `gabarito_completo`: texto corrido com a resposta esperada;
- `criterios`: lista estruturada de critérios de correção;
- `componentes`: subdivisões internas de cada critério.

Exemplo conceitual:

```json
{
  "gabarito_completo": "Resposta discursiva esperada...",
  "criterios": [
    {
      "id": "A",
      "secao": "Fundamentação",
      "peso_total": 0.6,
      "componentes": [
        {
          "tipo": "semantico",
          "referencia": "Tese jurídica esperada",
          "peso": 0.5
        },
        {
          "tipo": "legislacao",
          "termos": ["art. 71", "Súmula Vinculante 3"],
          "peso": 0.1
        }
      ]
    }
  ]
}
```

## 4. Arquitetura do projeto

O projeto foi modularizado em arquivos Python posicionados diretamente na raiz, com separação clara de responsabilidades.

### 4.1 Estrutura de diretórios

```text
.
|-- README.md
|-- curadorias.csv
|-- diego_bispo.ipynb
|-- artefatos_estudo/
|   |-- Metricas_Escolhidas.ipynb
|   |-- Parser_Gabarito.ipynb
|   `-- Teste_Métricas_Qualitativas.ipynb
|-- configuracoes.py
|-- prompts.py
|-- utilitarios.py
|-- carregar_dados.py
|-- modelos.py
|-- geracao_respostas.py
|-- avaliacao_objetiva.py
|-- avaliacao_discursiva.py
|-- relatorios.py
|-- pipeline.py
`-- artefatos_gerados/
```

### 4.2 Responsabilidade dos módulos

- `configuracoes.py`: centraliza caminhos, nomes de modelos, cache, limites de execução e variáveis de ambiente.
- `prompts.py`: armazena os prompts utilizados para geração das respostas.
- `utilitarios.py`: reúne funções auxiliares, como normalização textual, parsing de JSON e persistência de CSV.
- `carregar_dados.py`: lê o `curadorias.csv`, separa objetivas e discursivas, interpreta o gabarito estruturado e salva os dados preparados.
- `modelos.py`: baixa e carrega os modelos de geração, embeddings jurídicos e NLI.
- `geracao_respostas.py`: executa a inferência dos modelos candidatos.
- `avaliacao_objetiva.py`: calcula a acurácia das objetivas.
- `avaliacao_discursiva.py`: aplica a avaliação estruturada das discursivas.
- `relatorios.py`: consolida resultados, gera heatmap, tabelas e PDF.
- `pipeline.py`: fornece funções de alto nível para o notebook principal.

## 5. Notebook principal

O arquivo `diego_bispo.ipynb` é a interface principal de execução. Ele organiza o fluxo em sete blocos:

1. configuração do ambiente;
2. preparação dos dados;
3. download dos modelos;
4. geração ou reaproveitamento das respostas;
5. avaliação das objetivas;
6. avaliação discursiva estruturada;
7. consolidado executivo.

## 6. Modelos utilizados

### 6.1 Modelos candidatos para geração

Os modelos atualmente definidos para responder às questões são:

- `google/gemma-2-2b-it`
- `meta-llama/Llama-3.2-3B-Instruct`
- `meta-llama/Llama-3.2-1B-Instruct`

Esses modelos são usados tanto para questões objetivas quanto para questões discursivas.

### 6.2 Modelo de embeddings jurídicos

Para comparação semântica das respostas discursivas, o projeto utiliza:

- `stjiris/bert-large-portuguese-cased-legal-mlm-sts-v1.0`

Esse modelo fornece embeddings jurídicos em português e sustenta a avaliação de similaridade semântica entre a resposta do candidato e a referência esperada no gabarito.

### 6.3 Modelo de inferência textual NLI

Para verificar entailment, neutralidade e contradição semântica, o projeto utiliza:

- `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`

Esse modelo é aplicado na etapa discursiva para complementar a similaridade semântica com uma noção de consistência inferencial.

## 7. Métricas utilizadas

### 7.1 Métrica quantitativa das objetivas

Nas questões objetivas, a métrica principal é a acurácia:

```text
acurácia = total de acertos / total de questões objetivas
```

Também são registrados:

- `acertos_objetivas`;
- `total_objetivas`;
- `acuracia_objetivas`.

### 7.2 Métrica discursiva estruturada

As discursivas seguem a estrutura do gabarito, com critérios e componentes.

Cada critério pode possuir componentes de dois tipos:

- `semantico`;
- `legislacao`.

#### 7.2.1 Fórmula final da questão discursiva

A nota da questão discursiva é calculada pela soma dos componentes semânticos e legislativos previstos no espelho estruturado. Nos componentes semânticos, o NLI atua como trava contra contradição: se o score inferencial ficar abaixo de `0.2`, o componente zera; caso contrário, a nota combina `SBERT` e `NLI`. Nos componentes legislativos, `MATCH` vale `1` quando a referência legal esperada aparece na resposta e `0` quando não aparece.

```text
nota_questao =
  Σ_componentes_semanticos [ peso * ( 0, se NLI < 0.2; senão 0.7*SBERT + 0.3*NLI ) ]
  +
  Σ_componentes_legislacao [ peso * MATCH ]
```

#### 7.2.2 Leitura agregada por modelo

Após calcular a nota de cada questão discursiva, o projeto agrega os resultados por modelo em quatro indicadores:

- `nota_total`: soma de todos os pontos obtidos pelo modelo nas discursivas;
- `pontuacao_total_disc`: soma da pontuação máxima das discursivas avaliadas;
- `aproveitamento_medio`: média simples do aproveitamento questão a questão;
- `aproveitamento_geral`: razão entre pontos obtidos e pontos totais possíveis.

O indicador principal do benchmark discursivo é:

```text
aproveitamento_geral = nota_total / pontuacao_total_disc
```

Ele representa, de forma proporcional, quanto o modelo converteu da pontuação máxima total disponível nas questões discursivas.

## 8. Regras de pontuação

As regras de pontuação seguem o espelho estruturado informado em `curadorias.csv`.

- Cada critério possui `peso_total`.
- Cada componente possui seu próprio `peso`.
- A soma dos componentes de um critério deve refletir o peso máximo esperado para aquele bloco.
- A nota final da resposta discursiva é a soma dos pontos efetivamente obtidos em cada componente.

Se uma questão discursiva vier sem `criterios`, o sistema aplica um fallback com base em `gabarito_completo`, tratando a resposta esperada como um único critério semântico global.

## 9. Consolidado executivo

A etapa final não usa mais o consolidado antigo. O arquivo [relatorios.py](/Users/diegobispo/Documents/Atividade_1/Topicos_Avancados_2026-1_Equipe_JUD_4_atividade1/relatorios.py) hoje gera um consolidado executivo orientado por médias e por totais agregados.

Os principais indicadores do benchmark consolidado são:

- `media_discursivas`, definida como o `aproveitamento_geral` das discursivas;
- `media_geral`, que combina objetivas e discursivas no mesmo plano;
- tabelas de totais por dificuldade;
- tabelas de totais por disciplina;
- notas discursivas por questão;
- composição resumida dos critérios discursivos;
- heatmap consolidado;
- relatório executivo em PDF.

As fórmulas centrais ficaram assim:

```text
media_discursivas = aproveitamento_geral
```

```text
media_geral = (acuracia_objetivas + media_discursivas) / 2
```

Além do ranking geral por modelo, o consolidado também passa a mostrar:

- total de questões e total de pontuação por nível de dificuldade;
- total de questões e total de pontuação por disciplina;
- pontos obtidos por cada modelo em cada agrupamento;
- notas discursivas por questão, com linha de média;
- composição resumida dos critérios discursivos para auditoria do espelho.

Isso reflete melhor a lógica atual do projeto: objetivas e discursivas continuam comparáveis, mas o relatório final passou a enfatizar médias, totais por agrupamento e rastreabilidade da correção discursiva.

## 10. Artefatos gerados

O diretório `artefatos_gerados/` recebe os principais produtos da execução:

- questões objetivas e discursivas preparadas;
- respostas geradas;
- benchmark das objetivas;
- benchmark das discursivas;
- composição dos critérios discursivos;
- benchmark consolidado;
- vencedores por dificuldade;
- vencedores por disciplina;
- heatmap consolidado;
- relatório executivo em PDF;
- notas discursivas por questão;
- resumo das objetivas.

## 11. Ambiente de execução

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

## 12. Papel da pasta `artefatos_estudo`

A pasta `artefatos_estudo/` foi preservada e não participa diretamente da execução do pipeline modular. Ela funciona como base de estudo e validação metodológica, especialmente para:

- definição da estratégia de métricas;
- entendimento do parser de gabarito;
- testes exploratórios anteriores.

Entre esses artefatos, `Metricas_Escolhidas.ipynb` serviu como referência para a incorporação da avaliação semântica com SBERT e NLI no fluxo principal.

## 13. Principais decisões de projeto

- modularização do pipeline para facilitar manutenção;
- centralização da configuração em um único módulo;
- reaproveitamento opcional de respostas já geradas;
- substituição da avaliação qualitativa antiga por correção estruturada baseada no espelho;
- manutenção do notebook como camada de demonstração e execução;
- preservação integral da pasta `artefatos_estudo`.

## 14. Próximos aprimoramentos sugeridos

- incluir testes automatizados para parsing do gabarito;
- permitir configuração externa dos modelos candidatos;
- enriquecer a verificação legislativa com normalização mais robusta;
- adicionar validações de consistência entre `peso_total` e soma dos pesos dos componentes;
- registrar logs de execução por etapa;
- exportar relatórios adicionais por questão e por critério.
