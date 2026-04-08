[![Open in Colab](https://img.shields.io/badge/Open%20in-Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/diegofnf/Topicos_Avancados_2026-1_Equipe_JUD_4_atividade1/blob/main/diego_bispo.ipynb)  Configure o ambiente para GPUs: T4 e cadastre o secrets HF_TOKEN.
 


# Avaliação comparativa de pequenos modelos de linguagem para resposta a perguntas jurídicas em ambientes com recursos limitados

Universidade Federal de Sergipe

Disciplina: Tópicos Avançados em ES e SI I  
Aluno: Diego Alves Bispo  
Professor: Dr. Glauco de Figueiredo Carneiro

Atividade 1 - 04/2026

[Explicação YouTube](https://www.youtube.com/watch?v=jVHmo4xAS3g)

[App Curadoria](https://my-google-ai-studio-applet-60692236585.us-west1.run.app/)

## 1. Visão geral

Este projeto implementa um benchmark de modelos de linguagem em questões da prova da OAB. O fluxo parte de um único arquivo de entrada, `curadorias.csv`, e executa as seguintes etapas:

1. preparar os dados curados;
2. carregar os modelos de geração e de avaliação;
3. gerar respostas para questões objetivas e discursivas;
4. avaliar o desempenho quantitativo nas objetivas;
5. avaliar o desempenho discursivo com base no gabarito estruturado;
6. consolidar os resultados em relatórios, tabelas e visualizações.

O arquivo `curadorias.csv` é gerado na aplicação de curadorias do Grupo 4. Link da aplicação: [Curadorias Grupo 4](https://my-google-ai-studio-applet-60692236585.us-west1.run.app/).

No recorte atual do projeto, as questões de múltipla escolha correspondem ao intervalo **739 a 861** do dataset [eduagarcia/oab_exams](https://huggingface.co/datasets/eduagarcia/oab_exams), enquanto as questões abertas correspondem ao intervalo **71 a 82** do dataset [maritaca-ai/oab-bench](https://huggingface.co/datasets/maritaca-ai/oab-bench/viewer?row=0).


## 2. Objetivo do benchmark

O objetivo central é comparar diferentes modelos de linguagem em tarefas jurídicas da OAB, considerando dois eixos complementares:

- desempenho quantitativo nas questões objetivas, por meio de acurácia;
- desempenho discursivo estruturado, por meio de aderência semântica e atendimento a referências legais previstas no espelho.

Essa abordagem permite avaliar não apenas se o modelo acerta a alternativa correta, mas também se a resposta discursiva atende aos critérios jurídicos relevantes do gabarito.

## 3. Curadoria

O projeto utiliza o arquivo `curadorias.csv` na raiz do repositório como fonte única de dados. Os parâmetros da curadoria foram definidos pelo grupo. 

Esse arquivo é gerado na [aplicação de curadorias do grupo 4](https://my-google-ai-studio-applet-60692236585.us-west1.run.app/). 

Esta aplicação é uma plataforma de curadoria especializada para o OAB-Bench, focada em estruturar dados de exames da OAB para avaliação de IAs.
A ferramenta gerencia dois conjuntos de dados: J1 (Peças Prático-Profissionais e questões discursivas) e J2 (Questões de múltipla escolha).
O processo de curadoria consiste na revisão técnica de cada questão assistida por IA(Gemini 3.1 pro) em diversas áreas do Direito.
Os curadores classificam o nível de dificuldade (de 1 a 4) com base em critérios como "lei seca" ou "caso complexo".
É feita a categorização precisa da especialidade, definindo a disciplina, o assunto específico e o tema jurídico abordado.
A aplicação realiza o mapeamento da legislação pertinente, incluindo normas, artigos específicos e URLs de referência.
O sistema utiliza Firebase para persistência em tempo real, permitindo que múltiplos curadores trabalhem simultaneamente.
O Dashboard oferece controle total sobre o progresso das atribuições e a integridade dos dados coletados por área.
Ao final, a plataforma exporta os dados em JSON e CSV padronizados, prontos para uso em benchmarks de modelos de linguagem.

Dificuldade:

1 - Fácil: 
lei seca, resposta direta, sem interpretacao

2 - Médio: 
interpretacao simples, mono artigo, baixa ambiguidade

3 - Difícil: 
caso pratico, multiplos artigos, exige raciocinio

4 - Muito Difícil: 
caso complexo, excecoes, integracao de temas, alta ambiguidade

Área de especialidade: 
Domínio presente nos guidelines do dataset. 

Legislação:
Utilizamos a legislação base e os artigos, além do identificador único URN.

| Campo                    | Principal uso                         |
|--------------------------|---------------------------------------|
| dificuldade.nivel        | benchmark                             |
| dificuldade.escala       | métricas / fine-tuning                |
| dificuldade.criterios    | explicabilidade                       |
| especialidade            | análise por área / dashboards         |
| legislacao.norma/lei     | contexto jurídico                     |
| urn/url                  | integração / RAG                      |
| artigos                  | explicabilidade                       |

O arquivo `curadorias.csv` contém:

- identificadores das questões;
- tipo da questão;
- prompt de sistema;
- enunciado;
- perguntas complementares das discursivas;
- alternativas das objetivas;
- gabarito estruturado;
- classificação de dificuldade, especialidade e legislação.

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

Esses modelos são usados tanto para questões objetivas quanto para questões discursivas. Eles são modelos gated, que necessitam de autorização prévia para uso. <b> Os modelos não foram quantizados. </b>

### 6.2 Modelo de embeddings jurídicos

Para comparação semântica das respostas discursivas, o projeto utiliza:

- `stjiris/bert-large-portuguese-cased-legal-mlm-sts-v1.0`

Esse modelo fornece embeddings jurídicos em português e sustenta a avaliação de similaridade semântica entre a resposta do candidato e a referência esperada no gabarito por sentenças.

### 6.3 Modelo de inferência textual NLI

Para verificar entailment, neutralidade e contradição semântica, o projeto utiliza:

- `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`

Esse modelo multilingual é aplicado na etapa discursiva para complementar a similaridade semântica com uma noção de consistência/contradição inferencial.

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

As questões discursivas são avaliadas a partir do espelho presente no dataset. Cada critério do espelho pode ser decomposto em componentes semânticos e legislativos com o seu respectivo peso.

A escolha das métricas não foi arbitrária. Os arquivos da pasta `artefatos_estudo/`, em especial `Metricas_Escolhidas.ipynb` e `Teste_Métricas_Qualitativas.ipynb`, foram utilizados para estudar alternativas e justificar a configuração final adotada no projeto. A partir desses experimentos, chegou-se ao conjunto de métricas hoje utilizado:

- `Legal SBERT Score`, para medir aderência semântica entre a resposta e a referência esperada;
- `NLI`, para verificar consistência inferencial e penalizar contradições;
- `MATCH` de legislação, para verificar a presença das referências legais esperadas no espelho.

Em termos práticos:

- componentes `semantico` avaliam o conteúdo jurídico da resposta;
- componentes `legislacao` avaliam se a resposta menciona a base legal exigida;
- a nota final da questão é a soma ponderada desses componentes.

#### 7.2.1 Fórmula final da questão discursiva

A nota da questão discursiva é calculada pela soma dos componentes semânticos e legislativos previstos no espelho estruturado. Nos componentes semânticos, o NLI atua como trava contra contradição: se o score inferencial ficar abaixo de `0.2`, o componente zera; caso contrário, a nota combina `SBERT` e `NLI`. Nos componentes legislativos, `MATCH` vale `1` quando a referência legal esperada aparece na resposta e `0` quando não aparece.

```text
nota_questao =
  Σ_componentes_semanticos [ peso * ( 0, se NLI < 0.2; senão 0.7*SBERT + 0.3*NLI ) ]
  +
  Σ_componentes_legislacao [ peso * MATCH ]
```

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

O projeto foi desenvolvido para rodar no ambiente do Colab com GPU T4 e token do Hugging Face.

Requisitos principais:

- Python 3;
- GPU habilitada;
- `HF_TOKEN` configurado no Colab;

Pacotes utilizados:

- `transformers`
   Biblioteca central para carregamento dos modelos de linguagem, tokenizadores e pipelines de inferência utilizados tanto na geração de respostas quanto na avaliação semântica e inferencial.
- `accelerate`
   Utilizada para otimizar a execução dos modelos em GPU, facilitando o gerenciamento de memória e a distribuição eficiente da inferência no ambiente de experimentação.
- `bitsandbytes`
   Implementei a quantização em 4 bits, porém optei por não utilizá-la nos experimentos devido à degradação significativa na qualidade das respostas. Observou-se que modelos de menor porte, quando quantizados em 4 bits, apresentam desempenho insuficiente, o que inviabiliza sua utilização em benchmarks comparativos.
Ainda assim, o código foi mantido e estruturado para permitir testes futuros, possibilitando avaliar, na prática, a regra amplamente discutida em sala de aula: modelos maiores (por exemplo, 70B) quantizados em INT4 tendem a superar modelos menores (como 8B) em precisão FP16.
- `pandas`
  Responsável pela manipulação tabular dos dados do projeto, incluindo leitura da curadorias.csv, organização das respostas geradas e consolidação dos resultados em DataFrames e arquivos CSV.
- `numpy`
  Empregado em operações numéricas auxiliares, especialmente no cálculo e agregação de métricas ao longo do pipeline de avaliação.
- `tqdm`
  Utilizado para exibir barras de progresso durante etapas mais longas da execução, melhorando o acompanhamento do processamento no notebook.
- `seaborn`
  Biblioteca empregada na construção de visualizações estatísticas, especialmente nos heatmaps comparativos de desempenho entre modelos.
- `matplotlib`
  Usada como base para geração dos gráficos e ajustes visuais das figuras apresentadas no relatório final.
- `huggingface_hub`
  Responsável pelo acesso e download dos modelos e recursos hospedados no ecossistema Hugging Face, garantindo integração direta com os checkpoints utilizados nos experimentos.
- `sentencepiece`
  Dependência necessária para tokenização em alguns modelos utilizados, permitindo o correto processamento de texto em português durante a inferência.
- `scipy`
  Utilizada em rotinas matemáticas e estatísticas complementares empregadas no cálculo de similaridade e apoio às métricas do pipeline.
- `sentence-transformers`
Biblioteca usada para carregar modelos de embeddings semânticos, especialmente o Legal SBERT Score, base da avaliação discursiva nas dimensões de precisão, argumentação e coesão legal.

## 12. Papel da pasta `artefatos_estudo`

A pasta `artefatos_estudo/` foi preservada e não participa diretamente da execução do pipeline modular. Ela funciona como base de estudo e validação metodológica, especialmente para:

- definição da estratégia de métricas;
- entendimento do parser de gabarito;
- testes exploratórios anteriores.

Entre esses artefatos, `Metricas_Escolhidas.ipynb` serviu como referência para a incorporação da avaliação semântica com SBERT e NLI no fluxo principal.

## 13. Principais decisões de projeto

- métricas qualitativas - Legal-Sbert, NLI e Match de Legislação;
- fazer a curadoria assistida por IA em uma aplicação;
- modularização do pipeline para facilitar manutenção;
- centralização da configuração em um único módulo;
- reaproveitamento opcional de respostas já geradas;
- avaliação qualitativa por correção estruturada baseada no espelho;
- manutenção do notebook como camada de demonstração e execução;
- preservação da pasta `artefatos_estudo`.

## 14. Próximos aprimoramentos sugeridos

- enriquecer a verificação legislativa com normalização mais robusta;
- avaliação com LLM as Judge
- utilização de um modelo NLI pré treinado no domínio jurídico brasileiro
- fine tuning dos modelos candidatos
- utilizar a regra de ouro da quantização: 35B em INT4 quase sempre supera um modelo 4B em FP16.

## 15. Referências

ARIAI, Farid; MACKENZIE, Joel; DEMARTINI, Gianluca. Natural language processing for the legal domain: a survey of tasks, datasets, models, and challenges. Versão 3. arXiv preprint, 2024. Disponível em: <https://doi.org/10.48550/ARXIV.2410.21306>.

CHAUHAN, Jayendra. Legal-SBERT: creating a sentence tranformer for the legal domain and generating data. [S. l.: s. n.], s.d.

GARCIA, Eduardo; SILVA, Nadia; SIQUEIRA, Felipe et al. RoBERTaLexPT: a legal RoBERTa model pretrained with deduplication for Portuguese. In: GAMALLO, Pablo; CLARO, Daniela; TEIXEIRA, António et al. (org.). Proceedings of the 16th International Conference on Computational Processing of Portuguese - Vol. 1. [S. l.]: Association for Computational Linguistics, 2024. Disponível em: <https://aclanthology.org/2024.propor-1.38/>.

HASIMOTO, Márcia. Análise textual das decisões proferidas em colegiado Tribunal de Justiça do Tocantins. [S. l.: s. n.], 2025.

SELLAM, Thibault; DAS, Dipanjan; PARIKH, Ankur. BLEURT: learning robust metrics for text generation. In: JURAFSKY, Dan; CHAI, Joyce; SCHLUTER, Natalie; TETREAULT, Joel (org.). Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. [S. l.]: Association for Computational Linguistics, 2020. Disponível em: <https://doi.org/10.18653/v1/2020.acl-main.704>.

YE, Yuxuan; SIMPSON, Edwin; SANTOS RODRIGUEZ, Raul. SBERTScore: using similarity to evaluate factual consistency in summaries. [S. l.: s. n.], s.d.

ZHANG, Tianyi; KISHORE, Varsha; WU, Felix; WEINBERGER, Kilian Q.; ARTZI, Yoav. BERTScore: evaluating text generation with BERT. [S. l.: s. n.], 2020.
