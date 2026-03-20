# Repositório do Grupo 4

Este repositório reúne os artefatos individuais e coletivos produzidos pelo Grupo 4 na atividade da disciplina. A proposta da organização é concentrar o material de cada integrante em uma pasta própria na raiz do projeto, seguindo o padrão `nome_sobrenome`.

## Estrutura

- Cada aluno mantém seus artefatos dentro de uma pasta individual.
- Essa separação facilita a navegação, a avaliação e a identificação das decisões metodológicas de cada integrante.
- O material de Diego Alves Bispo está na pasta [diego_bispo](https://github.com/diegofnf/Topicos_Avancados_2026-1_Equipe_JUD_4_atividade1/tree/main/diego_bispo).

## Curadoria

A curadoria foi definida em grupo como uma etapa de apoio à análise das questões, com o objetivo de enriquecer o contexto de avaliação dos modelos.

No projeto atual, a curadoria considera principalmente:

- nível de dificuldade da questão, em escala de `1` a `4`
- justificativa curta da dificuldade
- legislação base mais relevante para o enunciado
- confiança da própria curadoria

Nas questões discursivas e objetivas, a área de especialidade é herdada diretamente do dataset, para manter consistência metodológica entre os integrantes e evitar inferências desnecessárias do modelo curador.

## Integrantes

### Diego Alves Bispo

GitHub: [Topicos_Avancados_2026-1_Equipe_JUD_4_atividade1/diego_bispo](https://github.com/diegofnf/Topicos_Avancados_2026-1_Equipe_JUD_4_atividade1/tree/main/diego_bispo)

Colab: [Abrir notebook no Google Colab](https://colab.research.google.com/github/diegofnf/Topicos_Avancados_2026-1_Equipe_JUD_4_atividade1/blob/main/diego_bispo/notebook.ipynb)

O trabalho de Diego Alves Bispo implementa um benchmark com modelos de linguagem aplicados a questões discursivas e objetivas da OAB. O projeto utiliza uma arquitetura com três papéis principais: candidato, curador e juiz, buscando comparar o comportamento dos modelos em geração de resposta, curadoria jurídica e avaliação.

Os principais parâmetros e escolhas metodológicas incluem:

- uso de modelos open-source quantizados em 4 bits
- três modelos candidatos para geração de respostas
- um modelo dedicado para curadoria e julgamento
- uso do campo `category` nas discursivas e `question_type` nas objetivas para definir a área de especialidade
- uso do campo `values` nas discursivas para limitar a pontuação atribuída pelo juiz

Os resultados incluem:

- respostas geradas para questões discursivas e objetivas
- curadoria das questões
- avaliação das discursivas
- benchmark por modelo e por área de especialidade

Para um nível maior de detalhamento, acesse [diego_bispo/README.md](/Users/Diego/Documents/Atividade1/diego_bispo/README.md).
