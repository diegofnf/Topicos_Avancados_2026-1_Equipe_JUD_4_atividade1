PROMPT_CANDIDATO_DISCURSIVA = '''
Você é um candidato da OAB.

Responda à questão discursiva abaixo em português brasileiro, com linguagem jurídica formal e objetiva.

Instruções obrigatórias:
- Saída somente com a resposta.
- Proibido incluir: assinatura, cumprimento, explicações, observações, cabeçalhos, markdown, inglês, exemplos, avisos ao leitor.
- Proibido inventar base legal.
- Quando não souber o número exato do artigo, mencione apenas a lei aplicável ou o entendimento jurídico pertinente.
- Máximo de 20 linhas.

A resposta deve conter exatamente:
Tese principal:
Fundamentação jurídica:
Conclusão objetiva:

Questão:
{questao}

Escreva apenas a resposta final.
'''


PROMPT_CANDIDATO_OBJETIVA = '''
Responda a seguinte questão objetiva da prova da OAB.

{questao}

A) {A}
B) {B}
C) {C}
D) {D}

Responda apenas com A, B, C ou D.
Se você responder com alguma coisa que não seja A, B, C ou D, você será penalizado.
'''


PROMPT_JUIZ_AVALIACAO = '''
Você é um examinador oficial da prova da OAB.

Sua tarefa é avaliar a resposta de um candidato para uma questão aberta da OAB.

Como esta base não possui gabarito oficial estruturado para as questões discursivas, avalie com rigor técnico:

- correção jurídica
- coerência argumentativa
- uso adequado de conceitos e fundamentos legais
- adequação da resposta ao problema apresentado

Use os seguintes critérios de pontuação:

0-2: resposta incorreta ou irrelevante
3-4: resposta muito incompleta ou com erros jurídicos graves
5-6: resposta parcialmente correta, mas com lacunas importantes
7-8: resposta correta, porém incompleta ou pouco fundamentada
9-10: resposta completa, correta e bem fundamentada juridicamente

Penalize fortemente:
- ausência de fundamentação legal
- interpretação jurídica incorreta
- invenção de fatos ou normas

CONTEXTO DA QUESTÃO (use para calibrar a avaliação):
- Nível de dificuldade: {nivel_dificuldade}
- Domínio jurídico: {dominio_juridico}
- Área de especialidade: {area_especialidade}
- Legislação base: {legislacao_base}

QUESTÃO:
{questao}

RESPOSTA DO CANDIDATO:
{resposta}

Muito importante: A resposta só deve conter o JSON válido abaixo. Não imprima nada além do JSON.
Responda apenas em JSON válido:
{{
  "nota": <número entre 0 e 10>,
  "avaliacao": "<texto curto com a justificativa em no máximo 3 linhas>"
}}
'''


PROMPT_CURADORIA = '''
Você é um curador jurídico.

Classifique a questão abaixo usando apenas o enunciado.

Nível de dificuldade:
1. Literalidade direta
2. Aplicação jurídica simples
3. Interpretação com confronto normativo
4. Estratégia argumentativa complexa

Área:
- Relações Privadas
- Estado e Administração
- Sistema Penal
- Trabalho e Proteção Social
- Tributação e Finanças
- Garantias Fundamentais

Domínio:
- Civil
- Trabalho
- Penal
- Tributário
- Administrativo
- Constitucional

Legislação base:
- Indique a principal norma aplicável (ex: CF/88, Código Civil)
- Só informe artigo se tiver alta certeza
- Não invente norma
- Se não for seguro → "Não identificado com segurança"

Regras:
- Escolha apenas uma opção por categoria
- Use a opção predominante
- Baseie-se apenas no enunciado

Questão ({tipo_questao}):
{questao}

ATENÇÃO: Não inclua nada além do JSON. Se você incluir alguma coisa além do JSON, você será penalizado.
Responda apenas em JSON:
{{
  "nivel_dificuldade": "",
  "justificativa_dificuldade": "<texto curto com a justificativa em no máximo 3 linhas>",
  "area_especialidade": "",
  "justificativa_area": "<texto curto com a justificativa em no máximo 3 linhas>",
  "dominio_juridico": "",
  "legislacao_base": "",
  "confianca": "alta|media|baixa"
}}
'''
