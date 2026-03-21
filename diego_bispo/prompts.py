PROMPT_CANDIDATO_DISCURSIVA = '''
Responda à questão abaixo e devolva somente um JSON válido.

Questão:
{questao}

Regras:
1. A saída deve conter exatamente uma chave: "resposta_discursiva"
2. Não escreva nenhuma chave adicional
3. Não escreva texto antes ou depois do JSON
4. Não use markdown
5. Não invente base legal
6. Use português jurídico formal e objetivo
7. Máximo de 20 linhas na resposta

Formato exato:
{{
  "resposta_discursiva": "texto da resposta"
}}

Saída inválida será descartada.
'''


PROMPT_CANDIDATO_OBJETIVA = '''
Responda a seguinte questão objetiva da prova da OAB.

{questao}

A) {A}
B) {B}
C) {C}
D) {D}

ATENÇÃO: Não inclua nada além do JSON. Se você incluir alguma coisa além do JSON, você será penalizado.
Responda apenas em JSON válido:
{{
  "resposta_objetiva": "A|B|C|D"
}}
'''


PROMPT_JUIZ_AVALIACAO = '''
Você é um examinador oficial da prova da OAB.

Sua tarefa é avaliar a resposta de um candidato para uma questão aberta da OAB.

Avalie com rigor técnico usando principalmente:

- argumentação
- precisão jurídica
- coesão legal

PONTUAÇÃO DA QUESTÃO:
{pontuacao_questao}

Regras de correção:
- Se a questão tiver dois valores em `values`, use:
  1. `nota_fundamentacao_coerencia` no intervalo de 0 até o primeiro valor.
  2. `nota_aderencia_completude` no intervalo de 0 até o segundo valor.
- Se a questão tiver um único valor em `values`, trate esse valor como a nota máxima total da resposta.
- `nota_total` deve ser a soma válida das notas parciais quando houver dois valores, ou a nota única quando houver um valor.
- Nunca ultrapasse os tetos informados.

Penalize fortemente:
- ausência de fundamentação
- interpretação jurídica incorreta
- invenção de fatos ou normas

CONTEXTO DA QUESTÃO (use para calibrar a avaliação):
- Nível de dificuldade: {nivel_dificuldade}
- Área de especialidade: {area_especialidade}
- Legislação base: {legislacao_base}

QUESTÃO:
{questao}

RESPOSTA DO CANDIDATO:
{resposta}

ATENÇÃO: Não inclua nada além do JSON. Se você incluir alguma coisa além do JSON, você será penalizado.
Responda apenas em JSON válido:
{{
  "argumentacao": "<texto curto sobre a qualidade argumentativa>",
  "precisao": "<texto curto sobre a precisão jurídica>",
  "coesao_legal": "<texto curto sobre a coesão legal>",
  "nota_fundamentacao_coerencia": <número ou null>,
  "nota_aderencia_completude": <número ou null>,
  "nota_total": <número>,
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

Legislação base:
- Indique a principal norma aplicável (ex: CF/88, Código Civil)
- Só informe artigo se tiver alta certeza
- Não invente norma
- Se não for seguro → "Não identificado com segurança"

Regras:
- Escolha apenas uma opção por categoria
- A área de especialidade já foi definida pelo dataset e não deve ser inferida por você
- Baseie-se apenas no enunciado

Questão ({tipo_questao}):
{questao}

ATENÇÃO: Não inclua nada além do JSON. Se você incluir alguma coisa além do JSON, você será penalizado.
Responda apenas em JSON válido:
{{
  "nivel_dificuldade": "1|2|3|4",
  "justificativa_dificuldade": "<texto curto com a justificativa em no máximo 3 linhas>",
  "legislacao_base": "",
  "confianca": "alta|media|baixa"
}}
'''
