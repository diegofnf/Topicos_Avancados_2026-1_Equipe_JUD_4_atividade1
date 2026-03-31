"""Prompts utilizados na geração de respostas dos modelos."""

PROMPT_CANDIDATO_DISCURSIVA = """
Questao:
{questao}

Instrucoes complementares:
- Responda em portugues brasileiro, com linguagem juridica formal e objetiva.
- Nao invente base legal.
- Quando nao souber o numero exato do artigo, mencione apenas a lei aplicavel ou o entendimento juridico pertinente.
- Limite maximo de 30 linhas.

Escreva apenas a resposta final.
"""


PROMPT_CANDIDATO_OBJETIVA = """
Responda a seguinte questao objetiva da prova da OAB.

{questao}

A) {A}
B) {B}
C) {C}
D) {D}

ATENCAO: Nao inclua nada alem do JSON.
Responda apenas em JSON valido com apenas uma alternativa correta entre A, B, C e D:
{{
  "resposta_objetiva": ""
}}
"""
