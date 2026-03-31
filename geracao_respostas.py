"""Geracao de respostas das questoes objetivas e discursivas."""

from __future__ import annotations

import json
import re

import pandas as pd
import torch
from tqdm.auto import tqdm

from configuracoes import ConfiguracaoExecucao
from modelos import carregar_modelo_geracao, descarregar_modelo
from prompts import PROMPT_CANDIDATO_DISCURSIVA, PROMPT_CANDIDATO_OBJETIVA
from utilitarios import (
    carregar_json_seguro,
    extrair_json_bruto,
    normalizar_texto,
    salvar_csv,
    timestamp_execucao,
)


def gerar_texto(modelo, tokenizer, prompt: str, usar_amostragem: bool, max_tokens: int, temperatura: float = 0.7, prompt_sistema: str | None = None) -> str:
    """Executa a inferencia textual do modelo."""
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    mensagens: list[dict[str, str]] = []
    if prompt_sistema and normalizar_texto(prompt_sistema):
        mensagens.append({"role": "system", "content": normalizar_texto(prompt_sistema)})
    mensagens.append({"role": "user", "content": prompt.strip()})

    try:
        # Quando disponivel, o chat template preserva o formato esperado pelo modelo instrucional.
        resultado = tokenizer.apply_chat_template(mensagens, return_tensors="pt", add_generation_prompt=True)
        input_ids = (resultado.input_ids if hasattr(resultado, "input_ids") else resultado).to(dispositivo)
        attention_mask = torch.ones_like(input_ids)
    except Exception:
        # O fallback garante compatibilidade com tokenizers que nao expõem chat template.
        codificado = tokenizer(prompt.strip(), return_tensors="pt").to(dispositivo)
        input_ids, attention_mask = codificado.input_ids, codificado.attention_mask

    parametros = {
        "max_new_tokens": max_tokens,
        "do_sample": usar_amostragem,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.3 if usar_amostragem else 1.1,
        "attention_mask": attention_mask,
    }
    if usar_amostragem:
        parametros.update({"temperature": temperatura, "top_p": 0.9})

    with torch.no_grad():
        saida = modelo.generate(input_ids=input_ids, **parametros)
    return tokenizer.decode(saida[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()


def gerar_resposta_objetiva(modelo, tokenizer, linha: pd.Series, nome_modelo: str) -> dict:
    """Gera uma resposta para uma questao objetiva."""
    alternativas = linha["alternativas"] if isinstance(linha["alternativas"], dict) else carregar_json_seguro(linha["alternativas"], {})
    prompt = PROMPT_CANDIDATO_OBJETIVA.format(
        questao=linha["texto_questao"],
        A=alternativas.get("A", ""),
        B=alternativas.get("B", ""),
        C=alternativas.get("C", ""),
        D=alternativas.get("D", ""),
    )
    saida = "{" + gerar_texto(
        modelo,
        tokenizer,
        prompt + "\n{",
        usar_amostragem=False,
        max_tokens=40,
        prompt_sistema=linha.get("Prompt System"),
    )
    # A saida objetiva e forcada para JSON para simplificar a avaliacao automatica da alternativa.
    bruto = extrair_json_bruto(saida)
    try:
        resposta_json = json.loads(bruto) if bruto else {}
    except Exception:
        resposta_json = {}

    letra = normalizar_texto(resposta_json.get("resposta_objetiva", "")).upper()
    if letra not in {"A", "B", "C", "D"}:
        letra = "N/A"

    return {
        "question_id": linha["ID da Questão"],
        "tipo_questao": linha["Tipo Questão"],
        "dataset": linha["Dataset"],
        "modelo": nome_modelo,
        "nivel_dificuldade": linha["nivel_dificuldade"],
        "disciplina": linha["disciplina"],
        "tema": linha["tema"],
        "texto_questao": linha["texto_questao"],
        "gabarito_oficial": linha["gabarito_oficial"],
        "resposta": letra,
        "correto": letra == linha["gabarito_oficial"],
        "json_parse_ok": bool(resposta_json),
        "saida_bruta": saida,
        "timestamp_execucao": timestamp_execucao(),
    }


def gerar_resposta_discursiva(modelo, tokenizer, linha: pd.Series, nome_modelo: str) -> dict:
    """Gera uma resposta para uma questao discursiva."""
    prompt = PROMPT_CANDIDATO_DISCURSIVA.format(questao=linha["texto_questao"])
    resposta = gerar_texto(
        modelo,
        tokenizer,
        prompt,
        usar_amostragem=True,
        max_tokens=700,
        temperatura=0.7,
        prompt_sistema=linha.get("Prompt System"),
    )
    # Remove rotulos introdutorios frequentes para avaliar apenas o corpo efetivo da resposta.
    resposta = re.sub(r"^resposta\s*(final)?\s*:\s*", "", resposta, flags=re.IGNORECASE).strip()

    return {
        "question_id": linha["ID da Questão"],
        "tipo_questao": linha["Tipo Questão"],
        "dataset": linha["Dataset"],
        "modelo": nome_modelo,
        "nivel_dificuldade": linha["nivel_dificuldade"],
        "disciplina": linha["disciplina"],
        "tema": linha["tema"],
        "pontuacao_total": linha["pontuacao_total"],
        "texto_questao": linha["texto_questao"],
        "gabarito_completo": linha["gabarito_completo"],
        "criterios_correcao": json.dumps(linha["criterios_correcao"], ensure_ascii=False),
        "resposta": resposta,
        "timestamp_execucao": timestamp_execucao(),
    }


def gerar_respostas(
    df_objetivas: pd.DataFrame,
    df_discursivas: pd.DataFrame,
    configuracao: ConfiguracaoExecucao,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gera respostas para todas as questoes e todos os modelos candidatos."""
    respostas_objetivas: list[dict] = []
    respostas_discursivas: list[dict] = []

    for apelido_modelo, nome_modelo in configuracao.modelos_candidatos.items():
        print(f"Candidato: {apelido_modelo} -> {nome_modelo}")
        modelo, tokenizer = carregar_modelo_geracao(nome_modelo, configuracao.diretorio_cache_hf)

        # As objetivas usam decodificacao mais controlada para reduzir variacao fora do formato esperado.
        for _, linha in tqdm(df_objetivas.iterrows(), total=len(df_objetivas), desc=f"Objetivas | {apelido_modelo}"):
            respostas_objetivas.append(gerar_resposta_objetiva(modelo, tokenizer, linha, apelido_modelo))

        # As discursivas mantem amostragem para estimular respostas juridicas mais desenvolvidas.
        for _, linha in tqdm(df_discursivas.iterrows(), total=len(df_discursivas), desc=f"Discursivas | {apelido_modelo}"):
            respostas_discursivas.append(gerar_resposta_discursiva(modelo, tokenizer, linha, apelido_modelo))

        descarregar_modelo(modelo, tokenizer)

    df_respostas_objetivas = pd.DataFrame(respostas_objetivas)
    df_respostas_discursivas = pd.DataFrame(respostas_discursivas)
    salvar_csv(df_respostas_objetivas, configuracao.caminhos_saida.respostas_objetivas_csv)
    salvar_csv(df_respostas_discursivas, configuracao.caminhos_saida.respostas_discursivas_csv)
    return df_respostas_objetivas, df_respostas_discursivas
