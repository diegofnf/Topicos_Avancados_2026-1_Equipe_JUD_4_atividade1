"""Carregamento dos modelos de linguagem e de avaliacao."""

from __future__ import annotations

import gc

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

from .configuracoes import ConfiguracaoExecucao
from .utilitarios import normalizar_texto


def baixar_modelos(configuracao: ConfiguracaoExecucao) -> None:
    """Baixa todos os modelos utilizados na execucao."""
    modelos = list(configuracao.modelos_candidatos.values()) + [
        configuracao.modelo_sbert,
        configuracao.modelo_nli,
    ]
    for nome_modelo in modelos:
        print("Baixando:", nome_modelo)
        snapshot_download(repo_id=nome_modelo, cache_dir=configuracao.diretorio_cache_hf)


def carregar_modelo_geracao(nome_modelo: str, diretorio_cache_hf: str):
    """Carrega um modelo causal em 4 bits para inferencia."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(nome_modelo, cache_dir=diretorio_cache_hf)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    modelo = AutoModelForCausalLM.from_pretrained(
        nome_modelo,
        cache_dir=diretorio_cache_hf,
        device_map="auto",
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
    )
    return modelo, tokenizer


def descarregar_modelo(modelo, tokenizer) -> None:
    """Libera memoria de GPU e CPU apos o uso do modelo."""
    try:
        modelo.to("cpu")
    except Exception:
        pass
    del modelo, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def carregar_pipeline_nli(nome_modelo: str):
    """Carrega o pipeline de inferencia textual para NLI."""
    dispositivo = 0 if torch.cuda.is_available() else -1
    pipeline_nli = pipeline("text-classification", model=nome_modelo, device=dispositivo)
    print(f"NLI carregado em: {'GPU' if dispositivo == 0 else 'CPU'}")
    return pipeline_nli


class BackendEmbeddingsJuridico:
    """Backend de embeddings com cache para comparacao semantica."""

    def __init__(self, nome_modelo: str, prefixo: str = "passage", tamanho_lote: int = 8, max_length: int = 256):
        self.prefixo = prefixo
        self.tamanho_lote = tamanho_lote
        self.max_length = max_length
        self.dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[SBERT] Carregando {nome_modelo}")
        self.tokenizer = AutoTokenizer.from_pretrained(nome_modelo)
        self.modelo = AutoModel.from_pretrained(nome_modelo).to(self.dispositivo)
        self.modelo.eval()
        self.cache: dict[str, np.ndarray] = {}

    def _formatar(self, texto: str) -> str:
        texto_normalizado = normalizar_texto(texto)
        return f"{self.prefixo}: {texto_normalizado}" if self.prefixo and texto_normalizado else texto_normalizado

    @staticmethod
    def _mean_pool(hidden_states, attention_mask):
        mascara = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        soma = torch.sum(hidden_states * mascara, dim=1)
        divisao = torch.clamp(mascara.sum(dim=1), min=1e-9)
        return soma / divisao

    def obter_embeddings(self, textos: list[str]) -> dict[str, np.ndarray]:
        """Calcula embeddings apenas para textos ainda nao vistos."""
        textos_normalizados = [normalizar_texto(texto) for texto in textos]
        textos_unicos = [texto for texto in dict.fromkeys(textos_normalizados) if texto]
        faltantes = [texto for texto in textos_unicos if texto not in self.cache]

        for inicio in range(0, len(faltantes), self.tamanho_lote):
            lote = faltantes[inicio:inicio + self.tamanho_lote]
            entradas = self.tokenizer(
                [self._formatar(texto) for texto in lote],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.dispositivo)
            with torch.no_grad():
                saida = self.modelo(**entradas)
                embeddings = self._mean_pool(saida.last_hidden_state, entradas["attention_mask"])
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            for texto, embedding in zip(lote, embeddings):
                self.cache[texto] = embedding.detach().cpu().numpy()

        return {texto: self.cache[texto] for texto in textos_unicos if texto in self.cache}
