"""Configuracoes de execucao do benchmark."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from google.colab import userdata
except ImportError:
    userdata = None


@dataclass(frozen=True)
class CaminhosSaida:
    """Agrupa os arquivos de saida gerados pela execucao."""

    questoes_objetivas_csv: Path
    questoes_discursivas_csv: Path
    respostas_objetivas_csv: Path
    respostas_discursivas_csv: Path
    benchmark_objetivas_csv: Path
    avaliacao_discursivas_csv: Path
    benchmark_discursivas_csv: Path
    composicao_discursivas_csv: Path
    benchmark_consolidado_csv: Path
    vencedores_dificuldade_csv: Path
    vencedores_disciplina_csv: Path
    heatmap_consolidado_png: Path
    relatorio_executivo_pdf: Path
    discursivas_notas_por_questao_csv: Path
    objetivas_resumo_csv: Path


@dataclass(frozen=True)
class ConfiguracaoExecucao:
    """Configuracao principal da execucao."""

    raiz_projeto: Path
    curadorias_csv: Path
    diretorio_saida: Path
    diretorio_cache_hf: str
    modelos_candidatos: dict[str, str]
    modelo_sbert: str
    modelo_nli: str
    max_length_sbert: int
    reprocessar_respostas: bool
    limite_objetivas: int | None
    limite_discursivas: int | None
    caminhos_saida: CaminhosSaida
    token_hf: str | None

    def preparar_ambiente(self) -> None:
        """Garante os requisitos de ambiente e diretorios."""
        import torch

        token_hf = self.token_hf or _obter_token_hf()
        self.diretorio_saida.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = self.diretorio_cache_hf
        os.environ["HF_TOKEN"] = token_hf

        if not self.curadorias_csv.exists():
            raise FileNotFoundError(f"Arquivo nao encontrado: {self.curadorias_csv}")
        if not torch.cuda.is_available():
            raise RuntimeError("GPU nao disponivel. Ative uma GPU no ambiente de execucao.")


def _obter_token_hf() -> str:
    """Recupera o token do Hugging Face do ambiente ou do Colab."""
    token_hf = os.environ.get("HF_TOKEN")
    if userdata is not None and not token_hf:
        try:
            token_hf = userdata.get("HF_TOKEN")
        except Exception:
            token_hf = None
    if not token_hf:
        raise ValueError("HF_TOKEN nao encontrado. Configure nas Secrets do Colab ou no ambiente.")
    return token_hf


def criar_configuracao_padrao(raiz_projeto: Path | None = None) -> ConfiguracaoExecucao:
    """Cria a configuracao padrao do projeto."""
    raiz = (raiz_projeto or Path.cwd()).resolve()
    diretorio_saida = raiz / "saida_prototipo"
    caminhos_saida = CaminhosSaida(
        questoes_objetivas_csv=diretorio_saida / "questoes_objetivas_preparadas.csv",
        questoes_discursivas_csv=diretorio_saida / "questoes_discursivas_preparadas.csv",
        respostas_objetivas_csv=diretorio_saida / "respostas_objetivas.csv",
        respostas_discursivas_csv=diretorio_saida / "respostas_discursivas.csv",
        benchmark_objetivas_csv=diretorio_saida / "benchmark_objetivas.csv",
        avaliacao_discursivas_csv=diretorio_saida / "avaliacao_discursivas.csv",
        benchmark_discursivas_csv=diretorio_saida / "benchmark_discursivas.csv",
        composicao_discursivas_csv=diretorio_saida / "composicao_discursivas.csv",
        benchmark_consolidado_csv=diretorio_saida / "benchmark_consolidado.csv",
        vencedores_dificuldade_csv=diretorio_saida / "vencedores_por_dificuldade.csv",
        vencedores_disciplina_csv=diretorio_saida / "vencedores_por_disciplina.csv",
        heatmap_consolidado_png=diretorio_saida / "heatmap_consolidado.png",
        relatorio_executivo_pdf=diretorio_saida / "relatorio_consolidado.pdf",
        discursivas_notas_por_questao_csv=diretorio_saida / "discursivas_notas_por_questao.csv",
        objetivas_resumo_csv=diretorio_saida / "objetivas_resumo.csv",
    )
    return ConfiguracaoExecucao(
        raiz_projeto=raiz,
        curadorias_csv=raiz / "curadorias.csv",
        diretorio_saida=diretorio_saida,
        diretorio_cache_hf="/content/hf_cache",
        modelos_candidatos={
            "gemma2": "google/gemma-2-2b-it",
            "llama323b": "meta-llama/Llama-3.2-3B-Instruct",
            "llama321b": "meta-llama/Llama-3.2-1B-Instruct",
        },
        modelo_sbert="stjiris/bert-large-portuguese-cased-legal-mlm-sts-v1.0",
        modelo_nli="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        max_length_sbert=256,
        reprocessar_respostas=True,
        limite_objetivas=None,
        limite_discursivas=None,
        caminhos_saida=caminhos_saida,
        token_hf=os.environ.get("HF_TOKEN"),
    )
