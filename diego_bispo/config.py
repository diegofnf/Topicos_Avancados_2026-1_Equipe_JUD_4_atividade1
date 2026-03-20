import os
from pathlib import Path

# ============================================================
# DIRETÓRIOS
# ============================================================
BASE_DIR = Path("/content/oab_pipeline/diego_bispo")
BASE_DIR.mkdir(parents=True, exist_ok=True)

# Modelos ficam na VM — somem ao reiniciar, mas o snapshot_download
# evita re-download dentro da mesma sessão
HF_CACHE_DIR = "/content/hf_cache"
os.environ["HF_HOME"] = HF_CACHE_DIR

# ============================================================
# CAMINHOS DOS ARQUIVOS DE SAÍDA
# ============================================================
HF_SAIDA_DIR = BASE_DIR / "saida"
HF_SAIDA_DIR.mkdir(parents=True, exist_ok=True)

QUESTOES_DISCURSIVAS_CSV     = HF_SAIDA_DIR / "questoes_discursivas.csv"
QUESTOES_OBJETIVAS_CSV       = HF_SAIDA_DIR / "questoes_objetivas.csv"
CURADORIA_DISCURSIVAS_CSV    = HF_SAIDA_DIR / "curadoria_discursivas.csv"
CURADORIA_OBJETIVAS_CSV      = HF_SAIDA_DIR / "curadoria_objetivas.csv"
RESPOSTAS_DISCURSIVAS_CSV    = HF_SAIDA_DIR / "respostas_discursivas.csv"
RESPOSTAS_OBJETIVAS_CSV      = HF_SAIDA_DIR / "respostas_objetivas.csv"
AVALIACAO_DISCURSIVAS_CSV    = HF_SAIDA_DIR / "avaliacao_discursivas.csv"
BENCHMARK_OBJETIVAS_CSV      = HF_SAIDA_DIR / "benchmark_objetivas.csv"
BENCHMARK_DISCURSIVAS_CSV    = HF_SAIDA_DIR / "benchmark_discursivas.csv"

# ============================================================
# MODELOS
# ============================================================
MODELOS_CANDIDATOS = {
    "gemma2":    "google/gemma-2-2b-it",              # ~5.25 GB
    "llama323b": "meta-llama/Llama-3.2-3B-Instruct",  # ~12.9 GB
    "llama321b": "meta-llama/Llama-3.2-1B-Instruct",  # ~4.95 GB
}

MODELO_CURADOR = "Qwen/Qwen3-4B-Instruct-2507"
MODELO_JUIZ    = "Qwen/Qwen3-4B-Instruct-2507"

# ============================================================
# RECORTES DO DATASET
# ============================================================
DISC_SLICE_START = 70
DISC_SLICE_END   = 71 #82

OBJ_SLICE_START  = 739
OBJ_SLICE_END    = 740 #862

# ============================================================
# GIT
# ============================================================
REPO_DIR    = "/content/oab_pipeline/diego_bispo"
# O remote Git continua sendo o repositório raiz; `diego_bispo` é uma subpasta dentro dele.
GITHUB_REPO = "diegofnf/Topicos_Avancados_2026-1_Equipe_JUD_4_atividade1"
