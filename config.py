import os
from pathlib import Path

# ============================================================
# PERSISTÊNCIA — aponta para o Google Drive para sobreviver
# ao reinício da VM no Colab
# ============================================================
DRIVE_BASE = Path("/content/drive/MyDrive/oab_pipeline")
DRIVE_BASE.mkdir(parents=True, exist_ok=True)

HF_CACHE_DIR = Path("/content/drive/MyDrive/hf_cache")
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR)
os.environ["HF_DATASETS_CACHE"] = str(HF_CACHE_DIR)

# ============================================================
# CAMINHOS DOS ARQUIVOS DE SAÍDA
# ============================================================
QUESTOES_DISCURSIVAS_CSV     = DRIVE_BASE / "questoes_discursivas.csv"
QUESTOES_OBJETIVAS_CSV       = DRIVE_BASE / "questoes_objetivas.csv"
CURADORIA_DISCURSIVAS_CSV    = DRIVE_BASE / "curadoria_discursivas.csv"
CURADORIA_OBJETIVAS_CSV      = DRIVE_BASE / "curadoria_objetivas.csv"
RESPOSTAS_DISCURSIVAS_CSV    = DRIVE_BASE / "respostas_discursivas.csv"
RESPOSTAS_OBJETIVAS_CSV      = DRIVE_BASE / "respostas_objetivas.csv"
AVALIACAO_DISCURSIVAS_CSV    = DRIVE_BASE / "avaliacao_discursivas.csv"
ACCURACY_MODELOS_CSV         = DRIVE_BASE / "accuracy_modelos.csv"
BENCHMARK_DISCURSIVAS_CSV    = DRIVE_BASE / "questoes_discursivas_benchmark.csv"

# ============================================================
# MODELOS
# ============================================================
MODELOS_CANDIDATOS = {
    # Tamanho aproximado em disco (4bit):
    "gemma2":    "google/gemma-2-2b-it",       # ~5.25 GB
    "llama323b": "meta-llama/Llama-3.2-3B-Instruct",  # ~12.9 GB
    "llama321b": "meta-llama/Llama-3.2-1B-Instruct",  # ~4.95 GB
}

MODELO_CURADOR = "Qwen/Qwen3-4B-Instruct-2507"
MODELO_JUIZ    = "Qwen/Qwen3-4B-Instruct-2507"

# ============================================================
# RECORTES DO DATASET
# ============================================================
DISC_SLICE_START = 70
DISC_SLICE_END   = 82

OBJ_SLICE_START  = 739
OBJ_SLICE_END    = 740
