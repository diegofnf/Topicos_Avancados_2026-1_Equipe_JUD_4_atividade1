import os
from pathlib import Path

# ============================================================
# DIRETÓRIOS
# ============================================================
BASE_DIR = Path("/content/oab_pipeline")
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
SIMILARIDADE_DISCURSIVAS_CSV = HF_SAIDA_DIR / "similaridade_discursivas.csv"
HEATMAP_DISCURSIVAS_PNG      = HF_SAIDA_DIR / "heatmap_similaridade_discursivas.png"
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
MODELO_BERTSCORE = "rufimelo/Legal-BERTimbau-base"
#MODELO_BERTSCORE = "eduagarcia/RoBERTaLexPT-base"

# ============================================================
# ANÁLISE QUALITATIVA REFERENCE-FREE
# ============================================================
MODELO_RF_ARGUMENTACAO = "intfloat/multilingual-e5-large"
MODELO_RF_PRECISAO = "rufimelo/Legal-BERTimbau-sts-large-ma-v3"
MODELO_RF_PRECISAO_FALLBACK = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODELO_RF_COESAO = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

PESO_RF_ARGUMENTACAO = 0.35
PESO_RF_PRECISAO = 0.40
PESO_RF_COESAO = 0.25

PESO_RF_PRECISAO_PROPOSICIONAL = 0.80
PESO_RF_PRECISAO_PERGUNTA = 0.20

MAX_LENGTH_RF_EMBEDDINGS = 256
MAX_LENGTH_RF_NLI = 192

# ============================================================
# RECORTES DO DATASET
# ============================================================
DISC_SLICE_START = 70
DISC_SLICE_END   = 82 #82

OBJ_SLICE_START  = 739
OBJ_SLICE_END    = 740 #862

# ============================================================
# GIT
# ============================================================
REPO_DIR    = "/content/oab_repo"
# O checkout Git auxiliar é criado apenas no final, para publicar o conteúdo de
# /content/oab_pipeline na subpasta diego_bispo do repositório remoto.
GITHUB_REPO = "diegofnf/Topicos_Avancados_2026-1_Equipe_JUD_4_atividade1"
