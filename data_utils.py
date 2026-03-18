import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


def carregar_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def salvar_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    print(f"CSV salvo: {path}")


def git_push(repo_dir: str, github_repo: str, token: str, mensagem: str = "Atualiza resultados") -> None:
    """
    Commita e envia os CSVs gerados para o repositório Git.
    Garante persistência dos resultados mesmo após a VM do Colab ser encerrada.
    """
    remote = f"https://{token}@github.com/{github_repo}.git"
    cmds = [
        "git config user.email 'colab@pipeline.com'",
        "git config user.name 'Colab Pipeline'",
        f"git remote set-url origin {remote}",
        "git add *.csv",
        f'git commit -m "{mensagem}" --allow-empty',
        "git push",
    ]
    for cmd in cmds:
        resultado = subprocess.run(cmd, shell=True, cwd=repo_dir, capture_output=True, text=True)
        if resultado.returncode != 0:
            print(f"Aviso git: {resultado.stderr.strip()}")
    print("CSVs enviados para o repositório.")


def area_confiavel(row) -> str:
    """
    Determina a área do benchmark com fallback para o question_id.

    Usa a classificação do curador se confiança for alta.
    Caso contrário, extrai a área diretamente do question_id,
    que é mais confiável que o modelo pequeno para esta taxonomia.

    Ex: '41_direito_administrativo_questao_1' → 'Direito Administrativo'
    """
    partes = str(row["question_id"]).split("_")
    area_id = f"Direito {partes[2].capitalize()}" if len(partes) >= 3 else "Não identificado"

    area_curador = row.get("area_especialidade_equipe")
    confianca = row.get("confianca")

    # Só aceita o curador se ele concordar com o question_id
    if confianca == "alta" and pd.notna(area_curador) and area_id.lower() in str(area_curador).lower():
        return area_curador
    return area_id


def timestamp_execucao() -> str:
    return datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat()


def extrair_json_bruto(texto: str) -> str | None:
    """
    Extrai o primeiro JSON válido do texto gerado pelo modelo.

    Estratégia:
    1. Tenta bloco markdown ```json ... ```
    2. Varre o texto caractere a caractere buscando objetos JSON válidos
       e retorna o último encontrado (mais completo)
    """
    texto = (texto or "").strip()

    blocos_md = re.findall(r"```json\s*(\{.*?\})\s*```", texto, flags=re.S | re.I)
    for candidato in reversed(blocos_md):
        try:
            json.loads(candidato)
            return candidato
        except Exception:
            pass

    decoder = json.JSONDecoder()
    candidatos_validos = []

    for i, ch in enumerate(texto):
        if ch != "{":
            continue
        try:
            obj, end = decoder.raw_decode(texto[i:])
            candidatos_validos.append(texto[i : i + end])
        except Exception:
            continue

    return candidatos_validos[-1] if candidatos_validos else None
