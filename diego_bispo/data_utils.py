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
    remote = f"https://{token}@github.com/{github_repo}.git"
    cmds = [
        "git config user.email 'colab@pipeline.com'",
        "git config user.name 'Colab Pipeline'",
        f"git remote set-url origin {remote}",
        "git add -A",  # ← adiciona tudo, incluindo subpastas
        f'git commit -m "{mensagem}" --allow-empty',
        "git push",
    ]
    for cmd in cmds:
        resultado = subprocess.run(cmd, shell=True, cwd=repo_dir, capture_output=True, text=True)
        if resultado.returncode != 0:
            print(f"Aviso git: {resultado.stderr.strip()}")
    print("Arquivos enviados para o repositório.")


def serializar_json_campo(valor) -> str:
    return json.dumps(valor, ensure_ascii=False)


def normalizar_values(values) -> list[float]:
    valores = []
    if not isinstance(values, list):
        return valores

    for value in values:
        try:
            valores.append(float(value))
        except (TypeError, ValueError):
            continue

    return valores


def somar_values(values) -> float:
    return round(sum(normalizar_values(values)), 2)


def formatar_area_especialidade_j1(category: str) -> str:
    partes = [parte for parte in str(category).split("_") if parte]
    if partes and partes[0].isdigit():
        partes = partes[1:]
    if partes and partes[0].lower() == "direito":
        partes = partes[1:]
    if not partes:
        return "Não identificado"
    return "Direito " + " ".join(parte.capitalize() for parte in partes)


def formatar_area_especialidade_j2(question_type: str) -> str:
    tokens = [token for token in str(question_type).lower().split("_") if token]
    if not tokens:
        return "Não identificado"
    return " ".join(token.capitalize() for token in tokens)


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
