from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


def plotar_heatmap_similaridade(
    matriz: pd.DataFrame | Sequence[Sequence[float]],
    labels: Sequence[str],
    output_path: str | Path | None = None,
    titulo: str = "Similaridade Semantica entre Modelos",
):
    """Plota um heatmap de similaridade e opcionalmente salva a figura em disco."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    df_matriz = pd.DataFrame(matriz, index=labels, columns=labels)

    figura, eixo = plt.subplots(figsize=(max(6, len(labels) * 1.2), max(5, len(labels) * 1.0)))
    sns.heatmap(df_matriz, annot=True, fmt=".2f", cmap="coolwarm", vmin=0, vmax=1, ax=eixo)
    eixo.set_title(titulo)
    figura.tight_layout()

    if output_path is not None:
        figura.savefig(Path(output_path), dpi=150, bbox_inches="tight")

    return figura, eixo
