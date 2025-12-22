import pandas as pd
from pathlib import Path

def df_to_latex_table(
    df: pd.DataFrame,
    outpath: str,
    caption: str,
    label: str,
    float_format="%.3f"
):
    """
    Export a publication-ready LaTeX table.
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    latex = df.to_latex(
        index=False,
        float_format=float_format,
        caption=caption,
        label=label,
        longtable=False,
        escape=False,
        column_format="l" + "c" * (len(df.columns) - 1)
    )

    with open(outpath, "w") as f:
        f.write(latex)
