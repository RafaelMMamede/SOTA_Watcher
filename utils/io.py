from pathlib import Path
import pandas as pd


def make_output_dirs(config: dict) -> None:
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)


def load_existing_table(path: str) -> pd.DataFrame:
    file_path = Path(path)

    if not file_path.exists():
        return pd.DataFrame()

    return pd.read_excel(file_path)


def save_table(df: pd.DataFrame, path: str) -> None:
    if not df.empty:
        sort_cols = [col for col in ["relevance_score", "published_date"] if col in df.columns]
        if sort_cols:
            df = df.sort_values(
                by=sort_cols,
                ascending=[False] * len(sort_cols),
            )

    df.to_excel(path, index=False)