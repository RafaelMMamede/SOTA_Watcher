from pathlib import Path
import json
from utils.deduplication import STRUCTURED_FIELDS, decode
import pandas as pd


def make_output_dirs(config: dict) -> None:
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)


def load_existing_table(path: str) -> pd.DataFrame:
    file_path = Path(path)

    if not file_path.exists():
        return pd.DataFrame()

    df = pd.read_excel(file_path)
    for field in STRUCTURED_FIELDS:
        if field in df:
            df[field] = df[field].map(lambda value: decode(value, {} if field == 'metadata_variants' else []))
    return df


def save_table(df: pd.DataFrame, path: str) -> None:
    df = df.copy()
    for field in STRUCTURED_FIELDS:
        if field in df:
            df[field] = df[field].map(lambda value: json.dumps(value, ensure_ascii=False, default=str))
    # Excel truncates oversized cells silently. Fail instead of losing provenance.
    if any(isinstance(value, str) and len(value) > 32767 for value in df.to_numpy().flat):
        raise ValueError("Excel cell exceeds 32767 characters; full records remain in the discovery archive.")
    import os
    import tempfile
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=destination.parent, suffix='.xlsx')
    os.close(fd)
    try:
        df.to_excel(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)