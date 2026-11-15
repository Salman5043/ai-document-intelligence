
from pathlib import Path

import pandas as pd

from app.document_processing.models import (
    DocumentContent,
    DocumentProcessingError,
)


def process_csv(file_path: str | Path) -> DocumentContent:
    path = Path(file_path)

    if not path.exists():
        raise DocumentProcessingError(
            f"File not found: {path}"
        )

    if path.suffix.lower() != ".csv":
        raise DocumentProcessingError(
            "Invalid CSV file extension"
        )

    dataframe = None
    last_error = None

    for encoding in [
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin1",
    ]:
        try:
            dataframe = pd.read_csv(
                path,
                encoding=encoding,
                sep=None,
                engine="python",
                on_bad_lines="error",
            )
            break

        except Exception as exc:
            last_error = exc

    if dataframe is None:
        raise DocumentProcessingError(
            f"CSV processing failed: {last_error}"
        )

    dataframe = dataframe.where(
        pd.notna(dataframe),
        None
    )

    records = dataframe.to_dict(
        orient="records"
    )

    return DocumentContent(
        filename=path.name,
        file_type="csv",
        extracted_text=dataframe.to_string(index=False),
        tables=[{
            "columns": [
                str(column)
                for column in dataframe.columns
            ],
            "row_count": len(records),
            "records": records,
        }],
        metadata={
            "row_count": len(dataframe),
            "column_count": len(dataframe.columns),
        },
    )
