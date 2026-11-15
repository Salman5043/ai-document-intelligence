
from pathlib import Path

import pandas as pd

from app.document_processing.models import (
    DocumentContent,
    DocumentProcessingError,
)


def process_excel(file_path: str | Path) -> DocumentContent:
    path = Path(file_path)

    if not path.exists():
        raise DocumentProcessingError(
            f"File not found: {path}"
        )

    if path.suffix.lower() not in {".xlsx", ".xls"}:
        raise DocumentProcessingError(
            "Unsupported Excel format"
        )

    try:
        workbook = pd.ExcelFile(path)

        tables = []
        all_text = []

        for sheet_name in workbook.sheet_names:
            dataframe = pd.read_excel(
                workbook,
                sheet_name=sheet_name
            )

            dataframe = dataframe.where(
                pd.notna(dataframe),
                None
            )

            records = dataframe.to_dict(
                orient="records"
            )

            tables.append({
                "sheet_name": sheet_name,
                "columns": [
                    str(column)
                    for column in dataframe.columns
                ],
                "row_count": len(records),
                "records": records,
            })

            all_text.append(
                f"Sheet: {sheet_name}\n"
                f"{dataframe.to_string(index=False)}"
            )

        return DocumentContent(
            filename=path.name,
            file_type="excel",
            extracted_text="\n\n".join(all_text),
            tables=tables,
            metadata={
                "sheet_count": len(workbook.sheet_names),
                "sheet_names": workbook.sheet_names,
            },
        )

    except Exception as exc:
        raise DocumentProcessingError(
            f"Excel processing failed: {exc}"
        ) from exc
