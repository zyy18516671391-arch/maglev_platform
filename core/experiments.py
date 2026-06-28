import csv
import json
from datetime import datetime
from pathlib import Path


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def flatten_split_metrics(split_metrics):
    """Flatten nested split metrics for compact CSV/table output."""
    flat = {}
    for split_name, metrics in split_metrics.items():
        for metric_name, value in metrics.items():
            flat[f"{split_name}_{metric_name}"] = value
    return flat


def save_experiment_record(record, output_dir="artifacts/experiments"):
    """Persist one experiment record as JSON plus a single-row CSV file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = str(record.get("mode", "comparison")).replace(" ", "_")
    stem = f"{timestamp}_{mode}"
    json_path = output_path / f"{stem}.json"
    csv_path = output_path / f"{stem}.csv"

    json_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    row = {
        "timestamp": record.get("timestamp", timestamp),
        "mode": record.get("mode", ""),
        "elapsed_ms": record.get("elapsed_ms", ""),
        "final_loss": record.get("final_loss", ""),
    }
    row.update(flatten_split_metrics(record.get("split_metrics", {})))
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    return {
        "json": str(json_path),
        "csv": str(csv_path),
    }
