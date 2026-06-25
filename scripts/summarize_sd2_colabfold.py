from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs" / "sd2_colabfold"
MANIFEST = OUTDIR / "sd2_homodimer_manifest.csv"
RESULTS_DIR = OUTDIR / "results"
SUMMARY_CSV = OUTDIR / "sd2_colabfold_scores_summary.csv"
SUMMARY_TXT = OUTDIR / "sd2_colabfold_scores_summary.txt"


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def first_existing(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def score_value(score_data: dict, *names: str):
    def flatten_numbers(value):
        if isinstance(value, list):
            for item in value:
                yield from flatten_numbers(item)
        else:
            try:
                yield float(value)
            except (TypeError, ValueError):
                return

    for name in names:
        if name in score_data:
            value = score_data[name]
            if isinstance(value, list):
                numbers = list(flatten_numbers(value))
                return statistics.fmean(numbers) if numbers else None
            return value
    return None


def summarize_job(job_dir: Path) -> list[dict]:
    ranking_file = first_existing(list(job_dir.glob("*ranking_debug.json")) + [job_dir / "ranking_debug.json"])
    if not ranking_file:
        return []
    ranking = read_json(ranking_file)
    order = ranking.get("order") or ranking.get("rank") or []
    rows = []
    for rank_idx, model_tag in enumerate(order, start=1):
        score_candidates = list(job_dir.glob(f"*{model_tag}*scores*.json"))
        if not score_candidates:
            score_candidates = list(job_dir.glob(f"*{model_tag}*.json"))
        score_file = first_existing(score_candidates)
        score_data = read_json(score_file) if score_file else {}
        rows.append(
            {
                "rank": rank_idx,
                "model_tag": model_tag,
                "ranking_confidence": (ranking.get("iptm+ptm") or ranking.get("plddts") or {}).get(model_tag),
                "mean_plddt": score_value(score_data, "plddt", "mean_plddt"),
                "ptm": score_value(score_data, "ptm", "ptms"),
                "iptm": score_value(score_data, "iptm", "iptms"),
                "pae_mean": score_value(score_data, "pae", "predicted_aligned_error"),
                "score_json": str(score_file) if score_file else "",
                "ranking_json": str(ranking_file),
            }
        )
    return rows


def main() -> None:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))

    all_rows = []
    for record in manifest:
        matches = sorted(RESULTS_DIR.glob(f"{record['id']}_*"))
        job_dirs = [path for path in matches if path.is_dir()]
        if not job_dirs:
            all_rows.append({**record, "status": "missing"})
            continue
        job_dir = job_dirs[0]
        job_rows = summarize_job(job_dir)
        if not job_rows:
            all_rows.append({**record, "status": "no_ranking_json", "job_dir": str(job_dir)})
            continue
        for job_row in job_rows:
            all_rows.append({**record, "status": "complete", "job_dir": str(job_dir), **job_row})

    fieldnames = sorted({key for row in all_rows for key in row})
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    complete_jobs = sorted({row.get("id") for row in all_rows if row.get("status") == "complete"})
    missing = [row for row in all_rows if row.get("status") != "complete"]
    with SUMMARY_TXT.open("w", encoding="utf-8") as handle:
        handle.write("SD2 ColabFold score summary\n")
        handle.write("==========================\n\n")
        handle.write(f"Complete jobs: {len(complete_jobs)} / {len(manifest)}\n")
        handle.write(f"Rows in model-level CSV: {sum(1 for row in all_rows if row.get('status') == 'complete')}\n")
        handle.write(f"Summary CSV: {SUMMARY_CSV}\n\n")
        if missing:
            handle.write("Incomplete jobs:\n")
            for row in missing:
                handle.write(f"- {row.get('id')} ({row.get('working_name')}): {row.get('status')}\n")
        else:
            handle.write("All jobs complete. Each completed job should have five ranked model rows.\n")

    print(SUMMARY_TXT)
    print(SUMMARY_CSV)


if __name__ == "__main__":
    main()
