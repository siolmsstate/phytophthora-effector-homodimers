from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs" / "sd2_colabfold"
MANIFEST = OUTDIR / "sd2_homodimer_manifest.csv"
RESULTS_DIR = OUTDIR / "results"
PARAMS_DIR = OUTDIR / "params"


def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ColabFold v1.6.1-style AF2 predictions for SD2 homodimers.")
    parser.add_argument("--start-at", type=int, default=1, help="1-based List_No to start from.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of records to run.")
    parser.add_argument("--overwrite", action="store_true", help="Re-run jobs even when rank JSON exists.")
    parser.add_argument("--msa-mode", default="mmseqs2_uniref_env")
    parser.add_argument("--pair-mode", default="unpaired_paired")
    parser.add_argument("--num-recycles", type=int, default=3)
    parser.add_argument("--num-models", type=int, default=5)
    parser.add_argument("--num-relax", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--single-sequence", action="store_true", help="Use single_sequence mode instead of MMseqs2.")
    return parser.parse_args()


def load_manifest() -> list[dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_query_csv(job_dir: Path, jobname: str, sequence: str) -> Path:
    query_csv = job_dir / f"{jobname}.csv"
    with query_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "sequence"])
        writer.writerow([jobname, sequence])
    return query_csv


def zip_job(job_dir: Path) -> Path:
    zip_path = job_dir.with_suffix(".result.zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in job_dir.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(job_dir.parent))
    return zip_path


def job_has_results(job_dir: Path) -> bool:
    return any(job_dir.glob("*_ranking_debug.json")) or any(job_dir.glob("ranking_debug.json"))


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PARAMS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from colabfold.batch import get_queries, run, set_model_type
        from colabfold.download import download_alphafold_params
        from colabfold.utils import setup_logging
    except Exception as exc:
        raise SystemExit(
            "ColabFold is not importable in this Python environment. "
            "Install the notebook dependency first, e.g. "
            "\"python -m pip install --no-warn-conflicts "
            "'colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold'\". "
            f"Original error: {exc}"
        ) from exc

    records = [r for r in load_manifest() if int(r["list_no"]) >= args.start_at]
    if args.limit is not None:
        records = records[: args.limit]

    completed = []
    for record in records:
        base = record["id"]
        sequence = record["homodimer_sequence"]
        jobname = f"{base}_{short_hash(sequence)}"
        job_dir = RESULTS_DIR / jobname
        if job_dir.exists() and args.overwrite:
            shutil.rmtree(job_dir)
        job_dir.mkdir(parents=True, exist_ok=True)

        if job_has_results(job_dir) and not args.overwrite:
            print(f"[skip] {record['list_no']} {base}: existing ranking JSON found")
            completed.append(job_dir)
            continue

        query_csv = write_query_csv(job_dir, jobname, sequence)
        setup_logging(job_dir / "log.txt")
        queries, is_complex = get_queries(str(query_csv))
        model_type = set_model_type(is_complex, "auto")
        download_alphafold_params(model_type, PARAMS_DIR)

        msa_mode = "single_sequence" if args.single_sequence else args.msa_mode
        max_msa = None
        use_cluster_profile = not ("multimer" in model_type and max_msa is not None)
        print(f"[run] List_No={record['list_no']} job={jobname} length={record['complex_length']} model_type={model_type}")
        run(
            queries=queries,
            result_dir=job_dir,
            use_templates=False,
            custom_template_path=None,
            num_relax=args.num_relax,
            msa_mode=msa_mode,
            model_type=model_type,
            num_models=args.num_models,
            num_recycles=args.num_recycles,
            relax_max_iterations=200,
            recycle_early_stop_tolerance=None,
            num_seeds=1,
            use_dropout=False,
            model_order=[1, 2, 3, 4, 5],
            is_complex=is_complex,
            data_dir=PARAMS_DIR,
            keep_existing_results=False,
            rank_by="auto",
            pair_mode=args.pair_mode,
            pairing_strategy="greedy",
            stop_at_score=100.0,
            prediction_callback=None,
            dpi=args.dpi,
            zip_results=False,
            save_all=False,
            max_msa=max_msa,
            use_cluster_profile=use_cluster_profile,
            input_features_callback=None,
            save_recycles=False,
            user_agent="colabfold/local-sd2-batch",
            calc_extra_ptm=False,
        )
        completed.append(job_dir)
        print(f"[zip] {zip_job(job_dir)}")

    run_log = OUTDIR / "run_completed_jobs.json"
    run_log.write_text(json.dumps([str(path) for path in completed], indent=2), encoding="utf-8")
    print(f"Completed or skipped {len(completed)} jobs")


if __name__ == "__main__":
    main()
