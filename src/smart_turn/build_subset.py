import argparse
import csv
import io
import os
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download

TRAIN_REPO_ID = "pipecat-ai/smart-turn-data-v3.2-train"
TEST_REPO_ID = "pipecat-ai/smart-turn-data-v3.2-test"
HF_TOKEN = os.environ.get("HF_TOKEN")

METADATA_COLS = ["id", "language", "endpoint_bool", "midfiller", "endfiller", "synthetic", "dataset"]


def list_shards(repo_id: str):
    api = HfApi()
    files = api.list_repo_files(repo_id, repo_type="dataset")
    return sorted(f for f in files if f.startswith("data/") and f.endswith(".parquet"))


def build_subset(
    target_langs: dict,
    repo_id: str = TRAIN_REPO_ID,
    out_dir: Path = Path("data/raw"),
    manifest_path: Path = Path("data/manifest.csv"),
    shard_tmp_dir: Path = Path("data/_shard_tmp"),
):
    """
    Args:
        target_langs: dict mapping ISO 639-3 code -> desired row count for
            that language, e.g. {"hin": 12000, "eng": 15000}. A value of
            None means "no cap, take every matching row" - use this for an
            official test set where you want full coverage, not a sample.
        repo_id: HF dataset repo to pull from (train or test split repo).
        out_dir: where extracted .wav files get written, under out_dir/{lang}/.
        manifest_path: where the CSV manifest gets written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    shard_tmp_dir.mkdir(parents=True, exist_ok=True)

    counts = {lang: 0 for lang in target_langs}
    shards = list_shards(repo_id)
    print(f"Repo: {repo_id} | Found {len(shards)} shards. Target quotas: {target_langs}")

    def quota_met(lang):
        return target_langs[lang] is not None and counts[lang] >= target_langs[lang]

    with open(manifest_path, "w", newline="") as manifest_f:
        writer = csv.writer(manifest_f)
        writer.writerow(METADATA_COLS + ["wav_path"])

        for shard_name in shards:
            if all(quota_met(l) for l in target_langs):
                print("Quotas met, stopping.")
                break

            print(f"Downloading {shard_name} (resumable, cached)...")

            local_path = hf_hub_download(
                repo_id, shard_name, repo_type="dataset", token=HF_TOKEN,
                local_dir=shard_tmp_dir,
            )

            try:
                table_meta = pq.read_table(local_path, columns=METADATA_COLS)
                langs = table_meta.column("language").to_pylist()
                wanted_indices = [
                    i for i, lang in enumerate(langs)
                    if lang in target_langs and not quota_met(lang)
                ]
                if not wanted_indices:
                    print("  no needed rows in this shard, skipping audio read")
                    continue

                print(f"  {len(wanted_indices)} matching rows, decoding audio for those...")
                table_full = pq.read_table(local_path)
                selected = table_full.take(wanted_indices).to_pylist()

                for row in selected:
                    lang = row["language"]
                    if quota_met(lang):
                        continue
                    array, sr = sf.read(io.BytesIO(row["audio"]["bytes"]))
                    lang_dir = out_dir / lang
                    lang_dir.mkdir(exist_ok=True)
                    wav_path = lang_dir / f"{row['id']}.wav"
                    sf.write(wav_path, array, sr)
                    writer.writerow([row[c] for c in METADATA_COLS] + [str(wav_path)])
                    counts[lang] += 1

                manifest_f.flush()
                print(f"  running counts: {counts}")
            finally:
                Path(local_path).unlink(missing_ok=True)

    print(f"\nDone. Final counts: {counts}")
    print(f"Manifest: {manifest_path}")
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="pull the official test set instead of train")
    parser.add_argument("--hin-quota", type=int, default=None, help="cap on hin rows (default: no cap)")
    parser.add_argument("--eng-quota", type=int, default=None, help="cap on eng rows (default: no cap)")
    args = parser.parse_args()

    if args.test:
        build_subset(
            target_langs={"hin": args.hin_quota, "eng": args.eng_quota},
            repo_id=TEST_REPO_ID,
            out_dir=Path("data/test_raw"),
            manifest_path=Path("data/manifest_test.csv"),
        )
    else:
        # train subset: defaults preserved for backward compatibility with
        # earlier runs, unless overridden via --hin-quota/--eng-quota
        build_subset(
            target_langs={"hin": args.hin_quota or 12000, "eng": args.eng_quota or 15000},
            repo_id=TRAIN_REPO_ID,
            out_dir=Path("data/raw"),
            manifest_path=Path("data/manifest.csv"),
        )