"""One-off migration: split the merged run_results_V2_<year>.json files
(one file holding all 3 weight groups' run_names) into
run_results_V2_<bucket>_<year>.json files, one per weight group, matching
the WEIGHT_GROUPS split already used by run_tda_all_years.py.

Bucket is inferred from the run_name suffix:
  *_value_750   -> tx_value
  *_gasfees_750 -> gasfees
  else          -> tx_count

Old merged files are renamed to *.pre_split_bak (not deleted) so nothing
is lost if this needs to be re-run or inspected.
"""
import json
from pathlib import Path

YEARS = [2020, 2021, 2022, 2023]
PREFIX = "run_results_V2"


def bucket_for(run_name):
    if run_name.endswith("_value_750"):
        return "tx_value"
    if run_name.endswith("_gasfees_750"):
        return "gasfees"
    return "tx_count"


for year in YEARS:
    src = Path(f"{PREFIX}_{year}.json")
    if not src.exists():
        print(f"{year}: no merged file, skipping")
        continue

    data = json.loads(src.read_text())
    buckets = {"tx_count": {}, "tx_value": {}, "gasfees": {}}
    for run_name, run in data.items():
        buckets[bucket_for(run_name)][run_name] = run

    for bucket_name, entries in buckets.items():
        if not entries:
            continue
        dst = Path(f"{PREFIX}_{bucket_name}_{year}.json")
        assert not dst.exists(), f"{dst} already exists -- refusing to overwrite"
        dst.write_text(json.dumps(entries, indent=2, ensure_ascii=False))
        print(f"{year}: wrote {dst}  ({len(entries)} run(s): {sorted(entries)})")

    total_split = sum(len(v) for v in buckets.values())
    assert total_split == len(data), f"{year}: lost entries in split! {total_split} != {len(data)}"

    backup = src.with_suffix(".json.pre_split_bak")
    src.rename(backup)
    print(f"{year}: archived original to {backup}\n")

print("Migration complete.")
