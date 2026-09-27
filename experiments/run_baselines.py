"""Run passive fixed-pair and optional matched-timeline controls without model calls."""
import argparse
import json
from pathlib import Path
import pandas as pd
from experiments.baselines import (select_coint_pair, simulate_baseline_window, metrics_row,
                                   LONG_SHORT, SHORT_LONG, EQUAL_WEIGHT_LONG)
from experiments.main_artifacts import write_json
from experiments.recovery import batch_lock, digest
from experiments.run_ablations import _snapshot, PROJECT_ROOT
from experiments.run_redesign import _code_hash


def load_timeline(path, dates, symbols):
    frame = pd.read_csv(path)
    if "date" not in frame or "selected_pair" not in frame:
        raise ValueError("Timeline needs date and selected_pair columns")
    if list(pd.to_datetime(frame.date).dt.date) != list(dates):
        raise ValueError("Timeline must have exactly one ordered row per test decision")
    pairs=[]
    for value in frame.selected_pair:
        if pd.isna(value) or value in ("FLAT", "NO_TRADE", ""):
            pairs.append(None); continue
        pair=tuple(str(value).split("/"))
        if len(pair)!=2 or len(set(pair))!=2 or not set(pair)<=set(symbols):
            raise ValueError("Invalid timeline pair")
        pairs.append(pair)
    return pairs


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",choices=["main","herculean","herculean_regime"],default="main")
    p.add_argument("--timeline",type=Path,help="Optional local MOIRA decision CSV for matched-pair controls")
    p.add_argument("--sizing",choices=["nobeta","beta","both"],default="both")
    p.add_argument("--output-dir",type=Path,default=PROJECT_ROOT/"runs/baselines")
    p.add_argument("--cache-dir",type=Path,default=PROJECT_ROOT/"data/herculean")
    args=p.parse_args(argv)
    plan={"dataset":args.dataset,"sizing":args.sizing,"code_hash":_code_hash(),
          "timeline_hash":digest(args.timeline) if args.timeline else None}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with batch_lock(args.output_dir):
        manifest=args.output_dir/"plan.json"
        if manifest.exists() and json.loads(manifest.read_text())!=plan:
            raise ValueError("Configuration changed; choose a new output directory")
        write_json(manifest,plan)
        spec=_snapshot(args.dataset,args.output_dir,args.cache_dir)
        dates,symbols=spec["test_dates"],spec["symbols"]
        pair=select_coint_pair(spec["pool_df"],symbols,dates[0])
        schedules={"Fixed": [pair]*len(dates)}
        if args.timeline:
            schedules["Timeline"]=load_timeline(args.timeline,dates,symbols)
        rows=[];curves={"date":[*dates,spec["settlement_date"]]}
        for mode in (["nobeta","beta"] if args.sizing=="both" else [args.sizing]):
            for label,pairs in schedules.items():
                for action in (LONG_SHORT,SHORT_LONG,EQUAL_WEIGHT_LONG):
                    name=f"{label}_{action}_{mode}"
                    result=simulate_baseline_window(dates=dates,env_data=spec["env_data"],symbols=symbols,
                        pair_for_index=lambda i, pairs=pairs:pairs[i],action=action,beta_hedged=mode=="beta",
                        history_seed={d:spec["env_data"][d] for d in spec["train_dates"]},
                        settlement_date=spec["settlement_date"])
                    if len(result["normalized_equity"])!=len(dates)+1:
                        raise RuntimeError("Incomplete baseline settlement")
                    rows.append({"sizing":mode,**metrics_row(name,result)})
                    curves[name]=result["normalized_equity"]
        pd.DataFrame(rows).to_csv(args.output_dir/"comparison.csv",index=False)
        pd.DataFrame(curves).to_csv(args.output_dir/"equity.csv",index=False)
        print(f"BASELINES_OK strategies={len(rows)} test_steps={len(dates)}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
