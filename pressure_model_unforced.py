#!/usr/bin/env python3
"""SupeRR Selector: unforced, supervised-calibrated pressure candidate model.

Dependencies:
    pip install pandas numpy scikit-learn

Input files in --data-dir:
    ball_by_ball.csv
    matches.csv
    competitions.csv
    manual_reference_labels.csv  (role, player_name, manual_reference_label)

Run:
    python pressure_model.py --data-dir ./data --output-dir ./results

Important:
The manual reference labels are used ONLY to fit a general statistical classifier.
The final candidate list is selected by predicted score across all eligible players.
There is NO name-based insertion, forced inclusion, rank override or post-selection
replacement. The model is supervised-calibrated against the team's manually chosen
players, so it is not an independent blind validation of those same choices.
"""
import argparse, os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

START_DATE = "2022-01-01"
END_DATE = "2026-11-30"
IPL_WEIGHT, T20I_WEIGHT = 0.60, 0.40
BAT_MIN_BALLS, BAT_MIN_PRESSURE_BALLS = 300, 60
BOWL_MIN_OVERS, BOWL_MIN_PRESSURE_BALLS = 30, 60
BAT_RELIABILITY_K, BOWL_RELIABILITY_K = 100.0, 120.0
KNOCKOUT_MULTIPLIER = 1.05

def pct(s, higher=True):
    v = pd.to_numeric(s, errors="coerce")
    if not higher:
        v = -v
    return v.rank(pct=True, method="average")

def load_inputs(data_dir):
    # Prefer the compressed dataset for easier GitHub hosting; fall back to CSV.
    balls_gz = os.path.join(data_dir, "ball_by_ball.csv.gz")
    balls_csv = os.path.join(data_dir, "ball_by_ball.csv")
    balls_path = balls_gz if os.path.exists(balls_gz) else balls_csv
    balls = pd.read_csv(balls_path, low_memory=False)
    matches = pd.read_csv(os.path.join(data_dir, "matches.csv"), low_memory=False)
    comps = pd.read_csv(os.path.join(data_dir, "competitions.csv"), low_memory=False)
    labels = pd.read_csv(os.path.join(data_dir, "manual_reference_labels.csv"))
    required = {"role", "player_name", "manual_reference_label"}
    if not required.issubset(labels.columns):
        raise ValueError("manual_reference_labels.csv must contain role, player_name, manual_reference_label")
    meta = matches.merge(comps[["comp_id","comp_name","match_format"]], on="comp_id", how="left")
    meta["match_date"] = pd.to_datetime(meta["match_date"], errors="coerce")
    meta = meta[(meta.match_date >= START_DATE) & (meta.match_date <= END_DATE)].copy()
    is_ipl = meta["comp_name"].astype(str).str.fullmatch(r"IPL\s+\d{4}", case=False, na=False)
    is_t20i = meta["match_format"].astype(str).str.upper().eq("T20I")
    meta["format_bucket"] = np.where(is_ipl, "IPL", np.where(is_t20i, "T20I", None))
    meta = meta[meta.format_bucket.notna()].copy()
    b = balls.merge(meta[["match_id","comp_name","match_format","match_date","stage","knock_out",
                          "home_team","away_team","result","revised_target",
                          "first_inn_revised_overs","second_inn_revised_overs"]], on="match_id", how="inner")
    b["format_bucket"] = np.where(b["comp_name"].astype(str).str.fullmatch(r"IPL\s+\d{4}",case=False,na=False),
                                  "IPL","T20I")
    return b, meta, labels

def add_pressure_context(b):
    x = b.copy()
    for c in ["runs","extras1_runs","extras_2_runs"]:
        x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0)
    x["extras_type"] = x["extras_type"].fillna("")
    x["wide"] = x.extras_type.str.lower().eq("wide")
    x["noball"] = x.extras_type.str.lower().isin(["overstep","noball","no ball"])
    x["legal"] = ~(x.wide | x.noball)
    x["wicket"] = x.is_wicket.astype(str).str.lower().isin(["true","1"])
    x["total_runs"] = x.runs + x.extras1_runs + x.extras_2_runs
    x = x[x.innings.isin([1,2])].sort_values(["match_id","innings","overid","over_ball_num"]).copy()
    g = x.groupby(["match_id","innings"], sort=False)
    x["runs_before"] = g.total_runs.cumsum() - x.total_runs
    x["wickets_before"] = g.wicket.cumsum() - x.wicket.astype(int)
    x["legal_before"] = g.legal.cumsum() - x.legal.astype(int)
    x["overs_before"] = x.legal_before / 6.0
    innings_limit = np.where(
        x.innings.eq(1),
        pd.to_numeric(x["first_inn_revised_overs"], errors="coerce").fillna(20),
        pd.to_numeric(x["second_inn_revised_overs"], errors="coerce").fillna(20)
    )
    x["innings_limit_balls"] = innings_limit * 6
    x["balls_left"] = np.maximum(0, x["innings_limit_balls"] - x.legal_before)
    x["innings_progress"] = np.where(x["innings_limit_balls"]>0,
                                      x.legal_before/x["innings_limit_balls"], 0)
    x["current_rr"] = np.where(x.legal_before>0, x.runs_before/(x.legal_before/6), 0)
    totals = x.groupby(["match_id","innings"]).total_runs.sum().unstack(fill_value=0)
    first = totals[1] if 1 in totals.columns else pd.Series(dtype=float)
    target = (first+1).rename("calculated_target")
    x = x.merge(target, on="match_id", how="left")
    # Respect revised target when available.
    x["target"] = x["revised_target"].where(x["revised_target"].notna(), x["calculated_target"])
    x["runs_needed"] = x.target - x.runs_before
    x["rrr"] = np.where((x.innings==2)&(x.balls_left>0), x.runs_needed/(x.balls_left/6), 0)
    chase = x.innings.eq(2)
    urgent = chase & ((x.rrr>=9.0) | ((x.rrr-x.current_rr)>=2.0))
    late = chase & (x.innings_progress>=0.70) & (x.runs_needed>0)
    wicket_stress = chase & (x.wickets_before>=4) & (x.overs_before>=5)
    rebuild = (x.innings==1) & (x.wickets_before>=3) & (x.innings_progress<=0.60)
    death = (x.innings==1) & (x.innings_progress>=0.75) & (x.wickets_before<=6) & (x.current_rr>=8.0)
    x["pressure_flag"] = urgent | late | wicket_stress | rebuild | death
    x["pressure_weight"] = np.select(
        [urgent & late, urgent, wicket_stress, rebuild, death, late],
        [1.50,1.30,1.15,1.15,1.10,1.05], default=0.0)
    knockout = x.knock_out.astype(str).str.lower().isin(["true","1","yes"])
    x.loc[x.pressure_flag & knockout, "pressure_weight"] *= KNOCKOUT_MULTIPLIER
    # Runs charged to bowler: batter runs plus wides/no-ball extras.
    x["bowler_runs"] = x.runs + np.where(x.wide|x.noball, x.extras1_runs+x.extras_2_runs, 0)
    return x

def batting_features(x):
    d=x.copy()
    d["player_name"]=d.striker_name
    d["balls_faced"]=(~d.wide).astype(int)
    d["pressure_runs"]=d.runs*d.pressure_weight
    d["pressure_balls"]=d.balls_faced*d.pressure_weight
    d["pressure_out"]=((d.wicket)&(d.dismissed_player_id==d.striker_id)).astype(int)*d.pressure_weight
    d["pressure_chase_runs"]=d.pressure_runs*d.innings.eq(2)
    d["dismissal_of_striker"]=((d.wicket)&(d.dismissed_player_id==d.striker_id)).astype(int)
    a=d.groupby(["format_bucket","player_name"],dropna=True).agg(
        runs=("runs","sum"), balls=("balls_faced","sum"), outs=("dismissal_of_striker","sum"),
        pressure_runs=("pressure_runs","sum"), pressure_balls=("pressure_balls","sum"),
        pressure_outs=("pressure_out","sum"), chase_pressure_runs=("pressure_chase_runs","sum")
    ).reset_index()
    a["average"]=a.runs/a.outs.replace(0,np.nan)
    a["strike_rate"]=100*a.runs/a.balls.replace(0,np.nan)
    a["pressure_sr"]=100*a.pressure_runs/a.pressure_balls.replace(0,np.nan)
    a["pressure_average"]=a.pressure_runs/a.pressure_outs.replace(0,np.nan)
    a["pressure_sr_delta"]=a.pressure_sr-a.strike_rate
    a["pressure_run_rate"]=100*a.pressure_runs/a.pressure_balls.replace(0,np.nan)
    a["eligible"]=(a.balls>=BAT_MIN_BALLS)&(a.pressure_balls>=BAT_MIN_PRESSURE_BALLS)
    metrics=["pressure_runs","pressure_sr","pressure_average","chase_pressure_runs",
             "pressure_balls","average","strike_rate","pressure_sr_delta"]
    for c in metrics:
        a[c+"_pct"]=np.nan
        for fmt,idx in a[a.eligible].groupby("format_bucket").groups.items():
            a.loc[idx,c+"_pct"]=pct(a.loc[idx,c])
    # Reliability shrinkage: low pressure samples move percentile features toward the format mean.
    rel=np.sqrt(a.pressure_balls/(a.pressure_balls+BAT_RELIABILITY_K))
    for c in metrics:
        a[c+"_adj"]=0.5+(a[c+"_pct"]-0.5)*rel
    return a

def bowling_features(x):
    d=x.copy()
    d["player_name"]=d.bowler_name
    credited=d.dismissal_type.astype(str).isin(["Bowled","Caught","LBW","Stump","Stumped","Hit Wicket","Caught and Bowled"])
    d["bowler_wicket"]=(d.wicket & credited).astype(int)
    d["pressure_wickets"]=d.bowler_wicket*d.pressure_weight
    d["pressure_runs_conceded"]=d.bowler_runs*d.pressure_weight
    d["pressure_legal_balls"]=d.legal.astype(int)*d.pressure_weight
    d["pressure_dots"]=(d.legal & d.total_runs.eq(0)).astype(int)*d.pressure_weight
    d["legal_balls"]=d.legal.astype(int)
    a=d.groupby(["format_bucket","player_name"],dropna=True).agg(
        legal_balls=("legal_balls","sum"), wickets=("bowler_wicket","sum"),
        pressure_wickets=("pressure_wickets","sum"), runs_conceded=("bowler_runs","sum"),
        pressure_runs=("pressure_runs_conceded","sum"), pressure_balls=("pressure_legal_balls","sum"),
        pressure_dots=("pressure_dots","sum")
    ).reset_index()
    a["overs"]=a.legal_balls/6
    a["economy"]=a.runs_conceded/a.overs.replace(0,np.nan)
    a["pressure_economy"]=a.pressure_runs/(a.pressure_balls/6).replace(0,np.nan)
    a["pressure_wicket_rate"]=a.pressure_wickets/(a.pressure_balls/6).replace(0,np.nan)
    a["pressure_dot_pct"]=100*a.pressure_dots/a.pressure_balls.replace(0,np.nan)
    a["pressure_economy_delta"]=a.economy-a.pressure_economy
    a["eligible"]=(a.overs>=BOWL_MIN_OVERS)&(a.pressure_balls>=BOWL_MIN_PRESSURE_BALLS)
    metrics=["pressure_wicket_rate","pressure_economy","pressure_dot_pct","pressure_balls",
             "economy","wickets","pressure_economy_delta"]
    for c in metrics:
        a[c+"_pct"]=np.nan
        for fmt,idx in a[a.eligible].groupby("format_bucket").groups.items():
            a.loc[idx,c+"_pct"]=pct(a.loc[idx,c],higher=(c not in ["pressure_economy","economy"]))
    rel=np.sqrt(a.pressure_balls/(a.pressure_balls+BOWL_RELIABILITY_K))
    for c in metrics:
        a[c+"_adj"]=0.5+(a[c+"_pct"]-0.5)*rel
    return a

def make_player_matrix(a, role, label_table):
    if role=="batter":
        metric_cols=["pressure_runs","pressure_sr","pressure_average","chase_pressure_runs",
                     "pressure_balls","average","strike_rate","pressure_sr_delta"]
        sample_cols=["balls","pressure_balls"]
        mins=[BAT_MIN_BALLS,BAT_MIN_PRESSURE_BALLS]
    else:
        metric_cols=["pressure_wicket_rate","pressure_economy","pressure_dot_pct",
                     "pressure_balls","economy","wickets","pressure_economy_delta"]
        sample_cols=["overs","pressure_balls"]
        mins=[BOWL_MIN_OVERS,BOWL_MIN_PRESSURE_BALLS]
    z=a[a.eligible].copy()
    names=sorted(z.player_name.unique())
    rows=[]
    for name in names:
        g=z[z.player_name==name]
        row={"player_name":name,"format_count":g.format_bucket.nunique(),
             "ipl_present":int((g.format_bucket=="IPL").any()),
             "t20i_present":int((g.format_bucket=="T20I").any())}
        for c in metric_cols:
            adj=c+"_adj"
            vals={r.format_bucket:getattr(r,adj) for _,r in g.iterrows()}
            num=den=0.0
            for fmt,w in [("IPL",IPL_WEIGHT),("T20I",T20I_WEIGHT)]:
                if fmt in vals and pd.notna(vals[fmt]):
                    num+=w*float(vals[fmt]); den+=w
            # Do not treat missing format as zero; add format coverage as a separate feature.
            row[c+"_score"]=num/den if den else np.nan
        row["sample_total"]=float(g[sample_cols[0]].sum())
        row["sample_pressure"]=float(g[sample_cols[1]].sum())
        row["ipl_sample"]=float(g.loc[g.format_bucket=="IPL",sample_cols[0]].sum())
        row["t20i_sample"]=float(g.loc[g.format_bucket=="T20I",sample_cols[0]].sum())
        rows.append(row)
    f=pd.DataFrame(rows)
    lab=label_table[label_table.role.eq(role)][["player_name","manual_reference_label"]]
    f=f.merge(lab,on="player_name",how="left")
    f["manual_reference_label"]=f.manual_reference_label.fillna(0).astype(int)
    missing=lab[~lab.player_name.isin(f.player_name)].player_name.tolist()
    if missing:
        raise ValueError(f"Reference players missing/under eligibility in {role} data: {missing}")
    return f

def fit_and_rank(features, role, output_dir):
    labels=features.manual_reference_label
    feature_cols=[c for c in features.columns if c not in ["player_name","manual_reference_label"]]
    X=features[feature_cols].replace([np.inf,-np.inf],np.nan)
    model=Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler()),
        ("logistic",LogisticRegression(C=0.15,class_weight="balanced",max_iter=5000,random_state=42))
    ])
    model.fit(X,labels)
    result=features[["player_name","manual_reference_label"]].copy()
    result["model_score"]=model.predict_proba(X)[:,1]
    result=result.sort_values(["model_score","player_name"],ascending=[False,True]).reset_index(drop=True)
    result.insert(0,"model_rank",np.arange(1,len(result)+1))
    result["candidate_type"]=np.where(result.manual_reference_label.eq(1),
        "Model-selected; used as manual calibration example",
        "Model-selected; additional candidate")
    # Save learned coefficients for transparent review.
    coef=pd.DataFrame({"feature":feature_cols,"coefficient":model.named_steps["logistic"].coef_[0]})
    coef["absolute_coefficient"]=coef.coefficient.abs()
    coef=coef.sort_values("absolute_coefficient",ascending=False)
    result.head(20).to_csv(os.path.join(output_dir,f"top20_{role}_candidates.csv"),index=False)
    result.to_csv(os.path.join(output_dir,f"all_{role}_rankings.csv"),index=False)
    coef.to_csv(os.path.join(output_dir,f"{role}_learned_coefficients.csv"),index=False)
    return model,result,coef

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",default="./data")
    ap.add_argument("--output-dir",default="./results")
    args=ap.parse_args()
    os.makedirs(args.output_dir,exist_ok=True)
    raw,meta,labels=load_inputs(args.data_dir)
    x=add_pressure_context(raw)
    bat=batting_features(x)
    bowl=bowling_features(x)
    bf=make_player_matrix(bat,"batter",labels)
    wf=make_player_matrix(bowl,"bowler",labels)
    bm,br,bc=fit_and_rank(bf,"batter",args.output_dir)
    wm,wr,wc=fit_and_rank(wf,"bowler",args.output_dir)
    bat.to_csv(os.path.join(args.output_dir,"batter_format_metrics.csv"),index=False)
    bowl.to_csv(os.path.join(args.output_dir,"bowler_format_metrics.csv"),index=False)
    summary=pd.DataFrame([
        ["matches",meta.match_id.nunique()],
        ["ball_by_ball_rows",len(raw)],
        ["first_match_date",str(meta.match_date.min().date())],
        ["last_match_date",str(meta.match_date.max().date())],
        ["IPL_matches",meta.loc[meta.format_bucket.eq("IPL"),"match_id"].nunique()],
        ["T20I_matches",meta.loc[meta.format_bucket.eq("T20I"),"match_id"].nunique()],
        ["eligible_batters",len(bf)],
        ["eligible_bowlers",len(wf)],
        ["reference_batters_in_top20",int(br.head(20).manual_reference_label.sum())],
        ["reference_bowlers_in_top20",int(wr.head(20).manual_reference_label.sum())],
    ],columns=["metric","value"])
    summary.to_csv(os.path.join(args.output_dir,"run_summary.csv"),index=False)
    print(summary.to_string(index=False))
    print("\\nTOP 20 BATTERS\\n",br.head(20).to_string(index=False))
    print("\\nTOP 20 BOWLERS\\n",wr.head(20).to_string(index=False))

if __name__=="__main__":
    main()
