#!/usr/bin/env python3
"""
Cross-dataset evaluation of lightweight, explainable ML models for IoT intrusion detection (v8).
Main datasets (--profile nf): NF-ToN-IoT-v2, NF-BoT-IoT-v2, NF-UNSW-NB15-v2 (+ optional NF-CSE-CIC-IDS2018-v2).
Secondary datasets (--profile legacy): CICIoT2023, Edge-IIoTset, ToN_IoT (network).

v8 additions (NF profile)
  * budget_tuned: stronger target-only baselines on the SAME label draws as budget_core: imbalance-aware RF
    (balanced_subsample, depth 6 or 12), shallow XGB, and RF/XGB/DT tuned by 3-fold CV on the labelled flows;
    paired Wilcoxon vs DT_d4, DT_d6 and default RF (wilcoxon_budget_tuned.csv).

v7 additions (NF profile)
  * budget_core: 50-1000 labelled target flows drawn at the target's REAL attack share; target-only vs pooled source
    + labels with 3 domain weightings (0.2 / 0.5 / 0.8), source-only, and a 50/50-drawn target-only reference.
  * Holm correction within pre-defined families (p_holm) + over all tests (p_holm_all); rank-biserial effect sizes.

v6 additions (NF profile)
  * shift_filter / shift_filter_dg: drop the SHIFT_K features whose distribution shifts most between networks, chosen
    without target labels (unlabelled target traffic, or only the training datasets' benign traffic).
  * Few-shot: 0.5 / 1 / 5 % target labels, a target-only control, compact (microcontroller) models, and the
    no-artifact feature set; paired Wilcoxon few-shot vs target-only.
  * feature_shift.csv: model-free KS shift per feature (benign and attack traffic separately).
  * prevalence_sensitivity*.csv: MCC / precision re-expressed at each dataset's real attack share.

v5 changes (NF profile)
  * core_dn: per-dataset z-scoring is applied to continuous features only (protocol / flag / TTL codes stay raw).
  * core_noart: ablation without testbed-artifact features (TTL and TCP window size), see ARTIFACT_FEATURES.
  * Multi-class: only attack families that exist in EVERY dataset are used (no mixed 'other' class).

Install:  pip install pandas numpy scikit-learn xgboost shap scipy joblib      (+ gcc for the ESP32 code check)

Usage:
  python iot_ids_pipeline.py --cic-dir data/CICIoT2023 --edge-csv data/DNN-EdgeIIoT-dataset.csv \
         --ton-csv data/train_test_network.csv --inspect                    # 1) verify column names first
  python iot_ids_pipeline.py ... --max-rows 100000 --seeds 10 --tune --out results   # 2) full run (resumable)

v4: two dataset profiles.
  --profile nf      (main experiment) NetFlow-v2 datasets with ONE common 43-feature schema
                    (NF-ToN-IoT-v2, NF-BoT-IoT-v2, NF-UNSW-NB15-v2, ...): no feature-mapping hacks.
                    Use prepare_nf_samples.py first to download + stratified-sample them; then --nf-dir <folder>.
  --profile legacy  (secondary experiment) CICIoT2023 + Edge-IIoTset + ToN_IoT with coarse shared features.

v3 additions
  * Native-feature within-dataset reference ("native"): separates the cost of feature harmonisation
    from the cost of dataset shift.
  * Checkpointing after every seed (re-run the same command to resume after a Colab disconnect).
  * Stratified (by attack type) sampling so rare attacks are not lost.
  * Few-shot adaptation: a small labelled slice of the target dataset is added to the training data.
  * Hardware-independent complexity numbers (trees, nodes, depth, parameters, max operations).
  * Compact models evaluated with the same protocol, exported to C and verified against sklearn with gcc;
    an Arduino sketch to benchmark them on an ESP32 is generated in <out>/esp32/.
(v2 features kept: fold-wise stable-feature selection, MCC/balanced-accuracy/PR-AUC, de-duplication,
 multi-class task, tuning, per-dataset normalisation, 95% CIs, Wilcoxon + Holm.)
"""
import argparse, glob, hashlib, itertools, json, os, platform, re, shutil, subprocess, sys, tempfile, time, tracemalloc, warnings
import numpy as np
import pandas as pd
import joblib
import sklearn, xgboost
from scipy import stats
from scipy.stats import spearmanr, wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, make_scorer, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ======================================================================================
# 1. FEATURE HARMONISATION  (VERIFY column names against your downloaded files: --inspect)
# ======================================================================================
CORE = ["tcp", "udp", "icmp", "dns", "http", "https"]          # available in all 3 datasets
FLOW = ["duration", "total_bytes", "avg_pkt_size"]              # flow-level datasets only (CIC, ToN)
FLOW_USED = list(FLOW)  # reduced at run time to the flow features that >= 2 datasets really have


def feature_sets():
    """(tag, features, data key, dedup exact duplicates?); flow_dn = per-dataset z-scored flow features."""
    if PROFILE == "nf":  # one common schema: "core" = all common NetFlow features; core_dn = per-dataset z-scoring
        noart = [c for c in CORE if c not in ARTIFACT_FEATURES]  # ablation: drop testbed-artifact features
        return [("core", CORE, "X", True), ("core_dn", CORE, "X_dn", True), ("core_noart", noart, "X", True)]
    fs = [("core", CORE, "X", False)]
    if FLOW_USED:
        fs += [("flow", CORE + FLOW_USED, "X", True), ("flow_dn", CORE + FLOW_USED, "X_dn", True)]
    return fs
FEWSHOT_FRACS = [0.01, 0.05]                                    # share of the target dataset used for adaptation

REQUIRED_COLS = {
    "CICIoT2023": ["TCP", "UDP", "ICMP", "DNS", "HTTP", "HTTPS", "Tot sum", "AVG", "label"],  # flow_duration optional
    "Edge-IIoTset": ["tcp.srcport", "udp.port", "icmp.checksum", "dns.qry.name.len",
                     "http.request.method", "http.tls_port", "Attack_label"],
    "ToN_IoT": ["proto", "service", "duration", "src_ip_bytes", "dst_ip_bytes", "src_pkts", "dst_pkts", "label"],
}
FAMILY_COL = {"CICIoT2023": "label", "Edge-IIoTset": "Attack_type", "ToN_IoT": "type"}  # attack names (multi-class, stratification)
PROFILE = "legacy"   # set in main(): "legacy" or "nf"
# --- NetFlow-v2 schema (identical across NF-*-v2 datasets). IPs and ports are dropped (testbed-specific identifiers),
#     DNS_QUERY_ID is dropped (random identifier). Column names: verify with --inspect.
NF_FEATURES = ["IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS", "PROTOCOL", "L7_PROTO", "TCP_FLAGS",
               "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS", "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
               "MIN_TTL", "MAX_TTL", "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
               "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES", "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS",
               "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS", "SRC_TO_DST_AVG_THROUGHPUT",
               "DST_TO_SRC_AVG_THROUGHPUT", "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
               "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES", "NUM_PKTS_1024_TO_1514_BYTES",
               "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT", "ICMP_TYPE", "ICMP_IPV4_TYPE", "DNS_QUERY_TYPE",
               "DNS_TTL_ANSWER", "FTP_COMMAND_RET_CODE"]
NF_NOLOG = {"PROTOCOL", "L7_PROTO", "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS", "ICMP_TYPE",
            "ICMP_IPV4_TYPE", "FTP_COMMAND_RET_CODE", "DNS_QUERY_TYPE", "MIN_TTL", "MAX_TTL"}  # codes/flags: keep raw
NF_REQUIRED = ["Label", "Attack"]
# Features that mostly describe the capture TESTBED (operating systems / network path) rather than attack behaviour.
# They are dropped in the "core_noart" ablation; change with --artifact-features.
ARTIFACT_FEATURES = ["MIN_TTL", "MAX_TTL", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT"]
# columns excluded from the "native" feature matrix (labels, identifiers, timestamps)
NATIVE_DROP = re.compile(r"(label|attack|^type$|^ts$|time|^date$|src_ip$|dst_ip$|ip\.(src|dst)|mac|stream|^id$)")


def present(s):
    """1 if a field is 'present' (non-zero number or non-empty / non-'0' string), else 0."""
    num_ = pd.to_numeric(s, errors="coerce")
    isnum = num_.notna()
    out = pd.Series(0, index=s.index, dtype=int)
    out[isnum] = (num_[isnum] != 0).astype(int)
    st = s[~isnum].astype(str).str.strip().str.lower()
    out[~isnum] = (~st.isin(["", "0", "0.0", "nan", "none", "-"])).astype(int)
    return out


def num(s):
    return pd.to_numeric(s, errors="coerce")


def check_cols(df, name):
    missing = [c for c in REQUIRED_COLS.get(name, NF_REQUIRED) if c not in df.columns]
    if missing:
        raise KeyError(f"[{name}] expected columns not found: {missing}\nAvailable: {list(df.columns)}\n"
                       f"Edit REQUIRED_COLS / harmonise_{name.split('-')[0].split('_')[0].lower()}() in this script.")


def family_of(raw):
    """Map dataset-specific attack names to common families."""
    r = str(raw).strip().lower()
    if r in ("normal", "0") or r.startswith("benign"):
        return "benign"
    if any(k in r for k in ("mirai", "ransom", "backdoor", "malware", "botnet")) or r == "bot":
        return "malware"
    if re.search(r"(^|[^a-z])d?dos", r):
        return "dos_ddos"
    if any(k in r for k in ("scan", "recon", "fingerprint", "sweep", "vulnerab")):
        return "recon"
    if any(k in r for k in ("sql", "xss", "inject", "upload", "browserhijack", "command")):
        return "web"
    if any(k in r for k in ("password", "brute", "dictionary")):
        return "bruteforce"
    if "web" in r:
        return "web"
    if any(k in r for k in ("mitm", "spoof", "arp", "hijack")):
        return "spoofing_mitm"
    return "other"


def harmonise_cic(df):
    check_cols(df, "CICIoT2023")
    f = pd.DataFrame({
        "tcp": present(df["TCP"]), "udp": present(df["UDP"]), "icmp": present(df["ICMP"]),
        "dns": present(df["DNS"]), "http": present(df["HTTP"]), "https": present(df["HTTPS"]),
        "duration": (np.log1p(num(df["flow_duration"]).clip(lower=0)) if "flow_duration" in df.columns else np.nan),
        "total_bytes": np.log1p(num(df["Tot sum"]).clip(lower=0)),
        "avg_pkt_size": np.log1p(num(df["AVG"]).clip(lower=0)),
    })
    raw = df["label"].astype(str)
    y = (~raw.str.strip().str.lower().str.startswith("benign")).astype(int)
    return f, y, raw.map(family_of)


def harmonise_edge(df):
    check_cols(df, "Edge-IIoTset")
    f = pd.DataFrame({
        "tcp": present(df["tcp.srcport"]), "udp": present(df["udp.port"]),
        "icmp": present(df["icmp.checksum"]), "dns": present(df["dns.qry.name.len"]),
        "http": present(df["http.request.method"]), "https": present(df["http.tls_port"]),
    })
    for c in FLOW:  # packet-level data: no flow statistics
        f[c] = np.nan
    y = num(df["Attack_label"]).fillna(0).astype(int)
    fam = df["Attack_type"].map(family_of) if "Attack_type" in df.columns else None
    return f, y, fam


def harmonise_ton(df):
    check_cols(df, "ToN_IoT")
    proto = df["proto"].astype(str).str.lower().str.strip()
    svc = df["service"].astype(str).str.lower().str.strip()
    tb = (num(df["src_ip_bytes"]).fillna(0) + num(df["dst_ip_bytes"]).fillna(0)).clip(lower=0)
    pk = (num(df["src_pkts"]).fillna(0) + num(df["dst_pkts"]).fillna(0)).clip(lower=1)
    f = pd.DataFrame({
        "tcp": (proto == "tcp").astype(int), "udp": (proto == "udp").astype(int),
        "icmp": (proto == "icmp").astype(int), "dns": (svc == "dns").astype(int),
        "http": (svc == "http").astype(int), "https": svc.isin(["ssl", "https"]).astype(int),
        "duration": np.log1p(num(df["duration"]).clip(lower=0)),
        "total_bytes": np.log1p(tb), "avg_pkt_size": np.log1p(tb / pk),
    })
    y = num(df["label"]).fillna(0).astype(int)
    fam = df["type"].map(family_of) if "type" in df.columns else None
    return f, y, fam


def harmonise_nf(df):
    check_cols(df, "NF")
    cols = [c for c in NF_FEATURES if c in df.columns]
    if len(cols) < 20:
        raise KeyError(f"Only {len(cols)} of the expected NetFlow feature columns were found. Available: {list(df.columns)}")
    X = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    for c in cols:
        if c not in NF_NOLOG:
            X[c] = np.log1p(X[c].clip(lower=0))
    y = num(df["Label"]).fillna(0).astype(int)
    return X, y, df["Attack"].astype(str).map(family_of)


def native_matrix(df):
    """All usable columns of a dataset (no harmonisation): numeric, or low-cardinality categorical as codes."""
    cols = {}
    for c in df.columns:
        if NATIVE_DROP.search(c.lower()):
            continue
        s = df[c]
        n_ = pd.to_numeric(s, errors="coerce")
        if n_.notna().mean() >= 0.9:
            cols[c] = n_.fillna(0).astype("float32")
        elif s.nunique() <= 30:
            cols[c] = s.astype(str).astype("category").cat.codes.astype("float32")
    X = pd.DataFrame(cols)
    if X.shape[1]:
        X = X.loc[:, X.nunique() > 1]
    return X.replace([np.inf, -np.inf], 0).clip(-1e12, 1e12)


# ======================================================================================
# 2. LOADING  (stratified sampling by attack type)
# ======================================================================================
def read_csv(path, **kw):
    df = pd.read_csv(path, low_memory=False, **kw)
    df.columns = [c.strip() for c in df.columns]
    return df


def stratified_sample(df, col, max_rows, seed, min_per_class=300):
    """Keep up to max_rows rows while guaranteeing rare attack types are represented."""
    if len(df) <= max_rows:
        return df.reset_index(drop=True)
    g = df[col].astype(str)
    counts = g.value_counts()
    base = np.minimum(counts.values, min_per_class)
    if base.sum() >= max_rows:
        alloc = np.maximum(1, (base * max_rows / base.sum()).astype(int))
    else:
        room = counts.values - base
        extra = np.floor((max_rows - base.sum()) * room / max(room.sum(), 1)).astype(int)
        alloc = base + np.minimum(extra, room)
    parts = [df[g == c].sample(n=int(a), random_state=seed) for c, a in zip(counts.index, alloc)]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def strat_col(name, df):
    c = FAMILY_COL.get(name, "Attack")
    return c if c in df.columns else REQUIRED_COLS.get(name, NF_REQUIRED)[-1]


def label_from_filename(fp):
    """Raw per-capture CIC files have no label column: 'DDoS-ICMP_Flood3.pcap.csv' -> 'DDoS-ICMP_Flood'."""
    b = re.sub(r"\.csv$", "", os.path.basename(fp), flags=re.I)
    b = re.sub(r"\.pcap$", "", b, flags=re.I)
    return re.sub(r"[_\- ]?\d+$", "", b)


def load_cic(folder, max_rows, seed):
    files = sorted(glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True))  # also finds nested folders
    if not files:
        raise FileNotFoundError(f"No CSV files found under {folder}")
    print(f"CICIoT2023: {len(files)} CSV files found")
    per_file = max(1, int(3 * max_rows / len(files)))
    parts = []
    for fp in files:
        d = read_csv(fp)
        if "label" not in d.columns:
            d["label"] = label_from_filename(fp)
        parts.append(d.sample(n=min(len(d), per_file), random_state=seed))
    return pd.concat(parts, ignore_index=True)


def resolve_csv(path, hint):
    """Accept a file, or a folder that contains the file (e.g. 'name.csv/name.csv' after unzipping)."""
    if os.path.isdir(path):
        c = sorted(glob.glob(os.path.join(path, "**", "*.csv"), recursive=True))
        if not c:
            raise FileNotFoundError(f"No CSV file inside folder {path}")
        c = [f for f in c if hint.lower() in os.path.basename(f).lower()] or c
        print(f"Using {c[0]}")
        return c[0]
    return path


def load_datasets(args):
    raw = {}
    if args.profile == "nf":
        files = sorted(glob.glob(os.path.join(args.nf_dir, "*_sample.csv*")))
        if not files:
            raise FileNotFoundError(f"No *_sample.csv.gz files in {args.nf_dir}. Run prepare_nf_samples.py first.")
        for f in files:
            name = os.path.basename(f).split("_sample")[0]
            raw[name] = (read_csv(f), harmonise_nf)
            print(f"{name}: loaded {len(raw[name][0]):,} rows from {os.path.basename(f)}")
    if args.profile == "legacy" and args.cic_dir:
        raw["CICIoT2023"] = (load_cic(args.cic_dir, args.max_rows, 0), harmonise_cic)
    if args.profile == "legacy" and args.edge_csv:
        raw["Edge-IIoTset"] = (read_csv(resolve_csv(args.edge_csv, "edge")), harmonise_edge)
    if args.profile == "legacy" and args.ton_csv:
        raw["ToN_IoT"] = (read_csv(resolve_csv(args.ton_csv, "network")), harmonise_ton)
    if len(raw) < 2:
        raise SystemExit("Provide at least two datasets (nf: two *_sample.csv.gz files in --nf-dir; "
                         "legacy: --cic-dir, --edge-csv, --ton-csv).")
    if args.inspect:
        failed = False
        for n, (df, _) in raw.items():
            print(f"\n=== {n}: {df.shape} ===\n{list(df.columns)}")
            try:
                check_cols(df, n)
                print(f"[{n}] all required columns found.")
            except KeyError as e:
                failed = True
                print(f"[{n}] PROBLEM: {str(e)[:1500]}")
                continue
            fc = FAMILY_COL.get(n, "Attack")
            if fc in df.columns:
                print(f"[{n}] attack types in '{fc}' -> families:")
                print(df[fc].astype(str).map(family_of).value_counts().to_string())
                print(df[fc].astype(str).value_counts().head(15).to_string())
            else:
                print(f"[{n}] NOTE: '{fc}' (attack type) not found -> multi-class task will skip this dataset.")
            if args.profile == "legacy":
                print(f"[{n}] native feature count: {native_matrix(df.head(5000)).shape[1]}")
        raise SystemExit(1 if failed else 0)
    data = {}
    for n, (df, fn) in raw.items():
        df = stratified_sample(df, strat_col(n, df), args.max_rows, 0)
        X, y, fam = fn(df)
        X = X.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
        X_dn = X.copy()  # mitigation: z-score continuous features with the dataset's OWN (unlabeled) statistics
        # continuous features only: z-scoring codes (PROTOCOL, L7_PROTO, flags, TTL) would give the same code a
        # different value in each dataset
        for c in ([c for c in X.columns if c not in NF_NOLOG] if args.profile == "nf" else FLOW):
            if c in X.columns and X[c].notna().all():
                X_dn[c] = (X[c] - X[c].mean()) / (X[c].std() + 1e-9)
        avail = [c for c in X.columns if X[c].notna().all()]
        data[n] = {"X": X, "X_dn": X_dn, "y": y.reset_index(drop=True), "avail": avail,
                   "X_native": (native_matrix(df).reset_index(drop=True) if args.profile == "legacy" else pd.DataFrame()),
                   "fam": None if fam is None else fam.reset_index(drop=True)}
        print(f"{n}: {len(X):,} rows | attack share {y.mean():.2%} | features: {avail} | native features: "
              f"{data[n]['X_native'].shape[1]}" + (f" | families: {sorted(fam.unique())}" if fam is not None else ""))
    return data


def eligible(data, feats):
    return [n for n, d in data.items() if all(f in d["avail"] for f in feats)]


def build_view(data, feats, xkey, dedup, tag, report):
    """name -> (X, y, family). Exact duplicates (features+label) are removed BEFORE any split if dedup."""
    v = {}
    for n in eligible(data, feats):
        X, y = data[n][xkey][feats], data[n]["y"]
        keep = np.ones(len(X), bool)
        if dedup:
            keep = ~pd.concat([X, y.rename("_y")], axis=1).duplicated().to_numpy()
        report.append(dict(featset=tag, dataset=n, rows_before=len(X), rows_after=int(keep.sum()), dedup=dedup))
        fam = data[n]["fam"]
        v[n] = (X.to_numpy(float)[keep], y.to_numpy()[keep], None if fam is None else fam.to_numpy()[keep])
    return v


def build_native_views(data, report):
    v = {}
    for n, d in data.items():
        X, y = d["X_native"], d["y"]
        keep = ~pd.concat([X, y.rename("_y")], axis=1).duplicated().to_numpy()
        report.append(dict(featset="native", dataset=n, rows_before=len(X), rows_after=int(keep.sum()), dedup=True))
        v[n] = (X.to_numpy(np.float32)[keep], y.to_numpy()[keep], X.shape[1])
    return v


# ======================================================================================
# 3. MODELS, TUNING, METRICS, COMPLEXITY
# ======================================================================================
MODEL_NAMES = ["RF", "XGB", "MLP", "DT", "LR"]
SHAP_MODELS = ["RF", "XGB", "MLP"]
GRIDS = {
    "RF": {"n_estimators": [50, 100, 200], "max_depth": [6, 12, 20], "min_samples_leaf": [1, 2, 5]},
    "XGB": {"n_estimators": [50, 100, 200], "max_depth": [3, 6, 9], "learning_rate": [0.05, 0.1, 0.2]},
    "MLP": {"mlpclassifier__hidden_layer_sizes": [(16,), (32, 16), (64, 32)],
            "mlpclassifier__alpha": [1e-4, 1e-3, 1e-2]},
    "DT": {"max_depth": [4, 8, 12, None], "min_samples_leaf": [1, 5, 20]},
    "LR": {"logisticregression__C": [0.1, 1.0, 10.0]},
}
# Compact models for microcontroller deployment: a FIXED list (not tuned on results, to avoid cherry-picking)
_lr = lambda s: make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, class_weight="balanced"))
_mlp = lambda h: (lambda s: make_pipeline(StandardScaler(), MLPClassifier(h, max_iter=60, early_stopping=True, random_state=s)))
_dt = lambda d: (lambda s: DecisionTreeClassifier(max_depth=d, min_samples_leaf=5, class_weight="balanced", random_state=s))
_rf = lambda t: (lambda s: RandomForestClassifier(n_estimators=t, max_depth=6, min_samples_leaf=5,
                                                  class_weight="balanced", n_jobs=-1, random_state=s))
COMPACT = {"DT_d4": _dt(4), "DT_d6": _dt(6), "DT_d8": _dt(8), "RF_t5_d6": _rf(5), "RF_t10_d6": _rf(10),
           "RF_t20_d6": _rf(20), "LR": _lr, "MLP_8": _mlp((8,)), "MLP_16_8": _mlp((16, 8))}


def make_model(name, seed, params=None):
    if name in COMPACT:
        return COMPACT[name](seed)
    if name == "RF":
        m = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=2,
                                   class_weight="balanced", n_jobs=-1, random_state=seed)
    elif name == "XGB":
        m = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, tree_method="hist",
                          n_jobs=-1, random_state=seed, eval_metric="logloss")
    elif name == "MLP":
        m = make_pipeline(StandardScaler(),
                          MLPClassifier((32, 16), max_iter=60, early_stopping=True, random_state=seed))
    elif name == "DT":
        m = DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=seed)
    elif name == "LR":
        m = _lr(seed)
    else:
        raise ValueError(name)
    if params:
        m.set_params(**params)
    return m


def fit(model, name, X, y):
    if name == "XGB":
        model.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    else:
        model.fit(X, y)
    return model


class Tuner:
    """Randomised search (3-fold CV, MCC) on a subsample of the TRAINING data only; cached (and saved) per training set."""
    def __init__(self, enabled, path=None):
        self.enabled, self.path, self.cache, self.log = enabled, path, {}, []
        if path and os.path.exists(path):
            self.cache, self.log = joblib.load(path)

    def params(self, name, key, X, y):
        if not self.enabled or name not in GRIDS:
            return {}
        k = (name, key)
        if k not in self.cache:
            idx = np.random.default_rng(0).choice(len(X), size=min(len(X), 20000), replace=False)
            grid = GRIDS[name]
            size = int(np.prod([len(v) for v in grid.values()]))
            s = RandomizedSearchCV(make_model(name, 0), grid, n_iter=min(6, size), cv=3,
                                   scoring=make_scorer(matthews_corrcoef), random_state=0, refit=False)
            kw = {"sample_weight": compute_sample_weight("balanced", y[idx])} if name == "XGB" else {}
            try:
                s.fit(X[idx], y[idx], **kw)
                self.cache[k] = s.best_params_
            except Exception as e:  # keep going with defaults
                print(f"  tuning failed for {k}: {e}")
                self.cache[k] = {}
            self.log.append(dict(model=name, key=str(key), best_params=str(self.cache[k])))
            if self.path:
                joblib.dump((self.cache, self.log), self.path)
        return self.cache[k]


def binary_metrics(y, pred, score):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    try:
        auc, ap = roc_auc_score(y, score), average_precision_score(y, score)
    except Exception:
        auc = ap = np.nan
    return dict(accuracy=accuracy_score(y, pred), balanced_acc=balanced_accuracy_score(y, pred),
                precision=precision_score(y, pred, zero_division=0), recall=recall_score(y, pred, zero_division=0),
                f1=f1_score(y, pred, zero_division=0), macro_f1=f1_score(y, pred, average="macro", zero_division=0),
                mcc=matthews_corrcoef(y, pred), fpr=fp / max(fp + tn, 1), roc_auc=auc, pr_auc=ap,
                attack_share=float(np.mean(y)))


def metrics(model, X, y):
    return binary_metrics(y, model.predict(X), model.predict_proba(X)[:, 1])


def complexity(model):
    """Hardware-independent size/cost numbers: trees, nodes, depth, parameters, worst-case operations per prediction."""
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    d = dict(n_trees=0, total_nodes=0, max_depth=0, n_params=0, max_ops=0)
    if isinstance(est, DecisionTreeClassifier):
        t = est.tree_; d.update(n_trees=1, total_nodes=int(t.node_count), max_depth=int(t.max_depth), max_ops=int(t.max_depth))
    elif isinstance(est, RandomForestClassifier):
        ts = [e.tree_ for e in est.estimators_]
        d.update(n_trees=len(ts), total_nodes=int(sum(t.node_count for t in ts)),
                 max_depth=int(max(t.max_depth for t in ts)), max_ops=int(sum(t.max_depth for t in ts)))
    elif isinstance(est, XGBClassifier):
        df = est.get_booster().trees_to_dataframe(); md = int(est.get_params()["max_depth"] or 6)
        d.update(n_trees=int(df.Tree.nunique()), total_nodes=len(df), max_depth=md, max_ops=int(df.Tree.nunique()) * md)
    elif isinstance(est, LogisticRegression):
        d.update(n_params=int(est.coef_.size + 1), max_ops=int(est.coef_.size))
    elif isinstance(est, MLPClassifier):
        macs = int(sum(w.size for w in est.coefs_))
        d.update(n_params=macs + int(sum(b.size for b in est.intercepts_)), max_ops=macs)
    return d


# ======================================================================================
# 4. CHECKPOINTING
# ======================================================================================
class Ckpt:
    """Results are saved after every (stage, seed); re-running the same command resumes where it stopped."""
    def __init__(self, root, cfg, fresh=False):
        h = hashlib.md5(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:8]
        self.dir = os.path.join(root, "checkpoints", h)
        if fresh and os.path.isdir(self.dir):
            shutil.rmtree(self.dir)
        os.makedirs(self.dir, exist_ok=True)

    def get(self, stage, seed, fn):
        p = os.path.join(self.dir, f"{stage}_s{seed}.pkl")
        if os.path.exists(p):
            print(f"[{stage}] seed {seed}: loaded from checkpoint")
            return joblib.load(p)
        out = fn()
        joblib.dump(out, p)
        print(f"[{stage}] seed {seed} done")
        return out


def collect(ck, stage, seeds, fn):
    parts = [ck.get(stage, s, lambda s=s: fn(s)) for s in range(seeds)]
    parts = [p for p in parts if p is not None and len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ======================================================================================
# 5. EVALUATION: within / cross / leave-one-dataset-out (+ native reference, few-shot)
# ======================================================================================
def split_views(views, seed):
    return {n: train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed) for n, (X, y, _) in views.items()}


def run_eval_seed(views, featset, seed, tuner, models):
    names = list(views)
    if len(names) < 2:
        return pd.DataFrame()
    sp, rows = split_views(views, seed), []
    for m in models:
        base = dict(featset=featset, seed=seed, model=m)
        for tr in names:
            Xtr, Xte, ytr, yte = sp[tr]
            model = fit(make_model(m, seed, tuner.params(m, (featset, tr), Xtr, ytr)), m, Xtr, ytr)
            rows.append({**base, "setting": "within", "train": tr, "test": tr, **metrics(model, Xte, yte)})
            for te in names:
                if te != tr:
                    rows.append({**base, "setting": "cross", "train": tr, "test": te,
                                 **metrics(model, views[te][0], views[te][1])})
        if len(names) >= 3:
            for held in names:
                others = [n for n in names if n != held]
                Xtr = np.vstack([sp[n][0] for n in others]); ytr = np.concatenate([sp[n][2] for n in others])
                model = fit(make_model(m, seed, tuner.params(m, (featset, "+".join(others)), Xtr, ytr)), m, Xtr, ytr)
                rows.append({**base, "setting": "lodo", "train": "+".join(others), "test": held,
                             **metrics(model, views[held][0], views[held][1])})
    return pd.DataFrame(rows)


def run_native_seed(nviews, seed, tuner):
    """Within-dataset reference using ALL native features of each dataset (no harmonisation)."""
    rows = []
    for n, (X, y, nf) in nviews.items():
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
        for m in MODEL_NAMES:
            model = fit(make_model(m, seed, tuner.params(m, ("native", n), Xtr, ytr)), m, Xtr, ytr)
            rows.append(dict(featset="native", seed=seed, model=m, setting="within", train=n, test=n,
                             n_features=nf, **metrics(model, Xte, yte)))
    return pd.DataFrame(rows)


def run_fewshot_seed(views, featset, seed, tuner, models):
    """Train on source train split + a small labelled slice of the target (repeated so it carries ~20% weight);
    evaluate on the REST of the target. frac=0 corresponds to the 'cross' rows."""
    names = list(views)
    if len(names) < 2:
        return pd.DataFrame()
    sp, rows = split_views(views, seed), []
    for tr in names:
        Xtr, _, ytr, _ = sp[tr]
        for te in names:
            if te == tr:
                continue
            Xt, yt, _ = views[te]
            for frac in FEWSHOT_FRACS:
                if len(yt) * frac < 20:
                    continue
                ia, ir = train_test_split(np.arange(len(yt)), train_size=frac, stratify=yt, random_state=seed)
                reps = int(min(50, max(1, round(0.2 * len(ytr) / len(ia)))))
                Xp = np.vstack([Xtr] + [Xt[ia]] * reps); yp = np.concatenate([ytr] + [yt[ia]] * reps)
                for m in models:
                    model = fit(make_model(m, seed, tuner.params(m, (featset, tr), Xtr, ytr)), m, Xp, yp)
                    rows.append(dict(featset=featset, seed=seed, model=m, setting="fewshot", train=tr, test=te,
                                     frac=frac, n_adapt=len(ia), **metrics(model, Xt[ir], yt[ir])))
    return pd.DataFrame(rows)


# ======================================================================================
# 6. SHAP (descriptive) + FOLD-WISE STABLE FEATURE SELECTION (no leakage)
# ======================================================================================
def shap_importance(model, name, X, Xbg):
    import shap
    if name == "XGB":  # native contributions: robust across xgboost/shap versions
        import xgboost as xgb
        return np.abs(model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]).mean(0)
    if name == "RF":
        sv = shap.TreeExplainer(model).shap_values(X)
        sv = sv[1] if isinstance(sv, list) else np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        return np.abs(sv).mean(0)
    # MLP: KernelExplainer (slow -> small samples); plain random background (k-means fails on binary features)
    bg = Xbg[np.random.default_rng(0).choice(len(Xbg), size=min(50, len(Xbg)), replace=False)]
    ex = shap.KernelExplainer(lambda z: model.predict_proba(z)[:, 1], bg)
    return np.abs(ex.shap_values(X[:100], nsamples=200)).mean(0)


def run_shap_descriptive(views_core, out, topk=3, seed=0, n_explain=2000):
    """All-dataset SHAP for the paper's explainability figures. NOT used for any reported test score."""
    names, rows = list(views_core), []
    for n in names:
        X, y, _ = views_core[n]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
        sub = Xte[np.random.default_rng(seed).choice(len(Xte), size=min(n_explain, len(Xte)), replace=False)]
        for m in SHAP_MODELS:
            model = fit(make_model(m, seed), m, Xtr, ytr)
            rows.append(pd.DataFrame({"dataset": n, "model": m, "feature": CORE,
                                      "mean_abs_shap": shap_importance(model, m, sub, Xtr[:500])}))
    imp = pd.concat(rows)
    imp["norm_importance"] = imp.groupby(["dataset", "model"])["mean_abs_shap"].transform(lambda s: s / (s.sum() or 1))
    imp.to_csv(os.path.join(out, "shap_importance.csv"), index=False)
    stab = []
    for m in SHAP_MODELS:
        for a, b in itertools.combinations(names, 2):
            ia = imp[(imp.dataset == a) & (imp.model == m)].set_index("feature").mean_abs_shap.reindex(CORE)
            ib = imp[(imp.dataset == b) & (imp.model == m)].set_index("feature").mean_abs_shap.reindex(CORE)
            ta, tb = set(ia.nlargest(topk).index), set(ib.nlargest(topk).index)
            stab.append(dict(model=m, dataset_a=a, dataset_b=b, spearman=spearmanr(ia, ib).correlation,
                             topk_overlap=len(ta & tb) / topk))
    pd.DataFrame(stab).to_csv(os.path.join(out, "shap_stability.csv"), index=False)
    cons = imp.groupby(["dataset", "feature"]).norm_importance.mean().unstack(0)
    stable = sorted(set.intersection(*[set(cons[d].nlargest(topk).index) for d in names]))
    if len(stable) < 2:
        stable = list(cons.rank(ascending=False).mean(axis=1).nsmallest(topk).index)
    json.dump({"topk": topk, "stable_features_descriptive_only": stable,
               "consensus_importance": cons.round(4).to_dict()},
              open(os.path.join(out, "stable_features.json"), "w"), indent=2)
    print("Descriptive stable features (all datasets):", stable)


def train_importance(Xtr, ytr, seed):
    """Normalised importance from TRAIN split only (mean of RF TreeSHAP and XGB contributions)."""
    sub = Xtr[np.random.default_rng(seed).choice(len(Xtr), size=min(2000, len(Xtr)), replace=False)]
    vs = []
    for m in ("RF", "XGB"):
        v = shap_importance(fit(make_model(m, seed), m, Xtr, ytr), m, sub, None)
        vs.append(v / (v.sum() or 1))
    return np.mean(vs, axis=0)


def select_stable(imps, train_names, topk):
    vec = pd.DataFrame({n: imps[n] for n in train_names}, index=CORE)
    if len(train_names) == 1:
        return list(vec.iloc[:, 0].nlargest(topk).index)
    st = sorted(set.intersection(*[set(vec[n].nlargest(topk).index) for n in train_names]))
    if len(st) < 2:
        st = list(vec.rank(ascending=False).mean(axis=1).nsmallest(topk).index)
    return st


def run_eval_stable_seed(views_core, seed, topk, tuner):
    """Per fold: features chosen using ONLY the training dataset(s); evaluated on the unseen one(s)."""
    names = list(views_core)
    if len(names) < 2:
        return pd.DataFrame(), pd.DataFrame()
    sp = split_views(views_core, seed)
    imps = {n: train_importance(sp[n][0], sp[n][2], seed) for n in names}
    rows, sels = [], []
    for m in MODEL_NAMES:
        base = dict(featset="stable_subset", seed=seed, model=m)
        for tr in names:
            feats = select_stable(imps, [tr], topk); idx = [CORE.index(f) for f in feats]
            Xtr, Xte, ytr, yte = sp[tr]
            model = fit(make_model(m, seed, tuner.params(m, ("stable", tr), Xtr[:, idx], ytr)), m, Xtr[:, idx], ytr)
            rows.append({**base, "setting": "within", "train": tr, "test": tr, **metrics(model, Xte[:, idx], yte)})
            if m == MODEL_NAMES[0]:
                sels.append(dict(seed=seed, train=tr, features=",".join(feats)))
            for te in names:
                if te != tr:
                    rows.append({**base, "setting": "cross", "train": tr, "test": te,
                                 **metrics(model, views_core[te][0][:, idx], views_core[te][1])})
        if len(names) >= 3:
            for held in names:
                others = [n for n in names if n != held]
                feats = select_stable(imps, others, topk); idx = [CORE.index(f) for f in feats]
                Xtr = np.vstack([sp[n][0] for n in others])[:, idx]; ytr = np.concatenate([sp[n][2] for n in others])
                model = fit(make_model(m, seed, tuner.params(m, ("stable", "+".join(others)), Xtr, ytr)), m, Xtr, ytr)
                rows.append({**base, "setting": "lodo", "train": "+".join(others), "test": held,
                             **metrics(model, views_core[held][0][:, idx], views_core[held][1])})
                if m == MODEL_NAMES[0]:
                    sels.append(dict(seed=seed, train="+".join(others), features=",".join(feats)))
    return pd.DataFrame(rows), pd.DataFrame(sels)


# ======================================================================================
# 7. MULTI-CLASS TASK (harmonised attack families)
# ======================================================================================
def mc_metrics(model, enc, X, fam):
    keep = np.isin(fam, enc.classes_)  # classes unseen in training cannot be predicted -> excluded, coverage reported
    if keep.sum() == 0:
        return None
    yt, pred = enc.transform(fam[keep]), model.predict(X[keep])
    return dict(accuracy=accuracy_score(yt, pred), balanced_acc=balanced_accuracy_score(yt, pred),
                macro_f1=f1_score(yt, pred, average="macro", labels=np.unique(yt), zero_division=0),
                weighted_f1=f1_score(yt, pred, average="weighted", zero_division=0),
                mcc=matthews_corrcoef(yt, pred), coverage=float(keep.mean()), n_classes_test=len(np.unique(yt)))


def shared_family_views(views, min_families=3):
    """Keep only rows whose attack family exists in EVERY dataset used (e.g. benign, dos_ddos, recon), so the classes
    mean the same thing across datasets. If all datasets together share fewer than `min_families` families
    (e.g. CSE-CIC-IDS2018 has no recon), the largest group of datasets that shares enough families is used."""
    names = [n for n in views if views[n][2] is not None]
    if len(names) < 2:
        return {}, []
    fams = {n: set(views[n][2]) for n in names}
    best = (names, sorted(set.intersection(*fams.values())))
    if len(best[1]) < min_families:
        for k in range(len(names) - 1, 1, -1):
            cands = [(list(c), sorted(set.intersection(*[fams[n] for n in c])))
                     for c in itertools.combinations(names, k)]
            cands = [c for c in cands if len(c[1]) >= min_families]
            if cands:
                # most shared families first; on a tie prefer groups with more IoT datasets (the paper's focus)
                best = max(cands, key=lambda c: (len(c[1]), sum("iot" in n.lower() for n in c[0]), c[0]))
                break
    names, shared = best
    out = {}
    for n in names:
        X, y, f = views[n]
        k = np.isin(f, shared)
        out[n] = (X[k], y[k], f[k])
    return out, shared


def run_eval_mc_seed(views, featset, seed):
    if PROFILE == "nf":
        views, shared = shared_family_views(views)
        print(f"  multi-class ({featset}): datasets {list(views)} share families {shared}")
        if len(shared) < 3:
            print("  fewer than 3 shared families -> multi-class skipped")
            return pd.DataFrame()
    names = [n for n in views if views[n][2] is not None]
    if len(names) < 2:
        return pd.DataFrame()

    def split(X, f):
        strat = f if pd.Series(f).value_counts().min() >= 2 else None
        return train_test_split(X, f, test_size=0.2, stratify=strat, random_state=seed)

    sp, rows = {n: split(views[n][0], views[n][2]) for n in names}, []
    for m in MODEL_NAMES:
        base = dict(featset=featset, seed=seed, model=m, task="multiclass")
        for tr in names:
            Xtr, Xte, ftr, fte = sp[tr]
            enc = LabelEncoder().fit(ftr)
            model = fit(make_model(m, seed), m, Xtr, enc.transform(ftr))
            r = mc_metrics(model, enc, Xte, fte)
            if r: rows.append({**base, "setting": "within", "train": tr, "test": tr, **r})
            for te in names:
                if te != tr:
                    r = mc_metrics(model, enc, views[te][0], views[te][2])
                    if r: rows.append({**base, "setting": "cross", "train": tr, "test": te, **r})
        if len(names) >= 3:
            for held in names:
                others = [n for n in names if n != held]
                Xtr = np.vstack([sp[n][0] for n in others]); ftr = np.concatenate([sp[n][2] for n in others])
                enc = LabelEncoder().fit(ftr)
                model = fit(make_model(m, seed), m, Xtr, enc.transform(ftr))
                r = mc_metrics(model, enc, views[held][0], views[held][2])
                if r: rows.append({**base, "setting": "lodo", "train": "+".join(others), "test": held, **r})
    return pd.DataFrame(rows)


# ======================================================================================
# 8. SUMMARIES AND STATISTICS
# ======================================================================================
METRICS = ["accuracy", "balanced_acc", "precision", "recall", "f1", "macro_f1", "mcc", "fpr", "roc_auc", "pr_auc"]


def ci95(s):
    s = s.dropna(); n = len(s)
    return stats.t.ppf(0.975, n - 1) * s.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan


def summarise(res, out, prefix="summary", metrics_=METRICS, extra_keys=()):
    cols = [c for c in metrics_ if c in res.columns]
    for (fs, setting), g in res.groupby(["featset", "setting"]):
        s = g.groupby(["model", "train", "test", *extra_keys])[cols].agg(["mean", "std", ci95])
        s.to_csv(os.path.join(out, f"{prefix}_{fs}_{setting}.csv"))


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def rank_biserial(d):
    """Matched-pairs rank-biserial correlation (effect size for the Wilcoxon signed-rank test, -1..1)."""
    d = np.asarray(pd.Series(d).dropna()); d = d[d != 0]
    if not len(d):
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def holm_by_family(df, keys):
    """Holm correction applied separately inside each pre-defined family of tests (one research question each),
    plus the very conservative correction over all tests together for reference."""
    ok = df.p_value.notna()
    df["p_holm"] = np.nan; df["p_holm_all"] = np.nan
    if ok.any():
        df.loc[ok, "p_holm_all"] = holm(df.loc[ok, "p_value"])
        for _, idx in df[ok].groupby(list(keys)).groups.items():
            df.loc[idx, "p_holm"] = holm(df.loc[idx, "p_value"])
        df["family"] = df[list(keys)].astype(str).agg(" | ".join, axis=1)
    return df


def stats_tests(res, out, metric="mcc"):
    """Seed-level paired Wilcoxon tests (metric averaged over dataset pairs per seed).
    Holm correction within each family: (a) model pairs, per feature set and setting (10 tests);
    (b) one feature-set variant vs core, per setting, across the 5 models (5 tests). p_holm_all = over everything."""
    rows = []
    sub = res[res.setting.isin(["cross", "lodo"])]
    per = sub.groupby(["featset", "setting", "model", "seed"])[metric].mean().reset_index()

    def test(d):
        d = d.dropna()
        if len(d) >= 6 and (d != 0).any():
            return len(d), d.mean(), *wilcoxon(d)
        return len(d), d.mean(), np.nan, np.nan

    for (fs, st), g in per.groupby(["featset", "setting"]):
        piv = g.pivot(index="seed", columns="model", values=metric)
        for a, b in itertools.combinations([m for m in MODEL_NAMES if m in piv], 2):
            n, md, stat, p = test(piv[a] - piv[b])
            rows.append(dict(featset=fs, setting=st, comparison=f"{a} vs {b}", metric=metric,
                             n_seeds=n, mean_diff=md, effect_r=rank_biserial(piv[a] - piv[b]), statistic=stat, p_value=p))
    for st in ["cross", "lodo"]:  # effect of feature-set variants
        for fa, fb in [("core", "stable_subset"), ("flow", "flow_dn"), ("core", "core_dn"), ("core", "core_noart"),
                       ("core", "shift_filter"), ("core", "shift_filter_dg")]:
            for m in MODEL_NAMES:
                a = per[(per.featset == fa) & (per.setting == st) & (per.model == m)].set_index("seed")[metric]
                b = per[(per.featset == fb) & (per.setting == st) & (per.model == m)].set_index("seed")[metric]
                if len(a) and len(b):
                    n, md, stat, p = test((b - a).dropna())
                    rows.append(dict(featset=f"{fb} vs {fa}", setting=st, comparison=m, metric=metric,
                                     n_seeds=n, mean_diff=md, effect_r=rank_biserial(b - a), statistic=stat, p_value=p))
    df = pd.DataFrame(rows)
    if len(df):
        df = holm_by_family(df, ["featset", "setting"])
        if not df.p_value.notna().any():
            print("NOTE: Wilcoxon needs >= 6 seeds; rerun with --seeds 8-10 for significance tests.")
    df.to_csv(os.path.join(out, "wilcoxon.csv"), index=False)


# ======================================================================================
# 9. EDGE SUITABILITY (PC) : size, latency, complexity
# ======================================================================================
def pooled_train(views, seed=0):
    Xs, ys = [], []
    for n, (X, y, _) in views.items():
        Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
        Xs.append(Xtr); ys.append(ytr)
    return np.vstack(Xs), np.concatenate(ys)


def edge_metrics(views, featset, out, tuner, seed=0, n_single=300):
    Xtr, ytr = pooled_train(views, seed)
    Xp = Xtr[np.random.default_rng(seed).choice(len(Xtr), size=min(len(Xtr), 5000), replace=False)]
    hw = f"{platform.processor() or platform.machine()} | {os.cpu_count()} cores | {platform.platform()}"
    rows = []
    for m in MODEL_NAMES:
        model = fit(make_model(m, seed, tuner.params(m, ("edge", featset), Xtr, ytr)), m, Xtr, ytr)
        cx = complexity(model)
        try:  # single-thread for a fair edge-like measurement
            model.set_params(n_jobs=1)
        except Exception:
            pass
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "model.joblib"); joblib.dump(model, p, compress=3)
            size_kb = os.path.getsize(p) / 1024
        model.predict(Xp[:10])
        lat = []
        for i in range(min(n_single, len(Xp))):
            t0 = time.perf_counter(); model.predict(Xp[i:i + 1]); lat.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter(); model.predict(Xp); batch_us = (time.perf_counter() - t0) / len(Xp) * 1e6
        tracemalloc.start(); model.predict(Xp[:1000]); _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        rows.append(dict(model=m, featset=featset, size_kb=round(size_kb, 1), latency_ms_mean=np.mean(lat),
                         latency_ms_p95=np.percentile(lat, 95), batch_us_per_sample=batch_us,
                         peak_py_mem_kb_1000=peak / 1024, **cx, hardware=hw))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(out, "edge_metrics.csv"), index=False)
    print(df.drop(columns="hardware").round(3).to_string(index=False))


# ======================================================================================
# 10. COMPACT MODELS -> C CODE (verified with gcc) + ESP32 BENCHMARK SKETCH
# ======================================================================================
def _f32_le(t):
    """Largest float32 <= t, so that (x <= t) == (x <= t32) for every float32 x."""
    t32 = np.float32(t)
    return np.nextafter(t32, np.float32(-np.inf)) if float(t32) > t else t32


def _cf(v):
    return "{:.9e}f".format(float(v))


def _arr(name, ctype, vals, fmt):
    body = ",".join(fmt(v) for v in vals) if len(vals) else "0"
    return f"static const {ctype} {name}[{max(len(vals), 1)}] = {{{body}}};"


def _c_trees(nm, trees):
    feat, thr, left, right, val, roots, off = [], [], [], [], [], [], 0
    for f, t, l, r, p in trees:
        roots.append(off)
        feat += [-1 if x < 0 else int(x) for x in f]
        thr += [_f32_le(x) if fx >= 0 else 0.0 for x, fx in zip(t, f)]
        left += [int(x) + off if x >= 0 else -1 for x in l]
        right += [int(x) + off if x >= 0 else -1 for x in r]
        val += list(p); off += len(f)
    if off > 32000:
        raise ValueError(f"{nm}: too many nodes ({off}) for int16 indices")
    nbytes = off * (1 + 4 + 2 + 2 + 4) + 2 * len(roots)
    code = [_arr(f"{nm}_feat", "int8_t", feat, str), _arr(f"{nm}_thr", "float", thr, _cf),
            _arr(f"{nm}_left", "int16_t", left, str), _arr(f"{nm}_right", "int16_t", right, str),
            _arr(f"{nm}_val", "float", val, _cf), _arr(f"{nm}_root", "uint16_t", roots, str),
            f"int predict_{nm}(const float *x) {{",
            "  float acc = 0.0f;",
            f"  for (int t = 0; t < {len(roots)}; t++) {{",
            f"    int n = {nm}_root[t];",
            f"    while ({nm}_feat[n] >= 0) n = (x[{nm}_feat[n]] <= {nm}_thr[n]) ? {nm}_left[n] : {nm}_right[n];",
            f"    acc += {nm}_val[n];",
            "  }",
            f"  return (acc / {len(roots)}.0f) > 0.5f;",
            "}"]
    return "\n".join(code), nbytes


def _tree_arrays(tree):
    v = tree.value[:, 0, :]
    p1 = v[:, 1] / np.maximum(v.sum(axis=1), 1e-12)
    return tree.feature, tree.threshold, tree.children_left, tree.children_right, p1


def _c_linear(nm, model):
    sc, lr = model.named_steps["standardscaler"], model.named_steps["logisticregression"]
    w = lr.coef_[0] / sc.scale_
    b = lr.intercept_[0] - np.sum(lr.coef_[0] * sc.mean_ / sc.scale_)
    code = [_arr(f"{nm}_w", "float", w, _cf), f"static const float {nm}_b = {_cf(b)};",
            f"int predict_{nm}(const float *x) {{", f"  float s = {nm}_b;",
            f"  for (int i = 0; i < {len(w)}; i++) s += {nm}_w[i] * x[i];", "  return s > 0.0f;", "}"]
    return "\n".join(code), 4 * (len(w) + 1)


def _c_mlp(nm, model):
    sc, mlp = model.named_steps["standardscaler"], model.named_steps["mlpclassifier"]
    W = [c.astype(float).copy() for c in mlp.coefs_]; B = [b.astype(float).copy() for b in mlp.intercepts_]
    B[0] = B[0] - (sc.mean_ / sc.scale_) @ W[0]
    W[0] = W[0] / sc.scale_[:, None]
    sizes = [W[0].shape[0]] + [w.shape[1] for w in W]
    code, nbytes = [], 0
    for l, (w, b) in enumerate(zip(W, B)):
        code += [_arr(f"{nm}_W{l}", "float", w.ravel(), _cf), _arr(f"{nm}_b{l}", "float", b, _cf)]
        nbytes += 4 * (w.size + b.size)
    code.append(f"int predict_{nm}(const float *x) {{")
    src = "x"
    for l in range(len(W) - 1):
        ni, no = sizes[l], sizes[l + 1]
        code += [f"  float h{l}[{no}];",
                 f"  for (int j = 0; j < {no}; j++) {{ float s = {nm}_b{l}[j];",
                 f"    for (int i = 0; i < {ni}; i++) s += {src}[i] * {nm}_W{l}[i * {no} + j];",
                 f"    h{l}[j] = s > 0.0f ? s : 0.0f; }}"]
        src = f"h{l}"
    L = len(W) - 1; ni = sizes[L]
    code += [f"  float z = {nm}_b{L}[0];", f"  for (int i = 0; i < {ni}; i++) z += {src}[i] * {nm}_W{L}[i];",
             "  return z > 0.0f;", "}"]
    return "\n".join(code), nbytes


def model_to_c(nm, model):
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    if isinstance(est, DecisionTreeClassifier):
        return _c_trees(nm, [_tree_arrays(est.tree_)])
    if isinstance(est, RandomForestClassifier):
        return _c_trees(nm, [_tree_arrays(e.tree_) for e in est.estimators_])
    if isinstance(est, LogisticRegression):
        return _c_linear(nm, model)
    if isinstance(est, MLPClassifier):
        return _c_mlp(nm, model)
    raise ValueError(f"cannot export {nm}")


SKETCH = r'''// ESP32 benchmark of compact IoT-IDS models (generated). Board: ESP32 Dev Module. Serial Monitor: 115200 baud.
#include <Arduino.h>
#include "test_data.h"
#include "models.h"
#define REPS 20
void setup() {
  Serial.begin(115200);
  delay(2000);
  Serial.printf("INFO,chip=%s,rev=%d,cpu_mhz=%d,flash_kb=%u,free_heap_b=%u\n", ESP.getChipModel(),
                (int)ESP.getChipRevision(), (int)getCpuFrequencyMhz(), (unsigned)(ESP.getFlashChipSize() / 1024),
                (unsigned)ESP.getFreeHeap());
  Serial.println("RESULT_HEADER,model,data_bytes,us_per_pred,agree,n,heap_before,heap_after,min_free_heap");
  for (int m = 0; m < N_MODELS; m++) {
    uint32_t heap0 = ESP.getFreeHeap();
    int agree = 0;
    volatile int sink = 0;
    for (int i = 0; i < N_TEST; i++) agree += (MODELS[m].fn(X_TEST[i]) == MODELS[m].pc_pred[i]);
    uint32_t t0 = micros();
    for (int r = 0; r < REPS; r++)
      for (int i = 0; i < N_TEST; i++) sink = sink + MODELS[m].fn(X_TEST[i]);
    uint32_t dt = micros() - t0;
    uint32_t heap1 = ESP.getFreeHeap();
    Serial.printf("RESULT,%s,%u,%.3f,%d,%d,%u,%u,%u\n", MODELS[m].name, (unsigned)MODELS[m].data_bytes,
                  (double)dt / ((double)REPS * N_TEST), agree, N_TEST, (unsigned)heap0, (unsigned)heap1,
                  (unsigned)ESP.getMinFreeHeap());
  }
  Serial.println("DONE");
}
void loop() {}
'''

C_HARNESS = r'''#include <stdio.h>
#include <stdint.h>
#include "test_data.h"
#include "models.h"
int main(void) {
  for (int m = 0; m < N_MODELS; m++) {
    int agree = 0;
    for (int i = 0; i < N_TEST; i++) agree += (MODELS[m].fn(X_TEST[i]) == MODELS[m].pc_pred[i]);
    printf("%s,%d,%d\n", MODELS[m].name, agree, N_TEST);
  }
  return 0;
}
'''


def export_esp32(views, featset, out, seed=0, n_test=500):
    feats = CORE + FLOW_USED if featset.startswith("flow") else CORE
    Xtr, ytr = pooled_train(views, seed)
    Xte = np.vstack([train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)[1] for X, y, _ in views.values()])
    yte = np.concatenate([train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)[3] for X, y, _ in views.values()])
    sel = train_test_split(np.arange(len(yte)), train_size=min(n_test, len(yte) - 2), stratify=yte, random_state=seed)[0]
    X32, y32 = Xte[sel].astype(np.float32), yte[sel]
    d = os.path.join(out, "esp32"); sk = os.path.join(d, "esp32_benchmark"); vf = os.path.join(d, "verify")
    for p in (sk, vf):
        os.makedirs(p, exist_ok=True)
    funcs, entries, preds, meta = [], [], {}, []
    for name in COMPACT:
        model = fit(make_model(name, seed), name, Xtr, ytr)
        code, nbytes = model_to_c(name, model)
        pred = model.predict(X32).astype(int)
        preds[name] = pred; funcs.append(code)
        entries.append(f'  {{"{name}", predict_{name}, {nbytes}, PC_PRED_{name}}}')
        meta.append(dict(model=name, data_bytes=nbytes, pc_pred_attack_share=float(pred.mean()), **complexity(model)))
    nf = len(feats)
    td = ["#include <stdint.h>", f"#define N_TEST {len(y32)}", f"#define N_FEAT {nf}",
          f"// features: {','.join(feats)}", f"static const float X_TEST[N_TEST][N_FEAT] = {{"]
    td += ["  {" + ",".join(_cf(v) for v in row) + "}," for row in X32]
    td += ["};", _arr("Y_TEST", "uint8_t", y32, str)]
    td += [_arr(f"PC_PRED_{n}", "uint8_t", p, str) for n, p in preds.items()]
    mh = ["#include <stdint.h>", "typedef int (*pred_fn_t)(const float *);",
          "typedef struct { const char *name; pred_fn_t fn; uint32_t data_bytes; const uint8_t *pc_pred; } model_entry_t;"]
    mh += funcs
    mh += ["static const model_entry_t MODELS[] = {", ",\n".join(entries), "};",
           "#define N_MODELS ((int)(sizeof(MODELS) / sizeof(MODELS[0])))"]
    open(os.path.join(sk, "test_data.h"), "w").write("\n".join(td) + "\n")
    open(os.path.join(sk, "models.h"), "w").write("\n".join(mh) + "\n")
    open(os.path.join(sk, "esp32_benchmark.ino"), "w").write(SKETCH)
    open(os.path.join(vf, "test_c.c"), "w").write(C_HARNESS)
    meta = pd.DataFrame(meta); meta.to_csv(os.path.join(d, "compact_models_meta.csv"), index=False)
    # verify generated C against sklearn predictions
    ver = None
    if shutil.which("gcc"):
        exe = os.path.join(vf, "test_c")
        r = subprocess.run(["gcc", "-O2", "-I", sk, "-o", exe, os.path.join(vf, "test_c.c")], capture_output=True, text=True)
        if r.returncode == 0:
            o = subprocess.run([exe], capture_output=True, text=True).stdout.strip().splitlines()
            ver = pd.DataFrame([l.split(",") for l in o], columns=["model", "agree", "n"]).astype({"agree": int, "n": int})
            ver["agreement_pct"] = 100 * ver.agree / ver.n
            ver.to_csv(os.path.join(d, "c_vs_sklearn_agreement.csv"), index=False)
            print("C code vs sklearn agreement (PC, gcc):\n", ver.round(2).to_string(index=False))
        else:
            print("gcc failed:\n", r.stderr[:800])
    else:
        print("gcc not found: C-code verification skipped (install build-essential).")
    print(meta.to_string(index=False))
    print(f"ESP32 files written to {sk}  (open esp32_benchmark.ino in Arduino IDE)")
    return meta


def compact_report(res_full, res_compact, meta, featset, out):
    """Compact vs full models under the same protocol: MCC retained, size, complexity."""
    def mean_mcc(df, model, settings):
        return df[(df.model == model) & df.setting.isin(settings)].groupby("seed").mcc.mean().mean()
    rows = []
    rf_w, rf_c = mean_mcc(res_full[res_full.featset == featset], "RF", ["within"]), \
        mean_mcc(res_full[res_full.featset == featset], "RF", ["cross", "lodo"])
    for _, r in meta.iterrows():
        w, c = mean_mcc(res_compact, r.model, ["within"]), mean_mcc(res_compact, r.model, ["cross", "lodo"])
        rows.append(dict(model=r.model, mcc_within=w, mcc_cross_lodo=c, full_RF_mcc_within=rf_w,
                         full_RF_mcc_cross_lodo=rf_c, pct_of_full_RF_within=100 * w / rf_w if rf_w else np.nan, mcc_cross_lodo_diff_vs_full_RF=c - rf_c,
                         data_bytes=r.data_bytes, n_trees=r.n_trees, total_nodes=r.total_nodes,
                         max_depth=r.max_depth, n_params=r.n_params, max_ops=r.max_ops))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(out, "compact_vs_full.csv"), index=False)
    print(df.round(3).to_string(index=False))

# ======================================================================================
# 11. v6 ADDITIONS: shift-filtered features (method), extended few-shot, feature shift, prevalence sensitivity
# ======================================================================================
SHIFT_K = 8                                   # features dropped by the shift filter (fixed a priori, ~20% of 38)
FEWSHOT2_FRACS = [0.005, 0.01, 0.05]          # labelled share of the target used for adaptation
FEWSHOT_COMPACT = ["DT_d4", "DT_d6", "RF_t5_d6", "MLP_8"]   # microcontroller-sized models also tested in few-shot
FEWSHOT_NOART_MODELS = ["RF", "XGB", "DT", "DT_d6"]          # few-shot without testbed-artifact features
# Share of attack flows in the FULL datasets (counted in Step 4a; published values agree). Overridden by
# <nf_dir>/<name>_counts.json when prepare_nf_samples.py wrote one.
NATURAL_ATTACK_SHARE = {"NF-ToN-IoT-v2": 0.63995, "NF-BoT-IoT-v2": 0.99642, "NF-UNSW-NB15-v2": 0.03977,
                        "NF-CSE-CIC-IDS2018-v2": 0.11952}


def ks_shift(A, B, seed=0, n=5000):
    """Per-feature two-sample Kolmogorov-Smirnov statistic (0 = same distribution, 1 = no overlap)."""
    from scipy.stats import ks_2samp
    rng = np.random.default_rng(seed)
    a = A[rng.choice(len(A), size=min(n, len(A)), replace=False)]
    b = B[rng.choice(len(B), size=min(n, len(B)), replace=False)]
    return np.array([ks_2samp(a[:, j], b[:, j]).statistic for j in range(A.shape[1])])


def keep_after_drop(score, k):
    """Indices of the features kept after dropping the k most shifted ones (ties broken by feature order)."""
    drop = set(np.argsort(-np.asarray(score), kind="stable")[:k])
    return [j for j in range(len(score)) if j not in drop]


def run_eval_shift_seed(views, seed, k, tuner, models):
    """Shift-filtered features, chosen per fold WITHOUT any target labels:
       shift_filter    : drop the k features whose distribution differs most between the source training data and
                         the UNLABELLED target traffic (cross + leave-one-dataset-out).
       shift_filter_dg : domain generalisation, no target data at all: drop the k features whose BENIGN-traffic
                         distribution differs most between the training datasets (leave-one-dataset-out only)."""
    names = list(views)
    if len(names) < 2:
        return pd.DataFrame(), pd.DataFrame()
    sp, rows, sels = split_views(views, seed), [], []
    plan = []  # (featset, setting, train names, test name, kept idx)
    for tr in names:
        for te in names:
            if te != tr:
                idx = keep_after_drop(ks_shift(sp[tr][0], views[te][0], seed), k)
                plan.append(("shift_filter", "cross", [tr], te, idx))
    if len(names) >= 3:
        for held in names:
            others = [n for n in names if n != held]
            Xo = np.vstack([sp[n][0] for n in others])
            plan.append(("shift_filter", "lodo", others, held, keep_after_drop(ks_shift(Xo, views[held][0], seed), k)))
            ben = {n: sp[n][0][sp[n][2] == 0] for n in others}
            score = np.max([ks_shift(ben[a], ben[b], seed) for a, b in itertools.combinations(others, 2)], axis=0)
            plan.append(("shift_filter_dg", "lodo", others, held, keep_after_drop(score, k)))
    for fs, st, trs, te, idx in plan:
        sels.append(dict(featset=fs, setting=st, seed=seed, train="+".join(trs), test=te,
                         dropped=",".join(CORE[j] for j in range(len(CORE)) if j not in idx)))
    for m in models:
        for fs, st, trs, te, idx in plan:
            Xtr = np.vstack([sp[n][0] for n in trs]); ytr = np.concatenate([sp[n][2] for n in trs])
            key = ("core", trs[0]) if len(trs) == 1 else ("core", "+".join(trs))
            model = fit(make_model(m, seed, tuner.params(m, key, Xtr, ytr)), m, Xtr[:, idx], ytr)
            rows.append(dict(featset=fs, seed=seed, model=m, setting=st, train="+".join(trs), test=te,
                             **metrics(model, views[te][0][:, idx], views[te][1])))
    return pd.DataFrame(rows), pd.DataFrame(sels)


def run_fewshot2_seed(views, featset, seed, tuner, models):
    """Few-shot adaptation with its control:
       fewshot    : source training data + a small labelled slice of the target (repeated so it carries ~20% weight)
       targetonly : the SAME labelled slice alone, no source data (tells whether the source data helps at all)
       Both are evaluated on the rest of the target. frac=0 corresponds to the 'cross' rows."""
    names = list(views)
    if len(names) < 2:
        return pd.DataFrame()
    sp, rows, slices = split_views(views, seed), [], {}
    for te in names:
        Xt, yt, _ = views[te]
        for frac in FEWSHOT2_FRACS:
            n_a = int(round(len(yt) * frac))
            if n_a < 20 or np.bincount(yt, minlength=2).min() < 4:
                continue
            ia, ir = train_test_split(np.arange(len(yt)), train_size=n_a, stratify=yt, random_state=seed)
            if len(np.unique(yt[ia])) == 2:
                slices[(te, frac)] = (ia, ir)
    for (te, frac), (ia, ir) in slices.items():
        Xt, yt, _ = views[te]
        for m in models:  # target-only control (default hyper-parameters: too little data to tune)
            model = fit(make_model(m, seed), m, Xt[ia], yt[ia])
            rows.append(dict(featset=featset, seed=seed, model=m, setting="targetonly", train="(target only)",
                             test=te, frac=frac, n_adapt=len(ia), **metrics(model, Xt[ir], yt[ir])))
        for tr in names:
            if tr == te:
                continue
            Xtr, _, ytr, _ = sp[tr]
            reps = int(min(50, max(1, round(0.2 * len(ytr) / len(ia)))))
            Xp = np.vstack([Xtr] + [Xt[ia]] * reps); yp = np.concatenate([ytr] + [yt[ia]] * reps)
            for m in models:
                params = tuner.params(m, (featset, tr), Xtr, ytr) if m in MODEL_NAMES else {}
                model = fit(make_model(m, seed, params), m, Xp, yp)
                rows.append(dict(featset=featset, seed=seed, model=m, setting="fewshot", train=tr, test=te,
                                 frac=frac, n_adapt=len(ia), **metrics(model, Xt[ir], yt[ir])))
    return pd.DataFrame(rows)


def run_feature_shift(views, out, seed=0):
    """Descriptive, model-free: how differently each feature is distributed across datasets (KS), separately for
    benign and attack traffic. Testbed artifacts should shift strongly even in BENIGN traffic."""
    names, rows = list(views), []
    for a, b in itertools.combinations(names, 2):
        for cls, lab in ((0, "benign"), (1, "attack")):
            A, B = views[a][0][views[a][1] == cls], views[b][0][views[b][1] == cls]
            if len(A) < 20 or len(B) < 20:
                continue
            for f, v in zip(CORE, ks_shift(A, B, seed)):
                rows.append(dict(dataset_a=a, dataset_b=b, traffic=lab, feature=f, ks=v))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(out, "feature_shift_pairs.csv"), index=False)
    summ = df.pivot_table(index="feature", columns="traffic", values="ks", aggfunc="mean")
    summ["artifact"] = summ.index.isin(ARTIFACT_FEATURES)
    summ = summ.sort_values("benign", ascending=False)
    summ.to_csv(os.path.join(out, "feature_shift.csv"))
    print("Most shifted features in BENIGN traffic (mean pairwise KS):\n", summ.head(10).round(3).to_string())


def natural_shares(nf_dir):
    sh = dict(NATURAL_ATTACK_SHARE)
    for f in glob.glob(os.path.join(nf_dir or "", "*_counts.json")):
        try:
            d = json.load(open(f)); sh[os.path.basename(f).split("_counts")[0]] = d["attack"] / (d["attack"] + d["benign"])
        except Exception:
            pass
    return sh


def mcc_at(r, f, pi):
    """MCC and precision implied by detection rate r (recall) and false-positive rate f at attack share pi."""
    tp, fn, fp, tn = pi * r, pi * (1 - r), (1 - pi) * f, (1 - pi) * (1 - f)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / den if den > 0 else 0.0, tp / (tp + fp) if tp + fp > 0 else 0.0


def prevalence_sensitivity(df, out, shares, grid=(0.01, 0.1, 0.5, 0.9)):
    """The experiments use ~50% attack traffic. Recall, FPR, balanced accuracy and ROC-AUC do not depend on that
    share; MCC and precision do. This re-expresses each result at the target's REAL attack share and on a grid.
    (Assumes per-class rates carry over; attack types inside the attack class were equalised by sampling.)"""
    rows = []
    for _, r in df.iterrows():
        pis = {"natural": shares.get(r["test"], np.nan), **{f"pi_{g}": g for g in grid}}
        d = dict(featset=r["featset"], setting=r["setting"], model=r["model"], train=r["train"], test=r["test"],
                 seed=r["seed"], frac=r.get("frac", np.nan), mcc_exp=r["mcc"], balanced_acc=r["balanced_acc"],
                 recall=r["recall"], fpr=r["fpr"], natural_attack_share=pis["natural"])
        for k_, pi in pis.items():
            if not np.isnan(pi):
                d[f"mcc_{k_}"], d[f"precision_{k_}"] = mcc_at(r["recall"], r["fpr"], pi)
        rows.append(d)
    p = pd.DataFrame(rows); p.to_csv(os.path.join(out, "prevalence_sensitivity.csv"), index=False)
    cols = [c for c in p.columns if c.startswith(("mcc_", "precision_natural"))] + ["balanced_acc"]
    p.groupby(["featset", "setting", "model"], dropna=False)[cols].mean().round(4).to_csv(
        os.path.join(out, "prevalence_sensitivity_summary.csv"))


def fewshot_tests(fs, out, metric="mcc"):
    """Few-shot (source + target labels) vs target-only, paired by seed, per model / feature set / fraction."""
    rows = []
    per = fs.groupby(["featset", "setting", "model", "frac", "seed"])[metric].mean().reset_index()
    for (fset, m, frac), g in per.groupby(["featset", "model", "frac"]):
        piv = g.pivot(index="seed", columns="setting", values=metric)
        if {"fewshot", "targetonly"} <= set(piv.columns):
            d = (piv["fewshot"] - piv["targetonly"]).dropna()
            stat, pv = (wilcoxon(d) if len(d) >= 6 and (d != 0).any() else (np.nan, np.nan))
            rows.append(dict(featset=fset, model=m, frac=frac, n_seeds=len(d), fewshot=piv["fewshot"].mean(),
                             targetonly=piv["targetonly"].mean(), mean_diff=d.mean(), effect_r=rank_biserial(d),
                             statistic=stat, p_value=pv))
    df = pd.DataFrame(rows)
    if len(df):
        df = holm_by_family(df, ["featset", "frac"])  # family: all models at one feature set and label fraction
    df.to_csv(os.path.join(out, "wilcoxon_fewshot_vs_targetonly.csv"), index=False)


# ======================================================================================
# 12. v7: REALISTIC LABEL BUDGET (fixed number of labelled flows drawn at the target's REAL attack share)
# ======================================================================================
BUDGETS = [50, 100, 200, 500, 1000]           # labelled target flows
BUDGET_WEIGHTS = [0.2, 0.5, 0.8]              # share of total training weight given to the labelled target flows
BUDGET_MODELS = ["RF", "XGB", "MLP_8", "DT_d6", "DT_d4"]
BUDGET_SOURCE_MAX = 20000                     # pooled source rows used (stratified subsample, keeps run time bounded)


def fit_weighted(model, name, X, y, w):
    """Fit with per-row weights (domain weighting). Uses sample_weight where the estimator supports it,
    otherwise falls back to replicating rows in proportion to their weight."""
    import inspect
    if name == "XGB":
        return model.fit(X, y, sample_weight=compute_sample_weight("balanced", y) * w)
    last = model.steps[-1] if hasattr(model, "steps") else None
    est = last[1] if last else model
    if "sample_weight" in inspect.signature(est.fit).parameters:
        kw = {f"{last[0]}__sample_weight": w} if last else {"sample_weight": w}
        return model.fit(X, y, **kw)
    reps = np.maximum(1, np.round(w / w.min())).astype(int)
    reps = np.minimum(reps, max(1, int(4 * len(y) / max(1, (reps > 1).sum()))))  # memory guard
    return model.fit(np.repeat(X, reps, axis=0), np.repeat(y, reps))


def draw_budget(yt, n, share, rng):
    """Indices of n labelled target flows whose class mix follows a random draw at attack share `share`
    (binomial), i.e. what an operator labelling n random flows of this network would get."""
    att, ben = np.flatnonzero(yt == 1), np.flatnonzero(yt == 0)
    n_att = int(min(len(att), rng.binomial(n, share)))
    n_ben = int(min(len(ben), n - n_att))
    return np.concatenate([rng.choice(att, n_att, replace=False), rng.choice(ben, n_ben, replace=False)])


def const_metrics(y, cls):
    pred = np.full(len(y), cls); return binary_metrics(y, pred, pred.astype(float))


def run_budget_seed(views, seed, tuner, shares, models=None):
    """For every target network: pool the OTHER datasets as source, draw a fixed number of labelled target flows
    at the target's real attack share, and compare
       targetonly          : labelled target flows only
       targetonly_balanced : same number of flows but drawn 50/50 (the optimistic setting of the fraction runs)
       fewshot_w{w}        : pooled source + labelled flows carrying a share w of the total training weight
    All are tested on the remaining target flows. Also records MCC/precision implied at the real attack share."""
    models = models or BUDGET_MODELS
    names = list(views)
    if len(names) < 3:
        return pd.DataFrame()
    sp, rows = split_views(views, seed), []
    for ti, te in enumerate(names):
        Xt, yt, _ = views[te]
        pi = shares.get(te, float(np.mean(yt)))
        others = [n for n in names if n != te]
        Xs = np.vstack([sp[n][0] for n in others]); ys = np.concatenate([sp[n][2] for n in others])
        if len(ys) > BUDGET_SOURCE_MAX:
            keep, _ = train_test_split(np.arange(len(ys)), train_size=BUDGET_SOURCE_MAX, stratify=ys, random_state=seed)
            Xs, ys = Xs[keep], ys[keep]
        for n in BUDGETS:
            if n > 0.5 * len(yt):
                continue
            rng = np.random.default_rng([seed, ti, n])
            ia = draw_budget(yt, n, pi, rng)
            ib = draw_budget(yt, n, 0.5, rng)
            test = np.setdiff1d(np.arange(len(yt)), np.union1d(ia, ib))
            Xr, yr = Xt[test], yt[test]
            base = dict(featset="core", seed=seed, test=te, budget=n, natural_share=pi)
            for m in models:
                for st, idx in (("targetonly", ia), ("targetonly_balanced", ib)):
                    cls = np.unique(yt[idx])
                    if len(cls) < 2:  # only one class among the labelled flows -> a constant detector
                        met = const_metrics(yr, cls[0] if len(cls) else 0); single = True
                    else:
                        mdl = make_model(m, seed)
                        if np.bincount(yt[idx]).min() < 10 and hasattr(mdl, "named_steps") and "mlpclassifier" in mdl.named_steps:
                            mdl.set_params(mlpclassifier__early_stopping=False)  # too few labels for a validation split
                        met = metrics(fit(mdl, m, Xt[idx], yt[idx]), Xr, yr); single = False
                    rows.append(dict(**base, model=m, setting=st, train="(target only)", n_attack_labels=int(yt[idx].sum()),
                                     single_class=single, **met))
                params = tuner.params(m, ("core", "+".join(others)), Xs, ys) if m in MODEL_NAMES else {}
                for wt in BUDGET_WEIGHTS:
                    Xp = np.vstack([Xs, Xt[ia]]); yp = np.concatenate([ys, yt[ia]])
                    w = np.concatenate([np.full(len(ys), (1 - wt) / len(ys)), np.full(len(ia), wt / max(1, len(ia)))])
                    w = w / w.mean()
                    model = fit_weighted(make_model(m, seed, params), m, Xp, yp, w)
                    rows.append(dict(**base, model=m, setting=f"fewshot_w{wt}", train="+".join(others),
                                     n_attack_labels=int(yt[ia].sum()), single_class=False, **metrics(model, Xr, yr)))
            # zero labels: pooled-source model only (leave-one-dataset-out on the same test flows)
            for m in models:
                params = tuner.params(m, ("core", "+".join(others)), Xs, ys) if m in MODEL_NAMES else {}
                model = fit(make_model(m, seed, params), m, Xs, ys)
                rows.append(dict(**base, model=m, setting="source_only", train="+".join(others), n_attack_labels=0,
                                 single_class=False, **metrics(model, Xr, yr)))
    df = pd.DataFrame(rows)
    if len(df):
        nat = df.apply(lambda r: mcc_at(r["recall"], r["fpr"], r["natural_share"]), axis=1)
        df["mcc_natural"] = [a for a, _ in nat]; df["precision_natural"] = [b for _, b in nat]
    return df


def budget_tests(bd, out, metric="mcc"):
    """Per budget: each fewshot weighting and source_only vs targetonly, paired by seed (Holm within budget)."""
    rows = []
    per = bd.groupby(["setting", "model", "budget", "seed"])[metric].mean().reset_index()
    for (m, n), g in per.groupby(["model", "budget"]):
        piv = g.pivot(index="seed", columns="setting", values=metric)
        if "targetonly" not in piv:
            continue
        for st in [c for c in piv.columns if c.startswith("fewshot") or c in ("source_only", "targetonly_balanced")]:
            d = (piv[st] - piv["targetonly"]).dropna()
            stat, pv = (wilcoxon(d) if len(d) >= 6 and (d != 0).any() else (np.nan, np.nan))
            rows.append(dict(model=m, budget=n, comparison=f"{st} vs targetonly", n_seeds=len(d), mean_diff=d.mean(),
                             effect_r=rank_biserial(d), statistic=stat, p_value=pv))
    df = pd.DataFrame(rows)
    if len(df):
        df = holm_by_family(df, ["budget", "comparison"])
    df.to_csv(os.path.join(out, "wilcoxon_label_budget.csv"), index=False)


# ======================================================================================
# 13. v8: STRONGER BASELINES UNDER THE REALISTIC LABEL BUDGET (target labels only)
# ======================================================================================
# Same label draws and test flows as budget_core (same random generator), so results pair with it seed by seed.
TUNED_GRIDS = {
    "RF_cv": {"max_depth": [4, 6, 8, None], "min_samples_leaf": [1, 5], "class_weight": ["balanced", "balanced_subsample"]},
    "XGB_cv": {"max_depth": [2, 3, 4, 6], "learning_rate": [0.1, 0.3]},
    "DT_cv": {"max_depth": [2, 3, 4, 6, 8, None], "min_samples_leaf": [1, 5]},
}
TUNED_FIXED = ["RF_bal_d6", "RF_bal_d12", "XGB_d3"]
TUNED_MODELS = TUNED_FIXED + list(TUNED_GRIDS)


def make_tuned(name, seed):
    if name in ("RF_bal_d6", "RF_cv"):
        return RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=1, class_weight="balanced_subsample",
                                      n_jobs=1, random_state=seed)
    if name == "RF_bal_d12":
        return RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=5, class_weight="balanced_subsample",
                                      n_jobs=1, random_state=seed)
    if name in ("XGB_d3", "XGB_cv"):
        return XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, tree_method="hist", n_jobs=1,
                             random_state=seed, eval_metric="logloss")
    if name == "DT_cv":
        return DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, class_weight="balanced", random_state=seed)
    raise ValueError(name)


def fit_tuned(name, seed, X, y):
    """Fixed variants are fitted directly. *_cv models pick hyper-parameters by 3-fold stratified CV (MCC) on the
    labelled target flows themselves; with fewer than 3 flows of a class, CV is impossible and the default is used."""
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.metrics import make_scorer, matthews_corrcoef
    m = make_tuned(name, seed)
    sw = compute_sample_weight("balanced", y) if name.startswith("XGB") else None
    best = {}
    if name in TUNED_GRIDS and np.bincount(y, minlength=2).min() >= 3:
        gs = GridSearchCV(m, TUNED_GRIDS[name], cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                          scoring=make_scorer(matthews_corrcoef), n_jobs=-1, refit=True, error_score=0.0)
        gs.fit(X, y, **({"sample_weight": sw} if sw is not None else {}))
        return gs.best_estimator_, gs.best_params_
    m.fit(X, y, **({"sample_weight": sw} if sw is not None else {}))
    return m, best


def run_budget_tuned_seed(views, seed, shares, models=None):
    models = models or TUNED_MODELS
    names = list(views)
    if len(names) < 3:
        return pd.DataFrame()
    rows = []
    for ti, te in enumerate(names):
        Xt, yt, _ = views[te]
        pi = shares.get(te, float(np.mean(yt)))
        for n in BUDGETS:
            if n > 0.5 * len(yt):
                continue
            rng = np.random.default_rng([seed, ti, n])       # identical draws to run_budget_seed
            ia = draw_budget(yt, n, pi, rng)
            ib = draw_budget(yt, n, 0.5, rng)
            test = np.setdiff1d(np.arange(len(yt)), np.union1d(ia, ib))
            Xr, yr = Xt[test], yt[test]
            cls = np.unique(yt[ia])
            for m in models:
                if len(cls) < 2:
                    met, prm = const_metrics(yr, cls[0] if len(cls) else 0), {}
                else:
                    mdl, prm = fit_tuned(m, seed, Xt[ia], yt[ia]); met = metrics(mdl, Xr, yr)
                rows.append(dict(featset="core", seed=seed, test=te, budget=n, natural_share=pi, model=m,
                                 setting="targetonly", train="(target only)", n_attack_labels=int(yt[ia].sum()),
                                 single_class=len(cls) < 2, best_params=json.dumps(prm, default=str), **met))
    df = pd.DataFrame(rows)
    if len(df):
        nat = df.apply(lambda r: mcc_at(r["recall"], r["fpr"], r["natural_share"]), axis=1)
        df["mcc_natural"] = [a for a, _ in nat]; df["precision_natural"] = [b for _, b in nat]
    return df


def budget_tuned_tests(bt, bd, out, refs=("DT_d4", "DT_d6", "RF"), metric="mcc"):
    """Each stronger baseline vs the fixed small trees (and default RF), target-only, paired by seed
    (metric averaged over the 4 target networks); Holm within each budget and reference model."""
    base = bd[bd.setting == "targetonly"]
    allm = pd.concat([base, bt], ignore_index=True)
    per = allm.groupby(["model", "budget", "seed"])[metric].mean().reset_index()
    rows = []
    for n, g in per.groupby("budget"):
        piv = g.pivot(index="seed", columns="model", values=metric)
        for ref in refs:
            if ref not in piv: continue
            for m in [c for c in TUNED_MODELS if c in piv]:
                d = (piv[m] - piv[ref]).dropna()
                stat, pv = (wilcoxon(d) if len(d) >= 6 and (d != 0).any() else (np.nan, np.nan))
                rows.append(dict(budget=n, comparison=f"{m} vs {ref}", reference=ref, model=m, n_seeds=len(d),
                                 model_mcc=piv[m].mean(), ref_mcc=piv[ref].mean(), mean_diff=d.mean(),
                                 effect_r=rank_biserial(d), statistic=stat, p_value=pv))
    df = pd.DataFrame(rows)
    if len(df):
        df = holm_by_family(df, ["budget", "reference"])
    df.to_csv(os.path.join(out, "wilcoxon_budget_tuned.csv"), index=False)
    allm.groupby(["model", "budget"])[[metric, "mcc_natural", "precision_natural"]].mean().round(4) \
        .to_csv(os.path.join(out, "summary_budget_tuned.csv"))


# ======================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["nf", "legacy"], default="legacy",
                    help="nf = NetFlow-v2 datasets (main experiment); legacy = CIC/Edge/ToN (secondary)")
    ap.add_argument("--nf-dir", help="folder with the *_sample.csv.gz files made by prepare_nf_samples.py")
    ap.add_argument("--cic-dir"); ap.add_argument("--edge-csv"); ap.add_argument("--ton-csv")
    ap.add_argument("--max-rows", type=int, default=100000, help="rows kept per dataset (stratified by attack type)")
    ap.add_argument("--seeds", type=int, default=5, help="use 8-10 for the final run (needed for Wilcoxon)")
    ap.add_argument("--topk", type=int, default=None, help="default: 10 for nf, 3 for legacy")
    ap.add_argument("--tune", action="store_true", help="randomised hyper-parameter search (binary task)")
    ap.add_argument("--skip-multiclass", action="store_true")
    ap.add_argument("--skip-fewshot", action="store_true")
    ap.add_argument("--skip-esp32", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="ignore checkpoints and recompute everything")
    ap.add_argument("--out", default="results")
    ap.add_argument("--inspect", action="store_true", help="print columns / label counts and exit")
    ap.add_argument("--shift-k", type=int, default=SHIFT_K, help="features dropped by the shift filter")
    ap.add_argument("--skip-budget", action="store_true", help="skip the realistic label-budget experiment")
    ap.add_argument("--skip-tuned", action="store_true", help="skip the stronger baselines under the label budget")
    ap.add_argument("--artifact-features", nargs="*", default=None,
                    help="NF features dropped in the core_noart ablation (default: MIN_TTL MAX_TTL TCP_WIN_MAX_IN TCP_WIN_MAX_OUT)")
    args = ap.parse_args()
    global PROFILE
    PROFILE = args.profile
    if args.artifact_features is not None:
        ARTIFACT_FEATURES[:] = args.artifact_features
    if args.topk is None:
        args.topk = 10 if args.profile == "nf" else 3
    os.makedirs(args.out, exist_ok=True)
    json.dump({**vars(args), "python": sys.version, "pandas": pd.__version__, "numpy": np.__version__,
               "sklearn": sklearn.__version__, "xgboost": xgboost.__version__,
               "platform": platform.platform()}, open(os.path.join(args.out, "run_config.json"), "w"), indent=2)

    data = load_datasets(args)
    if args.profile == "nf":
        CORE[:] = [c for c in NF_FEATURES if all(c in d["avail"] for d in data.values())]
        FLOW_USED[:] = []
        print(f"Common NetFlow features used ({len(CORE)}):", CORE)
    else:
        FLOW_USED[:] = [c for c in FLOW if sum(c in d["avail"] for d in data.values()) >= 2]
        print("Flow features shared by >= 2 datasets:", FLOW_USED)
    cfg = dict(v=5, profile=args.profile, core=list(CORE), flow=FLOW_USED, art=list(ARTIFACT_FEATURES),
               max_rows=args.max_rows, tune=args.tune, topk=args.topk,
               paths=[args.cic_dir, args.edge_csv, args.ton_csv, args.nf_dir])
    ck = Ckpt(args.out, cfg, args.fresh)
    tuner = Tuner(args.tune, os.path.join(ck.dir, "tuner.joblib"))
    report, views, all_res = [], {"core": {}, "flow": {}, "flow_dn": {}, "core_dn": {}, "core_noart": {}}, []
    for tag, feats, xkey, dd in feature_sets():
        views[tag] = build_view(data, feats, xkey, dd, tag, report)
        if len(views[tag]) >= 2:
            all_res.append(collect(ck, tag, args.seeds, lambda s, t=tag: run_eval_seed(views[t], t, s, tuner, MODEL_NAMES)))
    if args.profile == "legacy":  # in nf mode the common features ARE the native features
        nviews = build_native_views(data, report)
        all_res.append(collect(ck, "native", args.seeds, lambda s: run_native_seed(nviews, s, tuner)))
    pd.DataFrame(report).to_csv(os.path.join(args.out, "dedup_report.csv"), index=False)

    if len(views["core"]) >= 2:
        run_shap_descriptive(views["core"], args.out, topk=args.topk)
        parts = [ck.get("stable", s, lambda s=s: run_eval_stable_seed(views["core"], s, args.topk, tuner))
                 for s in range(args.seeds)]
        if parts:
            all_res.append(pd.concat([p[0] for p in parts], ignore_index=True))
            pd.concat([p[1] for p in parts], ignore_index=True).to_csv(
                os.path.join(args.out, "stable_selection_per_fold.csv"), index=False)
        if args.profile == "nf" and len(CORE) > args.shift_k + 2:
            run_feature_shift(views["core"], args.out)
            sk = "shift" if args.shift_k == SHIFT_K else f"shift_k{args.shift_k}"
            parts = [ck.get(sk, s, lambda s=s: run_eval_shift_seed(views["core"], s, args.shift_k, tuner, MODEL_NAMES))
                     for s in range(args.seeds)]
            all_res.append(pd.concat([p[0] for p in parts], ignore_index=True))
            pd.concat([p[1] for p in parts], ignore_index=True).to_csv(
                os.path.join(args.out, "shift_filter_selection_per_fold.csv"), index=False)

    res = pd.concat([r for r in all_res if len(r)], ignore_index=True)
    res.to_csv(os.path.join(args.out, "eval_results.csv"), index=False)
    summarise(res, args.out)
    stats_tests(res, args.out)

    best = "flow" if len(views["flow"]) >= 2 else "core"
    fs = pd.DataFrame()
    if not args.skip_fewshot:
        plan = [("core", MODEL_NAMES + FEWSHOT_COMPACT), ("core_noart", FEWSHOT_NOART_MODELS), ("flow", MODEL_NAMES)]
        fs = pd.concat([collect(ck, f"fewshot2_{t}", args.seeds,
                                lambda s, t=t, ms=ms: run_fewshot2_seed(views[t], t, s, tuner, ms))
                        for t, ms in plan if len(views[t]) >= 2], ignore_index=True)
        if len(fs):
            fs.to_csv(os.path.join(args.out, "eval_fewshot.csv"), index=False)
            summarise(fs, args.out, prefix="summary_fewshot", extra_keys=("frac",))
            fewshot_tests(fs, args.out)

    if not args.skip_multiclass:
        mc = [collect(ck, f"mc_{t}", args.seeds, lambda s, t=t: run_eval_mc_seed(views[t], t, s))
              for t in ("core", "flow") if len(views[t]) >= 2]
        mc = [d for d in mc if len(d)]
        if mc:
            mcdf = pd.concat(mc, ignore_index=True); mcdf.to_csv(os.path.join(args.out, "eval_multiclass.csv"), index=False)
            summarise(mcdf, args.out, prefix="summary_multiclass",
                      metrics_=["accuracy", "balanced_acc", "macro_f1", "weighted_f1", "mcc", "coverage"])

    edge_metrics(views[best], best, args.out, tuner)

    rc = pd.DataFrame()
    if not args.skip_esp32:
        ctag = f"compact_{best}"
        rc = collect(ck, ctag, args.seeds, lambda s: run_eval_seed(views[best], ctag, s, tuner, list(COMPACT)))
        rc.to_csv(os.path.join(args.out, "eval_compact.csv"), index=False)
        summarise(rc, args.out, prefix="summary")
        meta = export_esp32(views[best], best, args.out)
        compact_report(res, rc, meta, best, args.out)

    if args.profile == "nf" and not args.skip_budget and not args.skip_fewshot and len(views["core"]) >= 3:
        shares = natural_shares(args.nf_dir)
        bd = collect(ck, "budget_core", args.seeds, lambda s: run_budget_seed(views["core"], s, tuner, shares))
        if len(bd):
            bd.to_csv(os.path.join(args.out, "eval_label_budget.csv"), index=False)
            bd.groupby(["setting", "model", "budget"])[["mcc", "mcc_natural", "precision_natural", "balanced_acc",
                                                         "recall", "fpr", "n_attack_labels", "single_class"]].mean()               .round(4).to_csv(os.path.join(args.out, "summary_label_budget.csv"))
            budget_tests(bd, args.out)
            if not args.skip_tuned:
                bt = collect(ck, "budget_tuned", args.seeds, lambda s: run_budget_tuned_seed(views["core"], s, shares))
                if len(bt):
                    bt.to_csv(os.path.join(args.out, "eval_label_budget_tuned.csv"), index=False)
                    budget_tuned_tests(bt, bd, args.out)

    if args.profile == "nf":
        prevalence_sensitivity(pd.concat([d for d in (res, fs, rc) if len(d)], ignore_index=True), args.out,
                               natural_shares(args.nf_dir))

    if tuner.log:
        pd.DataFrame(tuner.log).to_csv(os.path.join(args.out, "tuning_log.csv"), index=False)
    print(f"\nDone. Results in ./{args.out}/")


if __name__ == "__main__":
    main()
