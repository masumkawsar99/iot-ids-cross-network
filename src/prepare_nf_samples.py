#!/usr/bin/env python3
"""
Download the big NetFlow-v2 datasets (UQ) and keep a SMALL stratified sample of each, so that Google Drive does not fill up.

For every dataset: pass 1 counts rows per attack type; pass 2 keeps
   * benign rows  : up to  N_TOTAL * BENIGN_SHARE  (default 50%)
   * attack rows  : the rest, split as equally as possible between attack types (rare types are kept in full)
so all datasets end up with a comparable benign/attack mix (state this in the paper) and rare attacks survive.

Usage (Colab):  python prepare_nf_samples.py --out /content/drive/MyDrive/iot_ids/nf_samples \
                    --raw-dir /content/nf_raw --datasets NF-ToN-IoT-v2 NF-BoT-IoT-v2 NF-UNSW-NB15-v2
If a download fails, download the file in your browser, upload it to --raw-dir (any file name starting with the
dataset name, e.g. NF-ToN-IoT-v2.csv or NF-ToN-IoT-v2.zip) and run again.
"""
import argparse, glob, json, os, subprocess, sys, zipfile, zlib
import numpy as np
import pandas as pd

SOURCES = {
    "NF-ToN-IoT-v2": "https://rdm.uq.edu.au/files/a4ad7080-ef9c-11ed-a964-b70596e96ad5",
    "NF-BoT-IoT-v2": "https://rdm.uq.edu.au/files/b97fa3c0-ef9c-11ed-8007-832d3bc76ba8",
    "NF-UNSW-NB15-v2": "https://rdm.uq.edu.au/files/8c6e2a00-ef9c-11ed-827d-e762de186848",
    "NF-CSE-CIC-IDS2018-v2": "https://rdm.uq.edu.au/files/ce5161d0-ef9c-11ed-827d-e762de186848",
}
CHUNK = 500_000


def sniff(path):
    with open(path, "rb") as f:
        m = f.read(4)
    return "zip" if m[:4] == b"PK\x03\x04" else "gzip" if m[:2] == b"\x1f\x8b" else None


def find_local(raw_dir, name):
    """A file you uploaded yourself (zip / csv / csv.gz whose name starts with the dataset name).
    An already-extracted csv is preferred; an unfinished '.download' is ignored (wget resumes it instead)."""
    c = [p for p in glob.glob(os.path.join(raw_dir, name + "*")) if os.path.isfile(p) and not p.endswith(".download")]
    ext = [p for p in c if p.endswith("_extracted.csv")]
    if ext:
        return ext[0]
    return max(c, key=os.path.getsize) if c else None


def download(name, url, raw_dir):
    os.makedirs(raw_dir, exist_ok=True)
    dst = os.path.join(raw_dir, name + ".download")
    print(f"[{name}] downloading {url}\n  -> {dst} (this can take a long time; progress below)")
    r = subprocess.run(["wget", "-c", "--progress=dot:giga", "-O", dst, url])
    ok = r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 1_000_000
    if ok and sniff(dst) is None:  # a web page (login / landing page) instead of the data file
        with open(dst, "rb") as f:
            ok = not f.read(512).lstrip().lower().startswith((b"<!doctype", b"<html"))
    if not ok:
        if os.path.exists(dst) and os.path.getsize(dst) < 1_000_000:
            os.remove(dst)
        raise RuntimeError(f"[{name}] download failed (or the server returned a web page instead of the file).\n"
                           f"Open https://www.cyber.uq.edu.au/project/machine-learning-based-nids-datasets in your browser,\n"
                           f"download {name}, upload the file to {raw_dir} (file name starting with '{name}') "
                           f"and run this step again.")
    return dst


def open_csv(path, raw_dir, name):
    """Return (csv_path, compression) usable by pandas; unzip a zip archive to a plain csv."""
    kind = sniff(path)
    if kind == "zip":
        with zipfile.ZipFile(path) as z:
            # the archive may also hold small csv files (e.g. feature descriptions): take the biggest one
            members = sorted([i for i in z.infolist() if i.filename.lower().endswith(".csv")],
                             key=lambda i: i.file_size, reverse=True)
            if not members:
                raise RuntimeError(f"No .csv inside {path}: {z.namelist()[:5]}")
            members = [m.filename for m in members]
            out = os.path.join(raw_dir, name + "_extracted.csv")
            if not os.path.exists(out):
                print(f"[{name}] extracting {members[0]}")
                with z.open(members[0]) as src, open(out + ".part", "wb") as dst:
                    while True:
                        b = src.read(1 << 24)
                        if not b:
                            break
                        dst.write(b)
                os.replace(out + ".part", out)  # only a complete extraction gets the final name
        return out, None
    return path, ("gzip" if kind == "gzip" else None)


def allocate(counts, n_total, benign_share, min_per_class):
    """Quota per attack type: benign up to n_total*benign_share, attacks share the rest equally (water-filling)."""
    ben = [c for c in counts.index if str(c).strip().lower().startswith("benign")]
    atk = [c for c in counts.index if c not in ben]
    alloc = {c: int(counts[c]) for c in ben}
    n_ben = sum(alloc.values())
    cap_ben = int(n_total * benign_share)
    if n_ben > cap_ben:  # scale benign down proportionally (normally a single 'Benign' class)
        alloc = {c: max(1, int(counts[c] * cap_ben / n_ben)) for c in ben}
    pool = n_total - sum(alloc.values())
    rem = {c: int(counts[c]) for c in atk}
    got = {}
    while rem and pool > 0:  # water-filling: small classes are taken in full
        share = max(pool // len(rem), 1)
        small = {c: n for c, n in rem.items() if n <= share}
        if not small:
            for c in rem:
                got[c] = share
            break
        for c, n in small.items():
            got[c] = n; pool -= n; del rem[c]
    for c in atk:
        got[c] = int(min(counts[c], max(got.get(c, 0), min_per_class)))
    alloc.update(got)
    return alloc


def prepare(name, url, raw_dir, out_dir, n_total, benign_share, min_per_class, keep_raw, seed=0):
    out = os.path.join(out_dir, f"{name}_sample.csv.gz")
    if os.path.exists(out):
        print(f"[{name}] sample already exists: {out}")
        return out
    local = find_local(raw_dir, name) or download(name, url, raw_dir)
    path, comp = open_csv(local, raw_dir, name)
    print(f"[{name}] pass 1: counting attack types in {os.path.basename(path)}")
    counts = pd.Series(dtype="int64")
    for ch in pd.read_csv(path, usecols=lambda c: c.strip() == "Attack", chunksize=CHUNK, compression=comp,
                          low_memory=False):
        ch.columns = [c.strip() for c in ch.columns]
        counts = counts.add(ch["Attack"].astype(str).str.strip().value_counts(), fill_value=0)
    counts = counts.astype("int64")
    print(counts.to_string())
    ben = int(sum(v for c, v in counts.items() if str(c).lower().startswith("benign")))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{name}_counts.json"), "w") as f:  # full-file counts: real attack share
        json.dump({"benign": ben, "attack": int(counts.sum()) - ben, "counts": {str(k): int(v) for k, v in counts.items()}}, f, indent=1)
    alloc = allocate(counts, n_total, benign_share, min_per_class)
    prob = pd.Series({c: min(1.0, alloc[c] / counts[c]) for c in counts.index})
    print(f"[{name}] pass 2: sampling (target ~{sum(alloc.values()):,} rows)")
    rng, parts = np.random.default_rng([seed, zlib.crc32(name.encode())]), []  # reproducible, independent per dataset
    for ch in pd.read_csv(path, chunksize=CHUNK, compression=comp, low_memory=False):
        ch.columns = [c.strip() for c in ch.columns]
        p = ch["Attack"].astype(str).str.strip().map(prob).fillna(0).to_numpy()
        parts.append(ch[rng.random(len(ch)) < p])
    df = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(out + ".part", index=False, compression="gzip")
    os.replace(out + ".part", out)  # an interrupted write never looks like a finished sample
    print(f"[{name}] saved {len(df):,} rows -> {out}\n{df['Attack'].value_counts().to_string()}")
    if not keep_raw:
        for f in (local, path):
            if os.path.exists(f) and os.path.dirname(os.path.abspath(f)) == os.path.abspath(raw_dir) and f.endswith(".download"):
                os.remove(f)
        if path.endswith("_extracted.csv") and os.path.exists(path):
            os.remove(path)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="folder for the small *_sample.csv.gz files (put it on Drive)")
    ap.add_argument("--raw-dir", default="/content/nf_raw", help="temporary folder for the big downloads (NOT Drive)")
    ap.add_argument("--datasets", nargs="+", default=["NF-ToN-IoT-v2", "NF-BoT-IoT-v2", "NF-UNSW-NB15-v2"])
    ap.add_argument("--n-total", type=int, default=200_000, help="rows kept per dataset")
    ap.add_argument("--benign-share", type=float, default=0.5)
    ap.add_argument("--min-per-class", type=int, default=2000)
    ap.add_argument("--keep-raw", action="store_true")
    a = ap.parse_args()
    for n in a.datasets:
        if n not in SOURCES:
            sys.exit(f"Unknown dataset {n}. Choose from {list(SOURCES)}")
        prepare(n, SOURCES[n], a.raw_dir, a.out, a.n_total, a.benign_share, a.min_per_class, a.keep_raw)


if __name__ == "__main__":
    main()
