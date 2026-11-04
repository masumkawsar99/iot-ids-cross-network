"""Draw the paper figures from a results folder.  Usage: python paper/make_figs.py <results folder> <output folder>"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
import os
R = sys.argv[1]; O = sys.argv[2]; os.makedirs(O, exist_ok=True)
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
AB = {"NF-BoT-IoT-v2": "BoT-IoT", "NF-CSE-CIC-IDS2018-v2": "CIC-IDS2018", "NF-ToN-IoT-v2": "ToN-IoT", "NF-UNSW-NB15-v2": "UNSW-NB15"}
ORD = ["ToN-IoT", "BoT-IoT", "UNSW-NB15", "CIC-IDS2018"]
M5 = ["RF", "XGB", "DT", "MLP", "LR"]
COL = {"RF": "#1f4e79", "XGB": "#d9822b", "DT": "#6a51a3", "MLP": "#2e8b57", "LR": "#7f7f7f",
       "DT_d4": "#b39ddb", "DT_d6": "#8e6cc8", "RF_t5_d6": "#5b8cc0", "MLP_8": "#86c59a"}
W1, W2 = 3.4, 5.15   # inches: single column / full width
def save(f, n): f.savefig(f"{O}/{n}.pdf"); f.savefig(f"{O}/{n}.png", dpi=300); plt.close(f)
def seed_ci(df, col="mcc"):
    s = df.groupby("seed")[col].mean(); n = len(s)
    return s.mean(), (stats.t.ppf(.975, n-1) * s.std(ddof=1) / np.sqrt(n) if n > 1 else 0)
r = pd.read_csv(f"{R}/eval_results.csv"); r["tr"] = r.train.map(AB); r["te"] = r.test.map(AB)

# Fig 2: ROC-AUC heatmaps (2 x 3 grid, last slot = colour bar)
g = r[(r.featset == "core") & r.setting.isin(["within", "cross"])]
SHORT = ["ToN", "BoT", "UNSW", "CIC"]
f, axs = plt.subplots(2, 3, figsize=(W2, 3.5), constrained_layout=True)
for a, m in zip(axs.flat, M5):
    p = g[g.model == m].pivot_table(index="tr", columns="te", values="roc_auc").reindex(index=ORD, columns=ORD)
    im = a.imshow(p.values, vmin=0, vmax=1, cmap="RdBu")
    for i in range(4):
        for j in range(4):
            v = p.values[i, j]; a.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="white" if abs(v - .5) > .3 else "black")
    a.set_xticks(range(4)); a.set_xticklabels(SHORT, fontsize=6); a.set_yticks(range(4)); a.set_yticklabels(SHORT, fontsize=6)
    a.set_title(m); a.tick_params(length=0); a.set_xlabel("Test")
    if m in ("RF", "MLP"): a.set_ylabel("Train")
    for sp in a.spines.values(): sp.set_visible(False)
axs.flat[5].axis("off")
cb = f.colorbar(im, ax=axs.flat[5], location="left", fraction=.12, shrink=.85); cb.set_label("ROC-AUC (< 0.5 = inverted)", fontsize=7)
save(f, "fig2_rocauc")

# Fig 3: generalisation gap (core)
f, a = plt.subplots(figsize=(W1, 2.2)); w = 0.27
SC = {"within": ("Within-dataset", "#1f4e79"), "cross": ("Cross-dataset", "#d9822b"), "lodo": ("Leave-one-dataset-out", "#c0392b")}
for k, (st, (lab, c)) in enumerate(SC.items()):
    v = [seed_ci(r[(r.featset == "core") & (r.setting == st) & (r.model == m)]) for m in M5]
    a.bar(np.arange(5) + (k-1)*w, [x for x, _ in v], w, yerr=[e for _, e in v], color=c, label=lab, capsize=1.5, error_kw={"lw": .7})
a.axhline(0, color="k", lw=.6); a.set_xticks(range(5)); a.set_xticklabels(M5); a.set_ylabel("MCC"); a.set_ylim(-.42, 1.02)
a.legend(frameon=False, loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=3, fontsize=6.5, handlelength=1.2, columnspacing=.8); save(f, "fig3_gap")

# Fig 4: SHAP heatmap (top 15)
imp = pd.read_csv(f"{R}/shap_importance.csv"); imp["d"] = imp.dataset.map(AB)
c = imp.groupby(["feature", "d"]).norm_importance.mean().unstack()[ORD]; c = c.loc[c.mean(1).sort_values(ascending=False).index[:15]]
ART = {"MIN_TTL", "MAX_TTL", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT"}
f, a = plt.subplots(figsize=(4.0, 3.6)); im = a.imshow(c.values, cmap="Blues", aspect="auto")
for i in range(c.shape[0]):
    for j in range(4):
        v = c.values[i, j]; a.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white" if v > .12 else "black")
a.set_xticks(range(4)); a.set_xticklabels(ORD, rotation=20, ha="right"); a.set_yticks(range(c.shape[0]))
a.set_yticklabels([x + (" (A)" if x in ART else "") for x in c.index], fontsize=6)
[t.set_color("#c0392b") for t, x in zip(a.get_yticklabels(), c.index) if x in ART]
a.tick_params(length=0); cb = f.colorbar(im, ax=a, fraction=.05, pad=.02); cb.set_label("Normalised mean |SHAP|", fontsize=7); cb.ax.tick_params(labelsize=6)
save(f, "fig4_shap")

# Fig 5: feature shift
sh = pd.read_csv(f"{R}/feature_shift.csv", index_col=0).sort_values("benign", ascending=False).head(18)
f, a = plt.subplots(figsize=(4.3, 3.2))
a.barh(range(len(sh)), sh.benign, color=["#c0392b" if x else "#9aa5b1" for x in sh.artifact], label="Benign traffic")
a.scatter(sh.attack, range(len(sh)), s=9, color="k", zorder=3, label="Attack traffic")
a.set_yticks(range(len(sh))); a.set_yticklabels(sh.index, fontsize=6); a.invert_yaxis(); a.set_xlim(0, .8)
med = pd.read_csv(f"{R}/feature_shift.csv", index_col=0).benign.median()
a.axvline(med, color="k", lw=.6, ls=":"); a.text(med - .01, -1.0, f"median of all 38 = {med:.2f}", fontsize=6, ha="right")
from matplotlib.patches import Patch
a.legend(handles=[Patch(color="#9aa5b1", label="Benign traffic"), Patch(color="#c0392b", label="Benign, testbed feature"),
                  plt.Line2D([], [], marker="o", ls="", color="k", ms=3, label="Attack traffic")],
         frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, fontsize=6, handlelength=1.2)
a.set_xlabel("Mean pairwise KS statistic between datasets")
save(f, "fig5_shift")

# Fig 6: feature sets, LODO
FS = [("core", "All 38"), ("core_noart", "No testbed\nfeatures"), ("core_dn", "Per-dataset\nnormalised"),
      ("stable_subset", "SHAP-\nstable"), ("shift_filter", "Shift filter"), ("shift_filter_dg", "Shift filter\n(no target)")]
f, a = plt.subplots(figsize=(W2, 2.3)); w = .16
for k, m in enumerate(M5):
    v = [seed_ci(r[(r.featset == fs) & (r.setting == "lodo") & (r.model == m)]) for fs, _ in FS]
    a.bar(np.arange(len(FS)) + (k-2)*w, [x for x, _ in v], w, yerr=[e for _, e in v], color=COL[m], label=m, capsize=1.2, error_kw={"lw": .6})
a.axhline(0, color="k", lw=.6); a.set_xticks(range(len(FS))); a.set_xticklabels([l for _, l in FS])
a.set_ylabel("LODO MCC"); a.legend(frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(.5, 1.0)); save(f, "fig6_featuresets")

# Fig 7: adaptation vs target-only
fs = pd.read_csv(f"{R}/eval_fewshot.csv"); cmp_ = pd.read_csv(f"{R}/eval_compact.csv")
f, ax = plt.subplots(1, 2, figsize=(W2, 2.5), sharey=True)
for a, models, ttl in ((ax[0], ["RF", "XGB", "DT", "MLP", "LR"], "(a) Full models"), (ax[1], ["DT_d4", "DT_d6", "RF_t5_d6", "MLP_8"], "(b) Microcontroller-sized models")):
    for m in models:
        base = r if m in M5 else cmp_
        z = seed_ci(base[(base.featset.isin(["core", "compact_core"])) & (base.setting == "cross") & (base.model == m)])
        for st, ls in (("fewshot", "-"), ("targetonly", "--")):
            d = fs[(fs.featset == "core") & (fs.model == m) & (fs.setting == st)]
            xs = sorted(d.frac.unique()); pts = [seed_ci(d[d.frac == x]) for x in xs]
            X = ([0] if st == "fewshot" else []) + [x * 100 for x in xs]; Y = ([z[0]] if st == "fewshot" else []) + [p[0] for p in pts]
            a.plot(X, Y, ls=ls, marker="o", ms=2.5, lw=1, color=COL[m], label=m if st == "fewshot" else None)
    a.set_xscale("symlog", linthresh=.5); a.set_xticks([0, .5, 1, 5]); a.set_xticklabels(["0", "0.5", "1", "5"])
    a.set_xlabel("Labelled target flows (% of target)"); a.set_title(ttl, loc="left"); a.axhline(0, color="k", lw=.5)
    if a is ax[0]: a.legend(frameon=False, loc="lower right", ncol=2, fontsize=6.5, columnspacing=.8)
ax[0].set_ylabel("MCC on remaining target"); ax[0].set_ylim(-.62, .98)
ax[1].plot([], [], "k-", lw=1, label="source + target"); ax[1].plot([], [], "k--", lw=1, label="target only")
h, l = ax[1].get_legend_handles_labels(); ax[1].legend(h, l, frameon=False, loc="lower right", ncol=2, fontsize=6, columnspacing=.8)
save(f, "fig7_adaptation")

# Fig 8: realistic label budget
bd = pd.read_csv(f"{R}/eval_label_budget.csv"); bd["te"] = bd.test.map(AB)
f, axs = plt.subplots(2, 2, figsize=(W2, 3.9), sharey=True, sharex=True); ax = axs.flat
LINES = [("DT_d4", "targetonly", "-", "DT_d4, target only"), ("DT_d4", "fewshot_w0.8", "--", "DT_d4, source + target"),
         ("RF", "targetonly", "-", "RF, target only"), ("RF", "fewshot_w0.8", "--", "RF, source + target"),
         ("RF", "targetonly_balanced", ":", "RF, target only, 50/50 labels")]
for a, te in zip(ax, ORD):
    g = bd[bd.te == te]
    for m, st, ls, lab in LINES:
        d = g[(g.model == m) & (g.setting == st)]; xs = sorted(d.budget.unique()); pts = [seed_ci(d[d.budget == x]) for x in xs]
        a.errorbar(xs, [p[0] for p in pts], yerr=[p[1] for p in pts], ls=ls, marker="o", ms=2.2, lw=1, capsize=1, elinewidth=.5,
                   color="#9aa5b1" if st == "targetonly_balanced" else COL[m], label=lab)
    a.set_xscale("log"); a.set_xticks([50, 100, 200, 500, 1000]); a.set_xticklabels(["50", "100", "200", "500", "1k"])
    a.set_title(f"{te} ({g.natural_share.iloc[0]:.1%} attacks)", fontsize=7.5); a.axhline(0, color="k", lw=.5); a.set_ylim(-.1, 1.02)
for a in axs[1]: a.set_xlabel("Labelled target flows")
for a in axs[:, 0]: a.set_ylabel("MCC on remaining target")
h, l = ax[0].get_legend_handles_labels(); f.legend(h, l, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(.5, -.01), fontsize=6.5)
f.subplots_adjust(bottom=.2, wspace=.08, hspace=.35); save(f, "fig8_budget")

# Fig 9: compact vs full: size vs MCC
cv = pd.read_csv(f"{R}/compact_vs_full.csv")
ec = pd.read_csv(f"{R}/eval_compact.csv")   # true LODO MCC (compact_vs_full's mcc_cross_lodo pools cross and LODO)
cv["mcc_lodo"] = cv.model.map(ec[ec.setting == "lodo"].groupby("model").mcc.mean())
f, a = plt.subplots(figsize=(W1, 2.2))
for _, row in cv.iterrows():
    a.scatter(row.data_bytes, row.mcc_within, s=14, color="#1f4e79", zorder=3)
    a.scatter(row.data_bytes, row.mcc_lodo, s=14, color="#c0392b", marker="s", zorder=3)
    OFF = {"LR": ((4, -9), "LR"), "DT_d4": ((-8, -11), "DT d4"), "DT_d6": ((-4, -11), "DT d6"), "MLP_8": ((-22, -11), "MLP 8"),
           "RF_t20_d6": ((-26, -11), "RF 20 trees")}
    if row.model in OFF:
        a.annotate(OFF[row.model][1], (row.data_bytes, row.mcc_within), fontsize=5.5, xytext=OFF[row.model][0], textcoords="offset points")
fr = cv.full_RF_mcc_within.iloc[0]; fl = r[(r.featset == "core") & (r.setting == "lodo") & (r.model == "RF")].mcc.mean()
a.axhline(fr, color="#1f4e79", ls=":", lw=.8); a.axhline(fl, color="#c0392b", ls=":", lw=.8)
a.text(130, fr + .03, "full RF (2.8 MB), within", fontsize=6, color="#1f4e79"); a.text(130, fl + .03, "full RF, LODO", fontsize=6, color="#c0392b")
a.scatter([], [], s=14, color="#1f4e79", label="Within-dataset"); a.scatter([], [], s=14, color="#c0392b", marker="s", label="LODO")
a.set_xscale("log"); a.set_xlabel("Model data size as C arrays (bytes)"); a.set_ylabel("MCC"); a.set_ylim(-.45, 1.1)
a.legend(frameon=False, loc="center right"); save(f, "fig9_compact")
print("ok")
