"""Generate all computed figures from results/*.json.
Colour figures: TIFF 600 dpi (LZW) for submission + PNG 300 dpi for the manuscript file."""
import json, pathlib, numpy as np, fourrun as fr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Circle

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)
cases = json.loads((RES / "cases.json").read_text())
partA = json.loads((RES / "partA.json").read_text())
partB = json.loads((RES / "partB.json").read_text())
faults = json.loads((RES / "faults.json").read_text())
pmap = json.loads((RES / "passmap.json").read_text())
rob = json.loads((RES / "robustness.json").read_text())

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#222222", "#666666", "#d9d9d9"
CASE_STYLE = {"BRT": (BLUE, "o", "-"), "IBR": (ORANGE, "s", "--"), "ICC": (AQUA, "^", ":")}
EST_STYLE = {"xy": (BLUE, "o", "-", r"$T_{xy}$"), "sum": (ORANGE, "s", "--", r"$T_{sum}$"),
             "graph": (AQUA, "^", ":", "graphical")}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.8, "axes.linewidth": 0.6,
    "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
    "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "lines.linewidth": 1.6,
    "lines.markersize": 5, "legend.frameon": False, "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "mathtext.fontset": "dejavusans",
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.tif", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def panel(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="bottom", ha="right")


def draw_construction(ax, V, title=None, show_xy=True, lim=None):
    V0, V1, V2, V3 = V
    al = np.deg2rad([0, 120, 240])
    cols = [BLUE, ORANGE, AQUA]
    ax.add_patch(Circle((0, 0), V0, fill=False, ls=(0, (2, 2)), lw=0.7, color=MUTED))
    for a, r, c, lab in zip(al, (V1, V2, V3), cols, ("1", "2", "3")):
        O = (V0 * np.cos(a), V0 * np.sin(a))
        ax.add_patch(Circle(O, r, fill=False, lw=1.1, color=c))
        ax.plot(*O, marker="o", ms=3.5, color=c)
        ax.annotate(f"$O_{lab}$", O, xytext=(4, 4), textcoords="offset points", fontsize=7.5, color=INK)
    g = fr.graphical_solve(V0, V1, V2, V3)
    P = g["points"]
    if g["spread"] > 1e-6:
        ax.add_patch(Polygon(P, closed=True, fill=True, fc="#00000014", ec=INK, lw=0.6))
    ax.plot(P[:, 0], P[:, 1], "o", ms=3.5, mfc="white", mec=INK, mew=0.8, zorder=5)
    ax.plot(*g["P"], "o", ms=5, color=INK, zorder=6)
    if show_xy:
        c = fr.correction(V0, V1, V2, V3, mt=1)
        th = np.deg2rad(c["theta"])
        ax.plot(c["Txy"] * np.cos(th), c["Txy"] * np.sin(th), "x", ms=6, mew=1.3, color="#c0392b", zorder=7)
    ax.plot([0, g["P"][0]], [0, g["P"][1]], color=INK, lw=0.8)
    ax.plot(0, 0, "+", color=INK, ms=6)
    ax.set_aspect("equal")
    if lim is not None:
        ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1])
    ax.grid(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8.5, color=INK)
    return g


# ---------------------------------------------------------------- Fig. 1
def fig1():
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.85))
    V0, T, phi = 5.0, 2.2, 40.0
    Vi = fr.ideal_amplitudes(V0, T, phi)
    Va = [V0] + [float(v) for v in Vi]
    Vb = [V0 + 0.15, Va[1] - 0.55, Va[2] + 0.60, Va[3] - 0.40]
    Vc = [5.0, 3.0, 5.2, 6.5]
    titles = ["noise-free data:\ncommon intersection", "noisy data:\nthree pairwise points", "pair (1, 2) does not intersect:\nradical-axis point used"]
    lims = [((-9, 9), (-9, 9))] * 3
    for ax, V, t, p in zip(axs, (Va, Vb, Vc), titles, "abc"):
        draw_construction(ax, V, f"({p}) " + t, show_xy=True, lim=((-12.5, 12.5), (-12.5, 12.5)))
        ax.title.set_fontsize(7.8)
    h = [plt.Line2D([], [], marker="o", ls="", mfc="white", mec=INK, label="retained pairwise point"),
         plt.Line2D([], [], marker="o", ls="", color=INK, label="graphical solution (mean)"),
         plt.Line2D([], [], marker="x", ls="", color="#c0392b", mew=1.3, label=r"analytical point ($T_{xy}$, $\theta$)")]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.subplots_adjust(wspace=0.05, bottom=0.12)
    save(fig, "Fig1_graphical_construction")


# ---------------------------------------------------------------- Fig. 2 (flowchart)
def box(ax, xy, w, h, text, kind="box", fc="white"):
    x, y = xy
    if kind == "diamond":
        pts = [(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)]
        ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=INK, lw=0.8))
    else:
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=fc, ec=INK, lw=0.8))
    ax.text(x, y, text, ha="center", va="center", fontsize=7.3, color=INK, linespacing=1.25)


def arrow(ax, p, q, text=None, tpos=None):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", lw=0.8, color=INK, mutation_scale=8))
    if text:
        ax.text(*(tpos or ((p[0] + q[0]) / 2 + 0.08, (p[1] + q[1]) / 2)), text, fontsize=7.2, color=INK, ha="left", va="center")


def fig2():
    fig, ax = plt.subplots(figsize=(6.4, 8.7))
    ax.set_xlim(0, 11.6); ax.set_ylim(0, 15.65); ax.axis("off")
    cx, xr = 4.3, 9.95
    E = [  # (y, height, width, kind, text, fill)
        (14.9, 0.95, 7.6, "box", "Run 0: $V_0$ (no trial weight)\nRuns 1-3: $V_1, V_2, V_3$ with $m_t$ at 0°, 120°, 240°\neach value = mean of $k$ readings", "white"),
        (13.35, 1.2, 7.6, "box", "Noise of the averaged values:\n" r"$\hat\sigma = s_p/\sqrt{k}$ from repeated readings ($\nu = 4(k-1)$)," "\n" r"or an assumed $\sigma = \sigma_{dev}/\sqrt{k}$", "#f3f3f3"),
        (11.9, 0.85, 7.6, "box", r"Compute $T^2_{sum}$, $x$, $y$, $T_{xy}$, $\rho$ and closure residual $g$  (Eqs. 3-4, 10)", "white"),
        (10.2, 1.55, 6.0, "diamond", "Consistency check\n" r"$|z| = |g|/u(g) \leq 2$ ($\sigma$ known)" "\n" r"or $\leq t_{0.975,\nu}$ ($\sigma$ estimated) ?", "white"),
        (8.3, 1.3, 6.0, "diamond", "Amplitude condition\n" r"all $V_i \geq 5\sigma$ ?", "white"),
        (6.45, 1.3, 6.6, "diamond", "Precision check\n" r"$U_{95}(m_c)$, $U_{95}(\theta)$ within tolerance ?", "white"),
        (4.85, 0.85, 7.6, "box", r"Correction from $T_{xy}$: $m_c$, $\theta$  (Eq. 6)" "\ngraphical construction as a visual cross-check", "white"),
        (3.6, 0.75, 7.6, "box", "Install correction weight, measure residual vibration", "white"),
        (2.2, 1.15, 5.0, "diamond", "Residual vibration\nacceptable ?", "white"),
        (0.7, 0.6, 3.0, "box", "End", "#f3f3f3"),
    ]
    for y, h, w, kind, txt, fc in E:
        box(ax, (cx, y), w, h, txt, kind=kind, fc=fc)
    for (y1, h1, *_), (y2, h2, *_) in zip(E[:-1], E[1:]):
        arrow(ax, (cx, y1 - h1 / 2 - 0.03), (cx, y2 + h2 / 2 + 0.03))
    for i in (3, 4, 5, 8):
        y, h = E[i][0], E[i][1]
        ax.text(cx + 0.12, y - h / 2 - 0.22, "yes", fontsize=7.2)
    side = [(3, "Flagged: check machine\nstate, mounting and\ntrial-weight positions;\nrepeat the runs", 1.45),
            (4, "Not assessable: change\ntrial-weight size or\nposition, or reduce\nthe noise", 1.35),
            (5, "Increase $m_t$\n(raise $T/\\sigma$) or\naverage more\nreadings", 1.3),
            (8, "Trim run, or check\nfor non-unbalance\nsources", 1.1)]
    for i, txt, hh in side:
        y, h, w = E[i][0], E[i][1], E[i][2]
        box(ax, (xr, y), 3.0, hh, txt, fc="#fff4ec")
        arrow(ax, (cx + w / 2, y), (xr - 1.5 - 0.03, y)); ax.text(cx + w / 2 + 0.1, y + 0.14, "no", fontsize=7.2)
    for i in (3, 4, 5):
        y = E[i][0]
        ax.plot([xr + 1.5, 11.55, 11.55], [y, y, E[0][0]], color=INK, lw=0.8)
    arrow(ax, (11.55, E[0][0]), (cx + 3.8 + 0.03, E[0][0]))
    save(fig, "Fig2_flowchart")


# ---------------------------------------------------------------- Fig. 4 (Part A)
def fig4():
    cells = partA["cells"]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9))
    ax = axs[0]
    for s, alpha_ in ((35, 1.0), (20, 0.45)):
        sub = [c for c in cells if c["s"] == s]
        r = [c["r"] for c in sub]
        for k, (col, mk, ls, lab) in EST_STYLE.items():
            ax.plot(r, [c[k]["rmse"] for c in sub], color=col, marker=mk, ls=ls, alpha=alpha_,
                    label=lab if s == 35 else None, ms=4)
    ax.set_xlabel(r"$V_0/T$"); ax.set_ylabel(r"RMSE of $m_c$ (%)")
    ax.set_yscale("log"); ax.set_ylim(1.5, 200)
    ax.text(3.05, 2.4, r"$T/\sigma = 35$", fontsize=7.5, color=INK)
    ax.text(3.05, 8.0, r"$T/\sigma = 20$" "\n(faded)", fontsize=7.5, color=MUTED)
    ax.legend(loc="upper left", ncol=3, handlelength=2.2, columnspacing=1.0)
    panel(ax, "(a)")
    ax = axs[1]
    R, S, P = np.array(pmap["r"]), np.array(pmap["s"]), 100 * np.array(pmap["pass_rate"])
    cs = ax.contourf(R, S, P, levels=[0, 50, 80, 90, 95, 99, 100.01], cmap="Blues", alpha=0.9)
    cl = ax.contour(R, S, P, levels=[90, 95, 99], colors=[MUTED, INK, MUTED], linewidths=[0.6, 1.2, 0.6])
    ax.clabel(cl, fmt="%d%%", fontsize=7, inline_spacing=2, manual=[(4.4, 15.5), (4.4, 18.6), (4.4, 25.0)])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([0.75, 1, 1.5, 2, 3, 4, 5]); ax.set_xticklabels(["0.75", "1", "1.5", "2", "3", "4", "5"])
    ax.set_yticks([10, 20, 30, 50, 100]); ax.set_yticklabels(["10", "20", "30", "50", "100"])
    ax.minorticks_off(); ax.grid(False)
    for key, (col, mk, _) in CASE_STYLE.items():
        c = cases[key]
        for sd, fill in ((0.10, col), (0.20, "white")):
            s = c["Txy"] / (sd / np.sqrt(3))
            ax.plot(c["V0_over_T"], s, marker=mk, ms=6, mfc=fill, mec=INK, mew=0.9, ls="")
        off = {"BRT": (6, 2), "IBR": (-26, 4), "ICC": (6, 2)}[key]
        ax.annotate(key, (c["V0_over_T"], c["Txy"] / (0.10 / np.sqrt(3))), xytext=off, textcoords="offset points", fontsize=7.5,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    ax.set_xlabel(r"$V_0/T$"); ax.set_ylabel(r"$T/\sigma$")
    hh = [plt.Line2D([], [], marker="o", ls="", mfc=MUTED, mec=INK, label=r"$\sigma_{dev}$ = 0.10 mm/s"),
          plt.Line2D([], [], marker="o", ls="", mfc="white", mec=INK, label=r"$\sigma_{dev}$ = 0.20 mm/s")]
    ax.legend(handles=hh, loc="lower right", fontsize=7, frameon=True, facecolor="white", edgecolor="none", framealpha=0.85)
    cb = fig.colorbar(cs, ax=ax, pad=0.02); cb.set_label("P(|Δm| ≤ 10 % and |Δθ| ≤ 10°)  (%)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    panel(ax, "(b)")
    fig.tight_layout(w_pad=1.5)
    save(fig, "Fig4_partA_accuracy")


# ---------------------------------------------------------------- Fig. 5 (screening)
def fig5():
    cells = partA["cells"]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.85), gridspec_kw=dict(width_ratios=[1, 1.25]))
    ax = axs[0]
    sub = [c for c in cells if c["s"] == 35]
    r = [c["r"] for c in sub]
    styles = [("fixed_0.08", BLUE, "o", "-", r"$|\rho-1|>0.08$"), ("fixed_0.15", ORANGE, "s", "--", r"$|\rho-1|>0.15$"),
              ("fixed_0.25", AQUA, "^", ":", r"$|\rho-1|>0.25$"), ("z2", INK, "D", "-", r"$|z|>2$")]
    for k, col, mk, ls, lab in styles:
        ax.plot(r, [100 * c["screening"][k]["flag_rate"] for c in sub], color=col, marker=mk, ls=ls, label=lab, ms=4)
    zall = [100 * c["screening"]["z2"]["flag_rate"] for c in cells]
    ax.axhspan(min(zall), max(zall), color="#00000010", lw=0)
    ax.set_xlabel(r"$V_0/T$  (consistent data, $T/\sigma = 35$)"); ax.set_ylabel("false-alarm rate (%)")
    ax.set_ylim(-2, 60); ax.legend(loc="upper left")
    ax.annotate(f"|z| > 2 over all 35 grid cells:\n{min(zall):.1f}-{max(zall):.1f} % (shaded band)", xy=(1.2, max(zall)), xytext=(0.78, 17),
                fontsize=7, color=MUTED, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    panel(ax, "(a)")
    ax = axs[1]
    fl = [("angle_one_10", "one\nposition\n+10°"), ("angle_one_15", "one\nposition\n+15°"), ("angle_all_20", "all\npositions\n+20°"),
          ("gain_one_10pct", "trial\neffect\n+10 %"), ("v0_drift_5pct", "$V_0$\ndrift\n+5 %")]
    x = np.arange(len(fl)); w = 0.26
    for i, (key, (col, _, _)) in enumerate(CASE_STYLE.items()):
        vals = [100 * faults["results"][key]["0.10"][f]["detect"] for f, _ in fl]
        ax.bar(x + (i - 1) * w, vals, w - 0.03, color=col, label=key, zorder=3)
    ax.axhline(5, color=INK, lw=0.7, ls=(0, (3, 2)), label="nominal 5 %")
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in fl], fontsize=7)
    ax.set_ylabel(r"detection rate, $|z|>2$ (%)"); ax.set_ylim(0, 122)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.legend(loc="upper left", ncol=4, bbox_to_anchor=(0.0, 1.03), columnspacing=1.0, handlelength=1.6); ax.grid(axis="x", visible=False)
    panel(ax, "(b)")
    fig.tight_layout(w_pad=1.5)
    save(fig, "Fig5_screening")


# ---------------------------------------------------------------- Fig. 6 (Part B)
def fig6():
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.45))
    for key, (col, mk, ls) in CASE_STYLE.items():
        rows = partB["cases"][key]["rows"]
        sd = [r["sig_dev"] for r in rows]
        axs[0].plot(sd, [r["p95_abs_m"] for r in rows], marker=mk, ls="", color=col, label=f"{key} (MC)")
        axs[0].plot(sd, [r["pred_U95_m"] for r in rows], ls=ls, color=col, lw=1.1)
        axs[1].plot(sd, [r["p95_abs_t"] for r in rows], marker=mk, ls="", color=col)
        axs[1].plot(sd, [r["pred_U95_t"] for r in rows], ls=ls, color=col, lw=1.1)
        axs[2].plot(sd, [100 * r["pass_rate"] for r in rows], marker=mk, ls=ls, color=col, label=key)
    axs[0].set_ylabel(r"95th percentile of $|\Delta m_c|$ (%)"); axs[1].set_ylabel(r"95th percentile of $|\Delta\theta|$ (°)")
    axs[2].set_ylabel("P(|Δm| ≤ 10 % and |Δθ| ≤ 10°) (%)"); axs[2].axhline(95, color=INK, lw=0.7, ls=(0, (3, 2)))
    for ax, p in zip(axs, "abc"):
        ax.set_xscale("log"); ax.set_xlabel(r"$\sigma_{dev}$ (mm/s), $k = 3$")
        ax.set_xticks([0.1, 0.2, 0.5, 1.0]); ax.set_xticklabels(["0.1", "0.2", "0.5", "1.0"]); ax.minorticks_off(); panel(ax, f"({p})")
    axs[0].legend(loc="upper left", fontsize=7)
    axs[1].plot([], [], color=MUTED, lw=1.1, label="first-order\nprediction"); axs[1].plot([], [], "o", color=MUTED, label="Monte Carlo")
    axs[1].legend(loc="upper left", fontsize=7)
    fig.tight_layout(w_pad=1.0)
    save(fig, "Fig6_partB_sensitivity")


# ---------------------------------------------------------------- Fig. 9 (case constructions)
def fig9():
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 3.0))
    for ax, key, p in zip(axs, ("BRT", "IBR", "ICC"), "abc"):
        c = cases[key]; V = c["V"]
        L = 1.08 * max(V[0] + max(V[1:]), 2 * V[0])
        draw_construction(ax, V, None, show_xy=True, lim=((-L, L), (-L, L)))
        ax.set_title(f"({p}) {key}\n$\\rho$ = {c['rho']:.3f}, spread = {c['spread']:.2f} mm/s", fontsize=7.8)
        # inset zoom on the solution region
        g = fr.graphical_solve(*V)
        ins = ax.inset_axes([0.62, 0.62, 0.38, 0.38])
        draw_construction(ins, V, None, show_xy=True)
        cx, cy = g["P"]; d = max(3.0 * g["spread"], 0.15 * c["Txy"])
        ins.set_xlim(cx - d, cx + d); ins.set_ylim(cy - d, cy + d)
        for s in ins.spines.values():
            s.set_visible(True); s.set_color(MUTED); s.set_linewidth(0.6)
        for t in ins.texts:
            t.set_visible(False)
    h = [plt.Line2D([], [], marker="o", ls="", mfc="white", mec=INK, label="retained pairwise point"),
         plt.Line2D([], [], marker="o", ls="", color=INK, label="graphical solution"),
         plt.Line2D([], [], marker="x", ls="", color="#c0392b", mew=1.3, label="analytical solution")]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(wspace=0.06, bottom=0.02, top=0.86)
    save(fig, "Fig9_case_constructions")


# ---------------------------------------------------------------- robustness figure
def fig_robust():
    YEL = "#eda100"
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.55))
    ax = axs[0]
    for r, col, mk, ls in (("1.0", ORANGE, "s", "--"), ("1.25", AQUA, "^", ":"), ("2.0", BLUE, "o", "-"), ("4.0", YEL, "D", "-.")):
        d = rob["phase"][r]
        ax.plot(d["phi"], [100 * v for v in d["fa"]], color=col, ls=ls, lw=1.3, label=f"$V_0/T$ = {float(r):g}")
    ax.axhline(5, color=INK, lw=0.7, ls=(0, (3, 2)))
    ax.set_xlabel(r"$\varphi$ (°), period 120°"); ax.set_ylabel(r"false-alarm rate, $|z|>2$ (%)")
    ax.set_xticks([0, 30, 60, 90, 120]); ax.set_ylim(-0.5, 8)
    ax.annotate("one trial run\nnear cancellation", xy=(60, 0.3), xytext=(68, 1.6), fontsize=6.8, color=MUTED,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.legend(loc="upper right", fontsize=6.5, ncol=2, columnspacing=0.8, handlelength=1.8)
    panel(ax, "(a)")
    ax = axs[1]
    q = [r["ratio"] for r in rob["misstated"]["BRT"]]
    ax.plot(q, [100 * r["none"] for r in rob["misstated"]["BRT"]], color=INK, marker="D", ms=3.5, lw=1.3, label="false alarm (all cases)")
    for key, (col, mk, ls) in CASE_STYLE.items():
        ax.plot(q, [100 * r["v0_drift_5pct"] for r in rob["misstated"][key]], color=col, marker=mk, ls=ls, ms=3.5, lw=1.1,
                label=f"{key}: $V_0$ drift detected")
    ax.set_xscale("log"); ax.set_xticks([0.5, 0.75, 1, 1.5, 2]); ax.set_xticklabels(["0.5", "0.75", "1", "1.5", "2"]); ax.minorticks_off()
    ax.set_xlabel(r"stated $\sigma$ / true $\sigma$"); ax.set_ylabel("rate (%)"); ax.set_ylim(-3, 105)
    ax.legend(loc="center right", fontsize=6.3, bbox_to_anchor=(1.0, 0.55))
    panel(ax, "(b)")
    ax = axs[2]
    zs = rob["z_vs_sigma"]; sd = zs["sigma_dev"]
    for key, (col, mk, ls) in CASE_STYLE.items():
        ax.plot(sd, zs[key], color=col, ls=ls, lw=1.4, label=key)
    for y in (2, -2):
        ax.axhline(y, color=INK, lw=0.7)
    for y in (rob["t_nu"], -rob["t_nu"]):
        ax.axhline(y, color=INK, lw=0.7, ls=(0, (3, 2)))
    ax.axvline(0.10, color=MUTED, lw=0.6, ls=":")
    ax.set_xscale("log"); ax.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5]); ax.set_xticklabels(["0.02", "0.05", "0.1", "0.2", "0.5"]); ax.minorticks_off()
    ax.set_ylim(-9, 3); ax.set_xlabel(r"assumed $\sigma_{dev}$ (mm/s), $k = 3$"); ax.set_ylabel(r"$z$ of the published readings")
    ax.plot([], [], color=INK, lw=0.7, label="±2 (σ known)")
    ax.plot([], [], color=INK, lw=0.7, ls=(0, (3, 2)), label=f"±{rob['t_nu']:.3f} (σ estimated, ν = 8)")
    ax.legend(loc="lower right", fontsize=6.2, frameon=True, facecolor="white", edgecolor="none", framealpha=0.92)
    panel(ax, "(c)")
    fig.tight_layout(w_pad=0.9)
    save(fig, "Fig6_robustness")


if __name__ == "__main__":
    fig1(); fig2(); fig4(); fig5(); fig6(); fig9(); fig_robust()
    print("figures written:", sorted(p.name for p in FIG.glob("*.png")))
