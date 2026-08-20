"""
main.py - Solver-agnostic benchmark orchestrator and multi-viewport visualizer.
"""
import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from math import pi

from simulator import random_state, qrot, SIM_FPS
from solver_casadi import solve_casadi
from solver_kinematic import solve_kinematic
from solver_penguin import solve_penguin
from solver_roundy import solve_roundy
from solver_btt import solve_btt
from solver_2DAVF import solve_dir2d
from solver_S2xS1 import solve_s2s1
from solver_casadi_fast import solve_casadi_fast

# ─── Configuration ────────────────────────────────────────────────────────────
N_SAMPLES = 125
START_SAMPLE = 0
TOL_ANGLE = 0.05
TOL_SPEED = 0.07
OUTPUT_DIR = "renders"
SAVE_ALL_GIFS = False
CACHE_FILE = "benchmark_cache.pkl"

ACTIVE_SOLVERS = {
    "CasADi":     {"fn": solve_casadi,    "color": "#00D2FF", "force": False},
    "CasADi_Fast": {"fn": solve_casadi_fast, "color": "#FFD200", "force": False},
    #"Roundy":     {"fn": solve_roundy,    "color": "#66fc25", "force": True},
    "RM_Kinematic": {"fn": solve_kinematic, "color": "#FF7B00", "force": False},
    #"PenguinBot": {"fn": solve_penguin,   "color": "#B05CFF"},
    #"BTT": {"fn": solve_btt, "color": "#44fc9d"},
    "2D_AVF": {"fn": solve_dir2d, "color": "#44fced", "force": False},
    #"S2xS1": {"fn": solve_s2s1, "color": "#4497fc"},
}

# 3D Chassis wireframe edges
CAR_LOCAL_EDGES = [
    (np.array([1.0, 0.5, -0.25]), np.array([1.0, -0.5, -0.25])),
    (np.array([1.0, -0.5, -0.25]), np.array([-1.0, -0.5, -0.25])),
    (np.array([-1.0, -0.5, -0.25]), np.array([-1.0, 0.5, -0.25])),
    (np.array([-1.0, 0.5, -0.25]), np.array([1.0, 0.5, -0.25])),
    (np.array([0.3, 0.35, 0.35]), np.array([0.3, -0.35, 0.35])),
    (np.array([0.3, -0.35, 0.35]), np.array([-0.6, -0.35, 0.35])),
    (np.array([-0.6, -0.35, 0.35]), np.array([-0.6, 0.35, 0.35])),
    (np.array([-0.6, 0.35, 0.35]), np.array([0.3, 0.35, 0.35])),
    (np.array([1.0, 0.5, -0.25]), np.array([0.3, 0.35, 0.35])),
    (np.array([1.0, -0.5, -0.25]), np.array([0.3, -0.35, 0.35])),
    (np.array([-1.0, -0.5, -0.25]), np.array([-0.6, -0.35, 0.35])),
    (np.array([-1.0, 0.5, -0.25]), np.array([-0.6, 0.35, 0.35])),
    (np.array([1.0, 0.5, -0.25]), np.array([1.3, 0.0, -0.1])),
    (np.array([1.0, -0.5, -0.25]), np.array([1.3, 0.0, -0.1])),
]


# ─── Agnostic Visualizer (Square 3D Grid + Static Telemetry Graphs) ───────────
def render_comparison_animation(results_dict, filename):
    """
    Renders an agnostic square-ish 3D grid of all provided solutions on the left,
    and full static Attitude Error & Angular Speed graphs with settle markers on the right.
    """
    solvers = list(results_dict.keys())
    n_solvers = len(solvers)

    # Dynamic square-ish subgrid for 3D viewports
    if n_solvers <= 2:
        nrows_3d, ncols_3d = 1, 2
    elif n_solvers <= 4:
        nrows_3d, ncols_3d = 2, 2
    elif n_solvers <= 6:
        nrows_3d, ncols_3d = 2, 3
    else:
        ncols_3d = int(np.ceil(np.sqrt(n_solvers)))
        nrows_3d = int(np.ceil(n_solvers / ncols_3d))

    fig = plt.figure(figsize=(15, 8.5), facecolor="#141414")

    # Outer layout: Left = 3D Viewports Grid, Right = 2D Telemetry Graphs
    gs_main = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.18)
    gs_3d = gs_main[0].subgridspec(nrows_3d, ncols_3d, wspace=0.05, hspace=0.05)
    gs_graphs = gs_main[1].subgridspec(2, 1, hspace=0.30)

    # 3D Axes
    axes_3d = []
    for i in range(n_solvers):
        r, c = i // ncols_3d, i % ncols_3d
        ax = fig.add_subplot(gs_3d[r, c], projection="3d", facecolor="#141414")
        axes_3d.append(ax)

    # 2D Graph Axes
    ax_err = fig.add_subplot(gs_graphs[0], facecolor="#1c1c1c")
    ax_spd = fig.add_subplot(gs_graphs[1], facecolor="#1c1c1c")

    # Style 2D graphs
    for ax, title, ylabel in [(ax_err, "Attitude Error Over Time", "Error [deg]"),
                              (ax_spd, "Total Angular Speed Over Time", "Speed [rad/s]")]:
        ax.set_title(title, color="#FFFFFF", fontsize=11, weight="bold", pad=8)
        ax.set_ylabel(ylabel, color="#CCCCCC", fontsize=9.5)
        ax.set_xlabel("Time [s]", color="#CCCCCC", fontsize=9.5)
        ax.tick_params(colors="#888888", labelsize=8.5)
        ax.grid(True, color="#333333", linestyle="--", alpha=0.7)
        for spine in ax.spines.values():
            spine.set_color("#444444")

    # Static tolerance limits on 2D plots
    ax_err.axhline(TOL_ANGLE * (180.0 / pi), color="#FF3366", linestyle=":", lw=1.2,
                   label=f"Tol ({TOL_ANGLE * (180 / pi):.1f}°)")
    ax_spd.axhline(TOL_SPEED, color="#FF3366", linestyle=":", lw=1.2, label=f"Tol ({TOL_SPEED:.2f} r/s)")

    # Plot full static curves & mark exact settle points
    max_t = 0.0
    for name in solvers:
        hist = results_dict[name]["hist"]
        col = results_dict[name]["color"]
        t_settle = results_dict[name]["t_settle"]

        t_arr = np.array([h["t"] for h in hist])
        err_arr = np.array([h["angle_rad"] * (180.0 / pi) for h in hist])
        spd_arr = np.array([h["speed"] for h in hist])

        # Plot curves with settle time in legend
        ax_err.plot(t_arr, err_arr, color=col, lw=2.0, label=f"{name} [{t_settle:.3f}s]")
        ax_spd.plot(t_arr, spd_arr, color=col, lw=2.0, label=f"{name} [{t_settle:.3f}s]")

        # Mark exact settle time with vertical dashed lines
        ax_err.axvline(t_settle, color=col, linestyle="--", alpha=0.65, lw=1.2)
        ax_spd.axvline(t_settle, color=col, linestyle="--", alpha=0.65, lw=1.2)

        # Plot marker at settle point on curve
        idx_s = min(len(t_arr) - 1, int(round(t_settle * SIM_FPS)))
        ax_err.scatter([t_settle], [err_arr[idx_s]], color=col, s=45, zorder=5, edgecolors="#FFFFFF", lw=0.8)
        ax_spd.scatter([t_settle], [spd_arr[idx_s]], color=col, s=45, zorder=5, edgecolors="#FFFFFF", lw=0.8)

        max_t = max(max_t, t_arr[-1])

    ax_err.set_xlim([0, max(max_t, 0.2)])
    ax_spd.set_xlim([0, max(max_t, 0.2)])
    ax_err.legend(loc="upper right", facecolor="#222222", edgecolor="#444444", labelcolor="#FFFFFF", fontsize=8.5)
    ax_spd.legend(loc="upper right", facecolor="#222222", edgecolor="#444444", labelcolor="#FFFFFF", fontsize=8.5)

    # 3D Viewport Drawing Function
    def draw_car_frame(ax, rot, title, color, status, t_settle):
        ax.cla()
        ax.set_xlim([-1.6, 1.6])
        ax.set_ylim([-1.6, 1.6])
        ax.set_zlim([-1.6, 1.6])
        ax.set_box_aspect([1, 1, 1])
        ax.axis("off")

        # Ghost target frame
        ax.plot([0, 1.1], [0, 0], [0, 0], color="#444444", linestyle=":", lw=1.0)
        ax.plot([0, 0], [0, 1.1], [0, 0], color="#444444", linestyle=":", lw=1.0)
        ax.plot([0, 0], [0, 0], [0, 1.1], color="#444444", linestyle=":", lw=1.0)

        # Wireframe Chassis
        for p1, p2 in CAR_LOCAL_EDGES:
            p1_r = qrot(rot, p1)
            p2_r = qrot(rot, p2)
            ax.plot([p1_r[0], p2_r[0]], [p1_r[1], p2_r[1]], [p1_r[2], p2_r[2]], color=color, lw=1.5)

        # Local Axes (Red=Nose, Green=Right, Blue=Roof)
        fwd = qrot(rot, np.array([1.3, 0.0, 0.0]))
        rgt = qrot(rot, np.array([0.0, 0.7, 0.0]))
        up = qrot(rot, np.array([0.0, 0.0, 0.7]))

        ax.plot([0, fwd[0]], [0, fwd[1]], [0, fwd[2]], color="#FF2255", lw=2.2)
        ax.plot([0, rgt[0]], [0, rgt[1]], [0, rgt[2]], color="#00FF66", lw=1.8)
        ax.plot([0, up[0]], [0, up[1]], [0, up[2]], color="#3388FF", lw=1.8)

        status_str = f"SETTLED ({t_settle:.3f}s)" if status == "SETTLED" else "ACTIVE"
        ax.text2D(0.05, 0.92, f"{title}\n[{status_str}]", transform=ax.transAxes, color=color, fontsize=10,
                  weight="bold")

    max_frames = max(len(results_dict[name]["hist"]) for name in solvers)
    stride = 4  # 120 FPS -> 30 FPS animation
    frame_indices = list(range(0, max_frames + 20, stride))

    def update(frame_idx):
        for i, name in enumerate(solvers):
            hist = results_dict[name]["hist"]
            item = hist[min(frame_idx, len(hist) - 1)]
            t_s = results_dict[name]["t_settle"]
            draw_car_frame(axes_3d[i], item["rot"], name, results_dict[name]["color"], item["status"], t_s)

    anim = FuncAnimation(fig, update, frames=frame_indices, interval=33)
    anim.save(filename, writer=PillowWriter(fps=30))
    plt.close(fig)


# ─── Cache Helpers ────────────────────────────────────────────────────────────
def _cache_is_valid(cache: dict) -> bool:
    """Return True if the cached metadata matches the current configuration."""
    meta = cache.get("meta")
    if meta is None:
        return False
    return (
            meta.get("tol_angle") == TOL_ANGLE
            and meta.get("tol_speed") == TOL_SPEED
    )


def _save_cache(cache_path: str, cache_data: dict) -> None:
    with open(cache_path, "wb") as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_cache(cache_path: str) -> dict:
    if not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, EOFError, OSError) as exc:
        print(f"[cache] Could not load '{cache_path}' ({exc}). Starting fresh.")
        return {}


# ─── Main Benchmark Loop ───────────────────────────────────────────────────────
def main():
    if SAVE_ALL_GIFS:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    target_rot = np.array([0., 0., 0., 1.])
    completion_times = {name: [] for name in ACTIVE_SOLVERS}
    compute_times    = {name: [] for name in ACTIVE_SOLVERS}

    # ← cache: load & validate
    cache = _load_cache(CACHE_FILE)
    if cache and not _cache_is_valid(cache):
        print("[cache] Configuration changed – cache invalidated, starting fresh.")
        cache = {}
    cache.setdefault("meta", {
        "n_samples": N_SAMPLES,
        "tol_angle": TOL_ANGLE, "tol_speed": TOL_SPEED,
        "solvers": list(ACTIVE_SOLVERS.keys()),
    })
    cache.setdefault("runs", {})
    n_cached = len(cache["runs"])
    if n_cached:
        print(f"[cache] Loaded {n_cached}/{N_SAMPLES} cached runs from '{CACHE_FILE}'.")

    print(f"==========================================================================")
    print(f"Running Multi-Solver Benchmark ({N_SAMPLES} Runs @ 120 FPS)")
    print(f"Tolerances: Angle < {TOL_ANGLE*(180/pi):.2f}° ({TOL_ANGLE:.3f} rad), Speed < {TOL_SPEED:.2f} rad/s")
    print(f"Active Solvers: {list(ACTIVE_SOLVERS.keys())}")
    print(f"GIF Directory:  {os.path.abspath(OUTPUT_DIR) if SAVE_ALL_GIFS else 'Disabled'}")
    print(f"==========================================================================\n")

    solver_names = list(ACTIVE_SOLVERS.keys())
    col_width = 14
    header = f"{'Run':<5} | " + " | ".join([f"{name[:col_width]:<{col_width}}" for name in solver_names]) + " | Fastest Solver"
    print(header)
    print("-" * len(header))

    for i in range(START_SAMPLE, N_SAMPLES):
        cached_run = cache["runs"].get(i)

        # Decide which solvers must (re)run:
        #   • not present in cache, OR
        #   • flagged with "force": True in ACTIVE_SOLVERS
        needs_run = {
            name
            for name, cfg in ACTIVE_SOLVERS.items()
            if cfg.get("force", False)
               or cached_run is None
               or name not in cached_run["results"]
        }

        # ── Fast path: nothing to do ──────────────────────────────────
        if not needs_run:
            q0, w0 = cached_run["q0"], cached_run["w0"]
            histories, run_times = {}, {}
            for name in ACTIVE_SOLVERS:
                r = cached_run["results"][name]
                completion_times[name].append(r["t_finish"])
                compute_times[name].append(r["compute_time"])
                run_times[name] = r["t_finish"]
                histories[name] = {"hist": r["hist"],
                                   "color": ACTIVE_SOLVERS[name]["color"],
                                   "t_settle": r["t_finish"]}

            fastest = min(run_times, key=run_times.get)
            st = sorted(run_times.values())
            lead = st[1] - st[0] if len(st) > 1 else 0.0
            ts = " | ".join(f"{run_times[n]:>{col_width - 2}.4f} s" for n in solver_names)
            print(f"{i + 1:<5} | {ts} | {fastest} (+{lead * SIM_FPS:.1f}f)  [cached]")

            if SAVE_ALL_GIFS:
                render_comparison_animation(histories,
                                            os.path.join(OUTPUT_DIR, f"run_{i + 1:03d}.gif"))
            continue

        # ── Partial / full re-run ─────────────────────────────────────
        # q0, w0 come from the cache when available (deterministic seed
        # means they'd be identical either way, but this avoids calling
        # random_state() and keeps the stream aligned).
        if cached_run is not None:
            q0, w0 = cached_run["q0"], cached_run["w0"]
        else:
            q0, w0 = random_state()

        histories, run_times = {}, {}

        for name, cfg in ACTIVE_SOLVERS.items():
            if name in needs_run:
                t0 = time.perf_counter()
                t_finish, hist = cfg["fn"](q0, w0, target_rot,
                                           tol_angle=TOL_ANGLE,
                                           tol_speed=TOL_SPEED)
                comp_time = time.perf_counter() - t0
            else:
                r = cached_run["results"][name]
                t_finish, hist, comp_time = r["t_finish"], r["hist"], r["compute_time"]

            run_times[name] = t_finish
            histories[name] = {"hist": hist, "color": cfg["color"], "t_settle": t_finish}
            completion_times[name].append(t_finish)
            compute_times[name].append(comp_time)

        # Merge fresh results into the cache entry (overwrites forced solvers,
        # keeps any previously-cached solvers that weren't re-run).
        if cached_run is None:
            cache["runs"][i] = {"q0": q0, "w0": w0, "results": {}}
        for name in needs_run:
            cache["runs"][i]["results"][name] = {
                "t_finish": run_times[name],
                "hist": histories[name]["hist"],
                "compute_time": compute_times[name][-1],
            }
        _save_cache(CACHE_FILE, cache)

        fastest = min(run_times, key=run_times.get)
        st = sorted(run_times.values())
        lead = st[1] - st[0] if len(st) > 1 else 0.0
        ts = " | ".join(f"{run_times[n]:>{col_width - 2}.4f} s" for n in solver_names)
        forced_tag = f"  [re-ran: {', '.join(sorted(needs_run))}]" if needs_run else ""
        print(f"{i + 1:<5} | {ts} | {fastest} (+{lead * SIM_FPS:.1f}f){forced_tag}")

        if SAVE_ALL_GIFS:
            render_comparison_animation(histories,
                                        os.path.join(OUTPUT_DIR, f"run_{i + 1:03d}.gif"))

    # ─── Solution Quality Summary ────────────────────────────────
    print("\n" + "=" * 75)
    print(f"                       BENCHMARK SUMMARY (N = {N_SAMPLES})")
    print("=" * 75)
    print(f"{'Solver':<22} | {'Mean Time':<10} | {'Median Time':<12} | {'Min Time':<10} | {'Max Time':<10}")
    print("-" * 75)

    ranked_solvers = sorted(ACTIVE_SOLVERS.keys(), key=lambda n: np.mean(completion_times[n]))
    for rank, name in enumerate(ranked_solvers, 1):
        times = completion_times[name]
        print(
            f"#{rank} {name:<19} | {np.mean(times):>8.4f} s | {np.median(times):>10.4f} s | {np.min(times):>8.4f} s | {np.max(times):>8.4f} s")
    print("-" * 75)

    # ─── Computation Time Summary ──────────────────────────────────────
    print("\n" + "=" * 75)
    print(f"              SOLVER COMPUTATION TIME (wall-clock, N = {N_SAMPLES})")
    print("=" * 75)
    print(f"{'Rank':<5} {'Solver':<22} | {'Mean':<10} | {'Median':<10} | {'Min':<10} | {'Max':<10} | {'P95':<10}")
    print("-" * 75)

    ranked_compute = sorted(ACTIVE_SOLVERS.keys(), key=lambda n: np.mean(compute_times[n]))
    for rank, name in enumerate(ranked_compute, 1):
        ct = compute_times[name]
        p95 = np.percentile(ct, 95)
        print(
            f"#{rank:<4} {name:<22} | {np.mean(ct):>7.4f} s | {np.median(ct):>8.4f} s | {np.min(ct):>8.4f} s | {np.max(ct):>8.4f} s | {p95:>8.4f} s")
    print("-" * 75)

    # Composite score: normalise each solver's mean solution time and mean compute time
    # to [0, 1] across solvers, then average.  Lower is better.
    mean_sol = np.array([np.mean(completion_times[n]) for n in ACTIVE_SOLVERS])
    mean_comp = np.array([np.mean(compute_times[n]) for n in ACTIVE_SOLVERS])

    sol_norm = (mean_sol - mean_sol.min()) / max(mean_sol.max() - mean_sol.min(), 1e-12)
    comp_norm = (mean_comp - mean_comp.min()) / max(mean_comp.max() - mean_comp.min(), 1e-12)
    score = 0.5 * sol_norm + 0.5 * comp_norm
    score_map = dict(zip(solver_names, score))

    print(f"\nComposite Score (0 = best, 1 = worst;  50 % solution time + 50 % compute time):")
    ranked_score = sorted(solver_names, key=score_map.__getitem__)
    for rank, name in enumerate(ranked_score, 1):
        bar = "█" * int(score_map[name] * 30) + "░" * (30 - int(score_map[name] * 30))
        print(f"  #{rank}  {name:<18}  {score_map[name]:.4f}  {bar}")
    print("=" * 75)

    if SAVE_ALL_GIFS:
        print(f"All {N_SAMPLES} comparison GIFs saved to '{os.path.abspath(OUTPUT_DIR)}/'.")


if __name__ == "__main__":
    np.random.seed(42)
    main()
