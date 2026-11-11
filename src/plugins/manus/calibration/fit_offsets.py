# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fit Manus wrist offsets from a dynamic capture, with validation.

The capture must be recorded with the offsets CURRENTLY COMPILED into the plugin -
this script reads them out of the source so the recovered Manus-local joint
positions are correct. Mixing those up silently biases every result.

    uv run --active --with scipy python fit_offsets.py \
        --dynamic <dir>/dynamic.npz [--static-dir <dir>] [--label "rev B gloves"]
"""

import argparse
import glob
import json
import os
import re

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

TIPS = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip"]
SRC_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           os.pardir, "core", "manus_hand_tracking_plugin.cpp")
STATIC_PAIR = {
    "middle_tips": "middle_tip", "little_tips": "little_tip", "index_tips": "index_tip",
    "thumb_tips": "thumb_tip", "palms_together": "middle_tip",
}


def read_compiled_offset(path):
    """Parse kRightHandOffset (quat xyzw, then position) out of the plugin source."""
    txt = open(path).read()
    m = re.search(
        r"kRightHandOffset\s*=\s*\{\s*\{([^}]*)\}\s*,\s*\{([^}]*)\}\s*\}", txt, re.S
    )
    if not m:
        raise SystemExit(f"could not parse kRightHandOffset from {path}")
    q = [float(x) for x in re.findall(r"-?\d*\.?\d+", m.group(1))]
    t = [float(x) for x in re.findall(r"-?\d*\.?\d+", m.group(2))]
    if len(q) != 4 or len(t) != 3:
        raise SystemExit(f"unexpected offset format: q={q} t={t}")
    return np.array(q), np.array(t)


class Model:
    """Maps measured joints to the offset-local frame and back under new offsets."""

    def __init__(self, q_cap, t_cap):
        self.t_cap = t_cap
        self.R_cap = {
            "right": Rot.from_quat(q_cap),
            "left": Rot.from_quat([q_cap[0], -q_cap[1], -q_cap[2], q_cap[3]]),
        }

    def _mirror(self, v, side, flip):
        return v * flip if side == "left" else v

    def t_at_capture(self, side):
        return self._mirror(self.t_cap, side, np.array([-1, 1, 1]))

    def to_local(self, side, aim_p, aim_R, world):
        return self.R_cap[side].inv().apply(
            aim_R.inv().apply(world - aim_p) - self.t_at_capture(side)
        )

    def to_world(self, side, aim_p, aim_R, local, t, rv):
        tt = self._mirror(t, side, np.array([-1, 1, 1]))
        rr = self._mirror(rv, side, np.array([1, -1, -1]))
        return aim_p + aim_R.apply(tt + Rot.from_rotvec(rr).apply(local))


def load_dynamic(path, model):
    d = np.load(path)
    aim_p = {s: d[f"{s}_aim_pos"] for s in ("left", "right")}
    aim_R = {s: Rot.from_quat(d[f"{s}_aim_quat"]) for s in ("left", "right")}
    loc = {
        (s, t): model.to_local(s, aim_p[s], aim_R[s], d[f"{s}_{t}_pos"])
        for s in ("left", "right") for t in TIPS
    }
    return aim_p, aim_R, loc, len(d["left_aim_pos"])


def main():
    global TIPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--dynamic", required=True)
    ap.add_argument("--static-dir", default=None)
    ap.add_argument("--source", default=SRC_DEFAULT)
    ap.add_argument("--label", default="new hardware")
    ap.add_argument("--reg", type=float, default=None, help="rotation regularisation; default = auto-select")
    ap.add_argument("--joints", default=",".join(TIPS),
                    help="comma-separated joint pairs to constrain. For a prayer-pose capture "
                         "add 'wrist' - it anchors wrist orientation, which fingertips alone cannot.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    TIPS = [j.strip() for j in args.joints.split(",")]
    q_cap, t_cap = read_compiled_offset(args.source)
    model = Model(q_cap, t_cap)
    rv_cap = model.R_cap["right"].as_rotvec()
    aim_p, aim_R, loc, N = load_dynamic(args.dynamic, model)
    print(f"capture-time offsets (from source): quat {np.round(q_cap,3)}  pos {np.round(t_cap,4)}")
    print(f"dynamic frames: {N}")

    def gaps(par, idx):
        return np.stack([
            np.linalg.norm(
                model.to_world("left", aim_p["left"][idx], aim_R["left"][idx], loc[("left", t)][idx], par[:3], par[3:6])
                - model.to_world("right", aim_p["right"][idx], aim_R["right"][idx], loc[("right", t)][idx], par[:3], par[3:6]),
                axis=1) for t in TIPS])

    # static captures are an independent check - never fitted on
    static = {}
    if args.static_dir:
        for f in sorted(glob.glob(os.path.join(args.static_dir, "cal_*.json"))):
            name = os.path.basename(f)[4:-5]
            if name in STATIC_PAIR:
                static[name] = json.load(open(f))

    def static_gap(js, j, par):
        o = []
        for s in ("left", "right"):
            a = np.array(js[f"{s}_aim_pos"]); R = Rot.from_quat(js[f"{s}_aim_quat"])
            l = model.to_local(s, a, R, np.array(js[f"{s}_{j}_pos"]))
            o.append(model.to_world(s, a, R, l, par[:3], par[3:6]))
        return np.linalg.norm(o[0] - o[1]) * 100

    def static_scores(par):
        if not static:
            return None, None
        v = [static_gap(js, STATIC_PAIR[k], par) for k, js in static.items()]
        return float(np.mean(v)), float(np.max(v))

    ALL = np.arange(N)
    x0 = np.concatenate([t_cap, rv_cap, np.full(len(TIPS), 0.02)])

    def fit(idx, reg):
        def r(x):
            res = (gaps(x[:6], idx) - x[6:][:, None]).ravel()
            return np.concatenate([res, [reg * np.linalg.norm(x[3:6] - rv_cap)]])
        return least_squares(r, x0).x

    # choose regularisation by held-out dynamic error, penalising static regression
    tr, te = ALL[::2], ALL[1::2]
    print(f"\n{'reg':>6}  {'rot':>7}  {'held-out sd/mean':>18}  {'static mean/worst':>18}")
    cur = np.concatenate([t_cap, rv_cap])
    gc = gaps(cur, te) * 100
    sm, sw = static_scores(cur)
    stat_cur = f"{sm:5.2f} / {sw:5.2f}" if sm else "   n/a"
    print(f"{'current':>6}  {0.0:6.2f}d  {gc.std(axis=1).mean():7.2f} / {gc.mean():5.2f}  {stat_cur:>18}")
    cands = []
    for reg in [10.0, 3.0, 1.0, 0.3, 0.0]:
        x = fit(tr, reg)
        g = gaps(x[:6], te) * 100
        dang = np.degrees((Rot.from_rotvec(x[3:6]).inv() * Rot.from_rotvec(rv_cap)).magnitude())
        sm2, sw2 = static_scores(x[:6])
        s = f"{sm2:5.2f} / {sw2:5.2f}" if sm2 else "   n/a"
        print(f"{reg:6.1f}  {dang:6.2f}d  {g.std(axis=1).mean():7.2f} / {g.mean():5.2f}  {s:>18}")
        # score: dynamic held-out mean, rejecting any static worst-case regression
        bad = (sw2 is not None and sw2 > sw * 1.05)
        cands.append((g.mean() + (1e3 if bad else 0), reg))
    reg = args.reg if args.reg is not None else min(cands)[1]
    print(f"\nselected reg = {reg}" + ("" if args.reg is not None else "  (auto: best held-out without static regression)"))

    xf = fit(ALL, reg)
    q = Rot.from_rotvec(xf[3:6]).as_quat()
    if q[3] < 0:
        q = -q
    t = xf[:3]
    dang = np.degrees((Rot.from_rotvec(xf[3:6]).inv() * Rot.from_rotvec(rv_cap)).magnitude())

    gc, gf = gaps(cur, ALL) * 100, gaps(xf[:6], ALL) * 100
    lines = [f"Manus wrist offsets - {args.label}", "=" * 60,
             f"fitted from {N} dynamic frames" + (f" + {len(static)} static poses (validation only)" if static else ""),
             "", "dynamic fingertip-contact gap, cm (mean +/- sd):"]
    for i, tname in enumerate(TIPS):
        lines.append(f"  {tname:11s} {gc[i].mean():5.2f}+/-{gc[i].std():4.2f}  ->  {gf[i].mean():5.2f}+/-{gf[i].std():4.2f}")
    if static:
        lines += ["", "static poses (independent of the fit), cm:"]
        for k, js in static.items():
            j = STATIC_PAIR[k]
            lines.append(f"  {k:16s} {static_gap(js,j,cur):5.2f}  ->  {static_gap(js,j,xf[:6]):5.2f}")
    lines += ["", f"rotation change from capture-time offsets: {dang:.2f} deg",
              f"fitted contact distances (cm): {np.round(xf[6:]*100,2).tolist()}",
              "", "Paste into src/plugins/manus/core/manus_hand_tracking_plugin.cpp:", ""]
    lines.append(f"static constexpr XrPosef kLeftHandOffset = {{ {{ {q[0]:.3f}f, {-q[1]:.3f}f, {-q[2]:.3f}f, {q[3]:.3f}f }},")
    lines.append(f"                                             {{ {-t[0]:.3f}f, {t[1]:.3f}f, {t[2]:.3f}f }} }};")
    lines.append(f"static constexpr XrPosef kRightHandOffset = {{ {{ {q[0]:.3f}f, {q[1]:.3f}f, {q[2]:.3f}f, {q[3]:.3f}f }},")
    lines.append(f"                                              {{ {t[0]:.3f}f, {t[1]:.3f}f, {t[2]:.3f}f }} }};")
    txt = "\n".join(lines)
    print("\n" + txt)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.dynamic)), "offsets_handoff.txt")
    open(out, "w").write(txt + "\n")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
