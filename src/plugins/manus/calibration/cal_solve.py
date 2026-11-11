# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Solve the Manus wrist offset from the captured calibration steps.

Each step_*.npz carries the offsets that were compiled in when it was recorded,
so files from different builds mix safely and nothing ever needs rebasing.

What each step contributes - they are not interchangeable:

  palms    Palms flat together. The WRIST joint's Manus-local position is ~0, so
           the wrist gap measures the TRANSLATION with no hand model involved at
           all - the only model-free measurement available. Palm-plane
           parallelism then pins 2 of the 3 rotation axes.
  point    Middle fingers pointing at each other. Supplies the third rotation
           axis - spin within the palm plane - which palms-together cannot see
           (both hands' palms stay flat however you spin them about that axis).
  pinkies  Little-finger edges together. Sharpens the lateral axis, which palms
           constrains most weakly.
  thumbs   Thumb edges together. Opposite lateral sensitivity to pinkies; only
           include it if the hands were NOT tilted, or it fights the pinky data.

  tipsweep All five fingertips in contact while the fingers open and close.
           MANUS Metagloves track fingertips MAGNETICALLY and directly, so tips
           are a primary measurement, not an accumulation of joint angles.
           Sweeping the fingers varies the hand CONFIGURATION, adding geometric
           diversity that rotating the hands cannot.

History worth knowing: an earlier round found the five fingers demanding offsets
26-267 mm apart while each was individually precise to 2-5 mm, and the left glove
reporting fingers ~9% longer than the right. That was traced to one glove being
miscalibrated. After recalibration the fingertip data should be trustworthy -
cal_solve prints the per-finger agreement so you can confirm that before relying
on it, rather than assuming either way.

    .venv/bin/python src/plugins/manus/calibration/cal_solve.py [--steps palms,point,pinkies] [--apply]
"""

import argparse
import os
import shutil
import subprocess

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

from fit_offsets import Model, read_compiled_offset, SRC_DEFAULT

PALM_JOINTS = ["wrist", "index_knuckle", "middle_knuckle", "ring_knuckle", "little_knuckle"]
# joint -> plausible separation (metres) between the two hands' joint CENTRES
# when that pose is held. Centres sit inside the flesh, so contact is not zero.
TARGET = {"wrist": 0.032, "index_knuckle": 0.025, "middle_knuckle": 0.025,
          "ring_knuckle": 0.025, "little_knuckle": 0.025, "little_meta": 0.025,
          "thumb_meta": 0.030, "thumb_prox": 0.030, "middle_tip": 0.020,
          "thumb_tip": 0.018, "index_tip": 0.018, "ring_tip": 0.018, "little_tip": 0.018}
STEP_SPEC = {
    "palms":   dict(joints=PALM_JOINTS, plane=True, weight=1.0),
    "point":   dict(joints=["middle_tip"], antiparallel="middle", weight=1.0),
    "pinkies": dict(joints=["little_knuckle", "little_meta"], weight=1.0),
    "thumbs":  dict(joints=["thumb_meta", "thumb_prox"], weight=1.0),
    "tipsweep": dict(joints=["thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip"],
                     weight=0.7),
}


def load_step(path, name, stride):
    z = np.load(path)
    if "_offset_quat" not in z.files:
        raise SystemExit(f"{os.path.basename(path)} has no offset stamp - recapture it "
                         f"(it predates self-describing captures and cannot be placed safely)")
    model = Model(z["_offset_quat"], z["_offset_pos"])
    spec = STEP_SPEC[name]
    need = set(spec["joints"]) | {"wrist", "index_knuckle", "little_knuckle"}
    if spec.get("antiparallel"):
        need |= {f"{spec['antiparallel']}_tip", f"{spec['antiparallel']}_knuckle"}
    ap = {s: z[f"{s}_aim_pos"][::stride] for s in ("left", "right")}
    aR = {s: Rot.from_quat(z[f"{s}_aim_quat"][::stride]) for s in ("left", "right")}
    loc = {(s, j): model.to_local(s, ap[s], aR[s], z[f"{s}_{j}_pos"][::stride])
           for s in ("left", "right") for j in need if f"{s}_{j}_pos" in z.files}
    return dict(ap=ap, aR=aR, loc=loc, spec=spec, n=len(ap["left"]),
                stamp=(z["_offset_quat"], z["_offset_pos"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)))
    p.add_argument("--steps", default="palms,point,pinkies")
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--palm-bump", type=float, default=0.0,
                   help="cm to ADD to the palms-pose contact targets. The joint centres sit "
                        "inside the flesh, and how deep is a guess - raise this if the hands "
                        "visibly clip when the palms are together, lower it if they float apart.")
    p.add_argument("--apply", action="store_true", help="write the result into the plugin source")
    p.add_argument("--source", default=SRC_DEFAULT)
    args = p.parse_args()

    if args.palm_bump:
        for j in PALM_JOINTS:
            TARGET[j] += args.palm_bump / 100.0
        print(f"palms targets raised by {args.palm_bump:.1f} cm -> "
              + ", ".join(f"{j}:{TARGET[j]*100:.1f}" for j in PALM_JOINTS))

    M0 = Model(*read_compiled_offset(args.source))      # only for reporting "current"
    q_cur, t_cur = read_compiled_offset(args.source)
    rv_cur = Rot.from_quat(q_cur).as_rotvec()

    steps = {}
    for name in [s.strip() for s in args.steps.split(",")]:
        f = os.path.join(args.dir, f"step_{name}.npz")
        if not os.path.exists(f):
            print(f"  (skipping {name}: not captured)")
            continue
        steps[name] = load_step(f, name, args.stride)
        q, t = steps[name]["stamp"]
        print(f"  {name:8s} {steps[name]['n']:5d} frames   recorded under quat "
              f"{np.round(q,3)} pos {np.round(t,4)}")
    if "palms" not in steps:
        raise SystemExit("the palms step is required - it carries the only model-free measurement")

    def W(st, s, j, t, rv):
        d = steps[st]
        return Model(*d["stamp"]).to_world(s, d["ap"][s], d["aR"][s], d["loc"][(s, j)], t, rv)

    def gap(st, j, t, rv):
        return np.linalg.norm(W(st, "left", j, t, rv) - W(st, "right", j, t, rv), axis=1)

    def plane(st, t, rv):
        o = []
        for s in ("left", "right"):
            w = W(st, s, "wrist", t, rv)
            n = np.cross(W(st, s, "index_knuckle", t, rv) - w, W(st, s, "little_knuckle", t, rv) - w)
            o.append(n / np.linalg.norm(n, axis=1, keepdims=True))
        return np.arccos(np.clip(np.sum(o[0] * o[1], axis=1), -1, 1))

    def fang(st, finger, t, rv):
        v = {}
        for s in ("left", "right"):
            d = W(st, s, f"{finger}_tip", t, rv) - W(st, s, f"{finger}_knuckle", t, rv)
            v[s] = d / np.linalg.norm(d, axis=1, keepdims=True)
        return np.arccos(np.clip(np.sum(v["left"] * v["right"], axis=1), -1, 1))

    def resid(x):
        """Each TERM is normalised by sqrt(#residuals) so that a term's influence
        comes from its weight, not from how many frames or landmarks it happens to
        contain. Without this the palm-plane term (one scalar) is outvoted
        thousands-to-one by the per-frame gap terms and simply collapses."""
        t, rv = x[:3], x[3:6]
        r = []

        def add(vec, w):
            vec = np.atleast_1d(vec)
            r.append(w * vec / np.sqrt(len(vec)))

        for name, d in steps.items():
            w = d["spec"]["weight"]
            for j in d["spec"]["joints"]:
                add(gap(name, j, t, rv) - TARGET[j], w)
            if d["spec"].get("plane"):
                # per-frame, not a single mean - this is the pose's orientation signal
                add(plane(name, t, rv), 3.0 * w)
            if d["spec"].get("antiparallel"):
                add(fang(name, d["spec"]["antiparallel"], t, rv) - np.pi, 3.0 * w)
        return np.concatenate(r)

    sol = least_squares(resid, np.concatenate([t_cur, rv_cur])).x
    t, rv = sol[:3], sol[3:6]

    print("\nresult per step (current -> fitted):")
    for name, d in steps.items():
        for j in d["spec"]["joints"]:
            print(f"  {name+'/'+j:24} {gap(name,j,t_cur,rv_cur).mean()*100:6.2f} -> "
                  f"{gap(name,j,t,rv).mean()*100:5.2f} cm   (target {TARGET[j]*100:.1f})")
        if d["spec"].get("plane"):
            print(f"  {name+'/palm-plane':24} {np.degrees(plane(name,t_cur,rv_cur)).mean():6.1f} -> "
                  f"{np.degrees(plane(name,t,rv)).mean():5.1f} deg  (target 0)")
        if d["spec"].get("antiparallel"):
            print(f"  {name+'/pointing':24} {np.degrees(fang(name,d['spec']['antiparallel'],t_cur,rv_cur)).mean():6.1f} -> "
                  f"{np.degrees(fang(name,d['spec']['antiparallel'],t,rv)).mean():5.1f} deg  (target 180)")

    # Per-finger agreement: does each finger independently imply the same offset?
    # This is the check that exposed the miscalibrated glove last time.
    if "tipsweep" in steps:
        from scipy.optimize import least_squares as _ls
        print("\nper-finger agreement (each finger solved on its own):")
        sols = {}
        for j in STEP_SPEC["tipsweep"]["joints"]:
            rr = lambda x, j=j: gap("tipsweep", j, x[:3], x[3:6]) - x[6]
            sols[j] = _ls(rr, np.concatenate([t_cur, rv_cur, [0.018]])).x
        med = np.median([v[:3] for v in sols.values()], axis=0)
        for j, v in sols.items():
            print(f"  {j:12} implies t {np.round(v[:3]*100,2)} cm   "
                  f"{np.linalg.norm(v[:3]-med)*1000:5.0f} mm from consensus")
        spread = max(np.linalg.norm(a[:3]-b[:3]) for a in sols.values() for b in sols.values()) * 1000
        verdict = ("fingers agree - tip data is trustworthy" if spread < 25 else
                   "fingers DISAGREE - something is still wrong with the hand model")
        print(f"  max pairwise spread {spread:.0f} mm  ->  {verdict}")

    # how well each axis is pinned down by the steps actually supplied
    J = np.zeros((len(resid(sol)), 6))
    for i in range(6):
        e = np.zeros(6); e[i] = 1e-6
        J[:, i] = (resid(np.r_[sol[:6] + e, sol[6:]]) - resid(np.r_[sol[:6] - e, sol[6:]])) / 2e-6
    s2 = np.sum(resid(sol) ** 2) / max(1, len(resid(sol)) - 6)
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(J.T @ J)))
    print(f"\nprecision from these steps: translation {np.round(se[:3]*1000,1)} mm, "
          f"rotation {np.round(np.degrees(se[3:6]),2)} deg")

    q = Rot.from_rotvec(rv).as_quat()
    if q[3] < 0:
        q = -q
    print(f"\ntranslation {np.round(t_cur,4)} -> {np.round(t,4)}")
    print(f"rotation change {np.degrees((Rot.from_rotvec(rv).inv()*Rot.from_rotvec(rv_cur)).magnitude()):.2f} deg")
    left = (f"static constexpr XrPosef kLeftHandOffset = {{ {{ {q[0]:.3f}f, {-q[1]:.3f}f, {-q[2]:.3f}f, {q[3]:.3f}f }},\n"
            f"                                             {{ {-t[0]:.3f}f, {t[1]:.3f}f, {t[2]:.3f}f }} }};")
    right = (f"static constexpr XrPosef kRightHandOffset = {{ {{ {q[0]:.3f}f, {q[1]:.3f}f, {q[2]:.3f}f, {q[3]:.3f}f }},\n"
             f"                                              {{ {t[0]:.3f}f, {t[1]:.3f}f, {t[2]:.3f}f }} }};")
    print("\n" + left + "\n" + right)
    open(os.path.join(args.dir, "offsets_result.txt"), "w").write(left + "\n" + right + "\n")

    if args.apply:
        src = open(args.source).read()
        import re
        src = re.sub(r"static constexpr XrPosef kLeftHandOffset\s*=\s*\{[^;]*\};", left, src, count=1)
        src = re.sub(r"static constexpr XrPosef kRightHandOffset\s*=\s*\{[^;]*\};", right, src, count=1)
        shutil.copy(args.source, args.source + ".bak")
        open(args.source, "w").write(src)
        print(f"\napplied to {args.source} (backup at .bak)")
        print("now: cmake --build build --target manus_hand_plugin --parallel && "
              "cmake --install build --component manus")


if __name__ == "__main__":
    main()
