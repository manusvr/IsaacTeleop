# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Short, resumable Manus calibration capture - one step at a time.

Records ALL 26 OpenXR hand joints (not just fingertips), so any constraint can be
chosen later without recapturing. Runs for ~10 s by default and appends to the
step's file, so an unreliable link just means doing a few short runs instead of
one long one that can die halfway.

    cd <repo root>
    source ~/.cloudxr/run/cloudxr.env
    .venv/bin/python src/plugins/manus/calibration/cal_capture.py --step palms --no-launch-cloudxr-runtime
    .venv/bin/python src/plugins/manus/calibration/cal_capture.py --step palms --no-launch-cloudxr-runtime   # adds more
    .venv/bin/python <this file> --step tips  --no-launch-cloudxr-runtime

Steps:
  palms  palms pressed FLAT together, fingers straight. Constrains the WRISTS and
         KNUCKLES - landmarks one joint from the wrist, so they carry far less
         flex-sensor error than fingertips, and they pin wrist ORIENTATION.
  tips   five fingertips touching. Gives orientation variety but noisy landmarks.

Keep moving through many hand ORIENTATIONS during the run; that is what makes the
offset identifiable. MANUS Core must be closed and manus_hand_plugin running.
"""

import argparse
import os
import sys
import time

import numpy as np

import isaaccapture.deviceio as deviceio
import isaaccapture.oxr as oxr
from isaaccapture.cloudxr import CloudXRLauncher

from fit_offsets import read_compiled_offset, SRC_DEFAULT

# Full OpenXR hand-joint set. Joint 0 (PALM) is skipped: the plugin aliases it to
# the little fingertip, so it carries no palm information.
JOINTS = {
    "wrist": 1,
    "thumb_meta": 2, "thumb_prox": 3, "thumb_dist": 4, "thumb_tip": 5,
    "index_meta": 6, "index_knuckle": 7, "index_mid": 8, "index_dist": 9, "index_tip": 10,
    "middle_meta": 11, "middle_knuckle": 12, "middle_mid": 13, "middle_dist": 14, "middle_tip": 15,
    "ring_meta": 16, "ring_knuckle": 17, "ring_mid": 18, "ring_dist": 19, "ring_tip": 20,
    "little_meta": 21, "little_knuckle": 22, "little_mid": 23, "little_dist": 24, "little_tip": 25,
}
KNUCKLES = ["index_knuckle", "middle_knuckle", "ring_knuckle", "little_knuckle"]
TIPS = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip"]

STEPS = {
    "palms": ("PALMS PRESSED FLAT TOGETHER, fingers straight. Press the heels of "
              "the hands together too.", ["wrist"] + KNUCKLES),
    "tips": ("FIVE FINGERTIPS touching their opposite number.", TIPS),
    # These two differ only in which EDGE of the hand is in contact, so a lateral
    # (thumb<->pinky) offset error makes one read too close and the other too far.
    # That opposite sign is what pins the lateral axis down; the palms pose is
    # nearly blind to it because there both hands slide the same way.
    "pinkies": ("HANDS SIDE BY SIDE, palms down, the LITTLE-FINGER EDGES of the "
                "hands pressed together along their whole length.",
                ["little_knuckle", "little_meta", "wrist"]),
    "thumbs": ("HANDS SIDE BY SIDE, palms down, the THUMB EDGES of the hands "
               "pressed together along their whole length.",
               ["thumb_meta", "thumb_prox", "index_knuckle", "wrist"]),
    # Palms-together pins the palm PLANE but leaves rotation WITHIN that plane
    # almost free. Pointing the middle fingers straight at one another is the
    # pose that measures it: the two fingers must come out antiparallel, which
    # no amount of palm data can tell us.
    # Metagloves measure fingertips directly (magnetic), so tips are a primary
    # measurement here rather than an accumulation of joint angles. Sweeping the
    # fingers through their range while the tips stay in contact varies the hand
    # CONFIGURATION, which adds geometric diversity that rotation alone cannot.
    "tipsweep": ("ALL FIVE FINGERTIPS touching their opposite number. Now slowly "
                 "OPEN and CLOSE your fingers through their full range, keeping "
                 "every fingertip in contact the whole time.",
                 ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip"]),
    "point": ("MIDDLE FINGERS POINTING STRAIGHT AT EACH OTHER, fingertips "
              "touching - like two arrows meeting tip to tip. Keep both middle "
              "fingers straight and in line; let the other fingers curl away.",
              ["middle_tip", "middle_knuckle"]),
}

GREEN, RED, YEL, BOLD, RST = "\033[32m", "\033[31m", "\033[33m", "\033[1m", "\033[0m"


def grab(ht, ct, session):
    rec = {}
    for side, gh, gc in (("left", ht.get_left_hand, ct.get_left_controller),
                         ("right", ht.get_right_hand, ct.get_right_controller)):
        hd = gh(session)
        if hd is None:
            return None, f"{side} hand"
        c = gc(session)
        if c is None or not c.aim_pose.is_valid:
            return None, f"{side} controller"
        for jn, ji in JOINTS.items():
            jp = hd.joints.poses(ji)
            if not jp.is_valid:
                return None, f"{side} {jn}"
            p, q = jp.pose.position, jp.pose.orientation
            rec[f"{side}_{jn}_pos"] = [p.x, p.y, p.z]
            rec[f"{side}_{jn}_quat"] = [q.x, q.y, q.z, q.w]
        a = c.aim_pose.pose
        rec[f"{side}_aim_pos"] = [a.position.x, a.position.y, a.position.z]
        rec[f"{side}_aim_quat"] = [a.orientation.x, a.orientation.y, a.orientation.z, a.orientation.w]
    return rec, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=list(STEPS), required=True)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--settle", type=float, default=6.0)
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--fresh", action="store_true", help="discard any existing data for this step")
    CloudXRLauncher.add_launcher_arguments(ap)
    args = ap.parse_args()

    cue, watch = STEPS[args.step]
    path = os.path.join(args.outdir, f"step_{args.step}.npz")
    # Stamp the offsets that are compiled into the plugin RIGHT NOW into the file.
    # Joint positions only mean something relative to the offset that produced
    # them; recording it here makes every capture self-describing, so the solver
    # never has to be told (or guess) which build a file came from.
    cap_q, cap_t = read_compiled_offset(SRC_DEFAULT)
    print(f"{YEL}capture-time offsets: quat {np.round(cap_q,3)} pos {np.round(cap_t,4)}{RST}")
    ht, ct = deviceio.HandTracker(), deviceio.ControllerTracker()
    trackers = [ht, ct]
    exts = deviceio.DeviceIOSession.get_required_extensions(trackers)

    prev = {}
    if os.path.exists(path) and not args.fresh:
        z = np.load(path)
        if "_offset_quat" in z.files and not (
            np.allclose(z["_offset_quat"], cap_q) and np.allclose(z["_offset_pos"], cap_t)
        ):
            print(f"{RED}existing {os.path.basename(path)} was recorded under DIFFERENT offsets.{RST}")
            print(f"{RED}Appending would mix incompatible data. Re-run with --fresh to replace it.{RST}")
            return 1
        prev = {k: z[k] for k in z.files if not k.startswith("_offset")}
        print(f"{YEL}appending to {os.path.basename(path)} ({len(prev['left_aim_pos'])} frames already){RST}")

    with CloudXRLauncher.launch_context(args):
        with (oxr.OpenXRSession("ManusCalStep", exts) as xs,
              deviceio.DeviceIOSession.run(trackers, xs.get_handles()) as session):
            print(f"\n{BOLD}=== step: {args.step} ({args.duration:.0f}s) ==={RST}")
            print(f"{cue}")
            print(f"{YEL}Keep varying hand ORIENTATION throughout.{RST}\n")

            t_end = time.time() + args.settle
            while time.time() < t_end:
                _, reason = grab(ht, ct, session)
                s = f"{GREEN}tracking OK{RST}" if reason is None else f"{RED}waiting: {reason}{RST}"
                sys.stdout.write(f"\r  starting in {t_end - time.time():4.1f}s   {s}\033[K")
                sys.stdout.flush()
                session.update()
                time.sleep(1 / 60)
            sys.stdout.write("\r\033[K")

            frames, drops = [], {}
            t0 = time.time()
            while time.time() - t0 < args.duration:
                session.update()
                rec, reason = grab(ht, ct, session)
                if rec is None:
                    drops[reason] = drops.get(reason, 0) + 1
                else:
                    frames.append(rec)
                    mg = float(np.mean([
                        np.linalg.norm(np.array(rec[f"left_{j}_pos"]) - np.array(rec[f"right_{j}_pos"]))
                        for j in watch])) * 100
                    col = GREEN if mg < 4 else (YEL if mg < 8 else RED)
                    el = time.time() - t0
                    sys.stdout.write(f"\r  [{'#' * int(el / args.duration * 24):<24}] {el:4.1f}s  "
                                     f"{len(frames)} frames  contact {col}{mg:5.2f} cm{RST}\033[K")
                    sys.stdout.flush()
                time.sleep(1 / 60)
            sys.stdout.write("\r\033[K")

    if not frames:
        print(f"{RED}no valid frames ({dict(drops)}) - nothing written{RST}")
        return 1

    new = {k: np.array([f[k] for f in frames], dtype=float) for k in frames[0]}
    if prev:
        new = {k: np.concatenate([prev[k], new[k]]) for k in new}
    new["_offset_quat"] = cap_q
    new["_offset_pos"] = cap_t
    np.savez_compressed(path, **new)
    total = len(new["left_aim_pos"])
    print(f"captured {len(frames)} frames (dropped {sum(drops.values())} {dict(drops)})")
    print(f"contact gap this run (cm):")
    for j in watch:
        g = np.array([np.linalg.norm(np.array(f[f"left_{j}_pos"]) - np.array(f[f"right_{j}_pos"]))
                      for f in frames]) * 100
        print(f"  {j:15s} mean {g.mean():5.2f}  sd {g.std():5.2f}")
    print(f"\n{GREEN}{path} now holds {total} frames{RST}")
    print("Run again to add more, or move on to the other step.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
