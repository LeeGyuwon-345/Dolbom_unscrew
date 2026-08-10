"""Summarize wrist poses that succeeded in cap grasp RL.

Usage:
    python task1/tools/summarize_cap_grasp_wrist_logs.py --top 20
    python task1/tools/summarize_cap_grasp_wrist_logs.py --success-csv task1/logs/cap_grasp_success_123.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path("/home/leegyuwon/Documents/task1")
LOG_DIR = ROOT / "logs"


def latest_success_csv(log_dir: Path) -> Path:
    files = sorted(log_dir.glob("cap_grasp_success_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No cap_grasp_success_*.csv found in {log_dir}")
    return files[0]


def f(row, key):
    return float(row[key])


def pose_key(row, pos_bin, dir_bin):
    rel = [f(row, "wrist_rel_cap_x"), f(row, "wrist_rel_cap_y"), f(row, "wrist_rel_cap_z")]
    palm = [f(row, "palm_dir_x"), f(row, "palm_dir_y"), f(row, "palm_dir_z")]
    return tuple(round(v / pos_bin) for v in rel) + tuple(round(v / dir_bin) for v in palm)


def mean(values):
    return sum(values) / max(1, len(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--success-csv", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--pos-bin", type=float, default=0.005, help="position bin size in meters")
    parser.add_argument("--dir-bin", type=float, default=0.05, help="palm direction bin size")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    src = args.success_csv or latest_success_csv(args.log_dir)
    with open(src, newline="") as fobj:
        rows = list(csv.DictReader(fobj))
    if not rows:
        print(f"No successful wrist poses in {src}")
        return

    groups = defaultdict(list)
    for row in rows:
        groups[pose_key(row, args.pos_bin, args.dir_bin)].append(row)

    summary = []
    for grouped in groups.values():
        latest = max(grouped, key=lambda r: int(float(r["train_env_frames"])))
        summary.append(
            {
                "success_count": len(grouped),
                "latest_train_env_frames": int(float(latest["train_env_frames"])),
                "wrist_pos_x": f(latest, "wrist_pos_x"),
                "wrist_pos_y": f(latest, "wrist_pos_y"),
                "wrist_pos_z": f(latest, "wrist_pos_z"),
                "wrist_quat_wxyz_w": f(latest, "wrist_quat_wxyz_w"),
                "wrist_quat_wxyz_x": f(latest, "wrist_quat_wxyz_x"),
                "wrist_quat_wxyz_y": f(latest, "wrist_quat_wxyz_y"),
                "wrist_quat_wxyz_z": f(latest, "wrist_quat_wxyz_z"),
                "palm_dir_x": mean([f(r, "palm_dir_x") for r in grouped]),
                "palm_dir_y": mean([f(r, "palm_dir_y") for r in grouped]),
                "palm_dir_z": mean([f(r, "palm_dir_z") for r in grouped]),
                "wrist_rel_cap_x": mean([f(r, "wrist_rel_cap_x") for r in grouped]),
                "wrist_rel_cap_y": mean([f(r, "wrist_rel_cap_y") for r in grouped]),
                "wrist_rel_cap_z": mean([f(r, "wrist_rel_cap_z") for r in grouped]),
                "avg_contact_count": mean([f(r, "contact_count") for r in grouped]),
                "avg_cap_speed": mean([f(r, "cap_speed") for r in grouped]),
            }
        )

    summary.sort(key=lambda r: (r["success_count"], r["latest_train_env_frames"]), reverse=True)
    top_rows = summary[: args.top]

    out = args.out or src.with_name(src.stem.replace("cap_grasp_success", "cap_grasp_success_pose_candidates") + ".csv")
    with open(out, "w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=list(top_rows[0].keys()))
        writer.writeheader()
        writer.writerows(top_rows)

    print(f"source: {src}")
    print(f"success rows: {len(rows)} | grouped candidates: {len(summary)}")
    print(f"saved: {out}")
    print("")
    for i, row in enumerate(top_rows, 1):
        pos = [row["wrist_pos_x"], row["wrist_pos_y"], row["wrist_pos_z"]]
        quat = [row["wrist_quat_wxyz_w"], row["wrist_quat_wxyz_x"], row["wrist_quat_wxyz_y"], row["wrist_quat_wxyz_z"]]
        palm = [row["palm_dir_x"], row["palm_dir_y"], row["palm_dir_z"]]
        rel = [row["wrist_rel_cap_x"], row["wrist_rel_cap_y"], row["wrist_rel_cap_z"]]
        print(
            f"{i:02d} count={row['success_count']:4d} "
            f"pos=({pos[0]:+.4f},{pos[1]:+.4f},{pos[2]:+.4f}) "
            f"quat_wxyz=({quat[0]:+.4f},{quat[1]:+.4f},{quat[2]:+.4f},{quat[3]:+.4f}) "
            f"palm=({palm[0]:+.3f},{palm[1]:+.3f},{palm[2]:+.3f}) "
            f"rel=({rel[0]:+.4f},{rel[1]:+.4f},{rel[2]:+.4f}) "
            f"contact={row['avg_contact_count']:.2f}"
        )


if __name__ == "__main__":
    main()
