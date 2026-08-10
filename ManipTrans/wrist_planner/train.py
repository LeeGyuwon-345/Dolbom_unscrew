"""High-level Wrist Planner BC 학습.

객체 목표궤적 window(18) → 양손 손목궤적 window(18) 를 Transformer로 회귀(MSE).
단일 task(97fc3@1) 정렬 데이터셋(prepare_dataset.py 출력)으로 학습.

실행:
  cd /home/leegyuwon/Documents/ManipTrans
  python wrist_planner/train.py --config wrist_planner/configs/97fc3.yaml
  (옵션 override: --epochs 4000 --lr 3e-4 ...)
"""
import os
import time
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from features import build_features, Normalizer
from model import WristPlannerTransformer

HERE = os.path.dirname(os.path.abspath(__file__))


def make_windows(feat, T):
    """(N,D) → (N-T+1, T, D) 슬라이딩 윈도우."""
    N = len(feat)
    return np.stack([feat[i:i + T] for i in range(N - T + 1)], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "97fc3.yaml"))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.lr is not None:
        cfg["lr"] = args.lr
    idx = cfg["data_idx"]
    T = cfg["horizon"]
    dev = args.device
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])

    npz = np.load(os.path.join(HERE, "datasets", f"{idx}.npz"), allow_pickle=True)
    obj_feat, wrist_feat, T_ref = build_features(npz)
    N = len(obj_feat)
    obj_norm = Normalizer(obj_feat)
    wrist_norm = Normalizer(wrist_feat)
    X = make_windows(obj_norm.norm(obj_feat), T).astype(np.float32)   # (W,T,18)
    Y = make_windows(wrist_norm.norm(wrist_feat), T).astype(np.float32)
    W = len(X)
    print(f"[data] N={N} 프레임 → {W} windows (T={T})")

    # train/val 분할 (랜덤)
    perm = np.random.permutation(W)
    n_val = max(1, int(W * cfg["val_ratio"]))
    val_i, tr_i = perm[:n_val], perm[n_val:]
    Xt = torch.tensor(X[tr_i], device=dev); Yt = torch.tensor(Y[tr_i], device=dev)
    Xv = torch.tensor(X[val_i], device=dev); Yv = torch.tensor(Y[val_i], device=dev)
    cat_t = torch.zeros(len(tr_i), dtype=torch.long, device=dev)
    cat_v = torch.zeros(len(val_i), dtype=torch.long, device=dev)

    model = WristPlannerTransformer(
        obj_dim=18, wrist_dim=18, d_model=cfg["d_model"], nhead=cfg["nhead"],
        layers=cfg["layers"], ff=cfg["ff"], n_cat=1, dropout=cfg["dropout"],
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])

    # 손실 가중치: 회전 6D 항에 rot_weight
    w = np.ones(18, dtype=np.float32)
    w[3:9] *= cfg["rot_weight"]; w[12:18] *= cfg["rot_weight"]
    wv = torch.tensor(w, device=dev)

    def loss_fn(pred, tgt):
        return (((pred - tgt) ** 2) * wv).mean()

    run_dir = os.path.join(HERE, "runs", f"wp_{idx.replace('@','_')}")
    os.makedirs(run_dir, exist_ok=True)
    sw = SummaryWriter(run_dir)
    bs = cfg["batch_size"]
    best_val = float("inf"); t0 = time.time()
    for ep in range(cfg["epochs"]):
        model.train()
        idxs = torch.randperm(len(Xt), device=dev)
        tot = 0.0
        for b in range(0, len(Xt), bs):
            bi = idxs[b:b + bs]
            pred = model(Xt[bi], cat_t[bi])
            loss = loss_fn(pred, Yt[bi])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(bi)
        sched.step()
        tr = tot / len(Xt)
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xv, cat_v), Yv).item()
        sw.add_scalar("loss/train", tr, ep); sw.add_scalar("loss/val", vl, ep)
        sw.add_scalar("lr", sched.get_last_lr()[0], ep)
        if vl < best_val:
            best_val = vl
            torch.save({
                "model": model.state_dict(), "cfg": cfg,
                "obj_norm": obj_norm.state(), "wrist_norm": wrist_norm.state(),
                "T_ref": T_ref, "epoch": ep, "val": vl,
            }, os.path.join(run_dir, "best.pth"))
        if ep % 100 == 0 or ep == cfg["epochs"] - 1:
            print(f"ep{ep:5d} train={tr:.5f} val={vl:.5f} best={best_val:.5f} "
                  f"({(time.time()-t0):.0f}s)", flush=True)
    print(f"[완료] best_val={best_val:.5f}  ckpt={run_dir}/best.pth")


if __name__ == "__main__":
    main()
