"""학습된 High-level Wrist Planner 추론 래퍼.

저수준 env가 online으로 호출:
  planner = WristPlanner(ckpt_path, device)
  rh_p, rh_aa, lh_p, lh_aa = planner.generate(cap_T, body_T)   # gym 프레임 양손 손목궤적

- 정준화는 '입력 goal 궤적의' body 초기포즈 기준(병 위치 불변).
- 슬라이딩 윈도우 예측 → 겹침 평균 스티칭 → 전체 N프레임 손목궤적.
- torch 텐서 입출력도 지원(env에서 바로 쓰기 위함).
"""
import os
import numpy as np
import torch

from features import obj_features_from_arrays, wrist_feat_to_world, Normalizer
from model import WristPlannerTransformer

HERE = os.path.dirname(os.path.abspath(__file__))


class WristPlanner:
    def __init__(self, ckpt_path, device="cuda:0"):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.cfg = ck["cfg"]
        self.T = self.cfg["horizon"]
        self.device = device
        self.obj_norm = Normalizer.from_state(ck["obj_norm"])
        self.wrist_norm = Normalizer.from_state(ck["wrist_norm"])
        self.model = WristPlannerTransformer(
            18, 18, self.cfg["d_model"], self.cfg["nhead"],
            self.cfg["layers"], self.cfg["ff"], 1, self.cfg["dropout"],
        ).to(device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()

    @classmethod
    def from_idx(cls, idx, device="cuda:0"):
        p = os.path.join(HERE, "runs", f"wp_{idx.replace('@','_')}", "best.pth")
        return cls(p, device)

    @torch.no_grad()
    def generate(self, cap_T, body_T):
        """cap_T, body_T: (N,4,4) gym 프레임 → (rh_p,rh_aa,lh_p,lh_aa) 각 (N,3)."""
        if torch.is_tensor(cap_T):
            cap_T = cap_T.detach().cpu().numpy()
        if torch.is_tensor(body_T):
            body_T = body_T.detach().cpu().numpy()
        obj_feat, T_ref = obj_features_from_arrays(cap_T, body_T)
        N = len(obj_feat)
        Xn = self.obj_norm.norm(obj_feat)
        T = self.T
        acc = np.zeros((N, 18), np.float32); cnt = np.zeros((N, 1), np.float32)
        cat = torch.zeros(1, dtype=torch.long, device=self.device)
        for i in range(N - T + 1):
            win = torch.tensor(Xn[i:i + T][None], dtype=torch.float32, device=self.device)
            pred = self.model(win, cat)[0].cpu().numpy()
            acc[i:i + T] += self.wrist_norm.denorm(pred); cnt[i:i + T] += 1
        # 끝쪽 짧은 구간(윈도우 부족) 보정: 마지막 유효 윈도우로 채움
        if N < T:
            win = torch.tensor(Xn[None], dtype=torch.float32, device=self.device)
            pred = self.model(win, cat)[0].cpu().numpy()
            acc[:] = self.wrist_norm.denorm(pred)[:N]; cnt[:] = 1
        pred_feat = acc / np.clip(cnt, 1, None)
        return wrist_feat_to_world(pred_feat, T_ref)


if __name__ == "__main__":
    # 자체검증: 학습 npz의 goal로 생성한 손목이 eval.py 결과와 일치하는지
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--data_idx", default="97fc3@1"); a = ap.parse_args()
    npz = np.load(os.path.join(HERE, "datasets", f"{a.data_idx}.npz"), allow_pickle=True)
    pl = WristPlanner.from_idx(a.data_idx)
    rh_p, rh_aa, lh_p, lh_aa = pl.generate(npz["cap_T"], npz["body_T"])
    from scipy.spatial.transform import Rotation
    for h, p, aa in [("RH", rh_p, rh_aa), ("LH", lh_p, lh_aa)]:
        pe = np.linalg.norm(p - npz[f"{h[0].lower()}h_wpos"], axis=1) * 100
        re = np.degrees((Rotation.from_rotvec(aa) * Rotation.from_rotvec(npz[f"{h[0].lower()}h_waa"]).inv()).magnitude())
        print(f"[{h}] vs GT: 위치 mean={pe.mean():.2f}cm max={pe.max():.2f}cm | 회전 mean={re.mean():.1f}deg max={re.max():.1f}deg")
