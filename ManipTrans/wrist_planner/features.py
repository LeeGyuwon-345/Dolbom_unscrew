"""손목 플래너 입출력 특징 변환.

- 객체중심(object-centric) 정규화: 모든 포즈를 몸통 초기프레임 T_ref = body_T[0] 기준으로 변환
  → 병이 어디 놓였는지에 불변. eval에서 T_ref로 되돌린다.
- 회전은 6D 표현(회전행렬 앞 두 열) 사용 — 연속성 좋아 회귀에 적합.

입력 특징(18): [cap_pos(3), cap_6d(6), body_pos(3), body_6d(6)]   (객체 목표궤적 G)
출력 특징(18): [rh_pos(3), rh_6d(6), lh_pos(3), lh_6d(6)]         (양손 손목 a^W)
"""
import numpy as np
from scipy.spatial.transform import Rotation


def rotmat_to_6d(R):
    # 앞 두 열 [col0(3), col1(3)]
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def sixd_to_rotmat(x):
    a, b = x[..., 0:3], x[..., 3:6]
    b1 = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b2 = b - (b1 * b).sum(-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)  # 열 벡터로 쌓음 → (...,3,3)


def _apply_inv(inv, p, R):
    """inv(4,4)를 (p(N,3),R(N,3,3)) 포즈열에 적용 → (p',R')."""
    W = np.tile(np.eye(4, dtype=np.float32), (len(p), 1, 1))
    W[:, :3, :3] = R
    W[:, :3, 3] = p
    W = np.einsum("ij,tjk->tik", inv, W)
    return W[:, :3, 3].astype(np.float32), W[:, :3, :3].astype(np.float32)


def obj_features_from_arrays(cap_T, body_T):
    """임의 goal 궤적(cap_T,body_T (N,4,4)) → (obj_feat(N,18), T_ref(4,4)).

    온라인 추론용: 정준화 프레임 T_ref는 '이 궤적의' body 초기포즈로 계산한다
    (학습 때의 T_ref가 아님 — 병 위치에 불변하게 하는 게 목적).
    """
    cap_T = np.asarray(cap_T, np.float32); body_T = np.asarray(body_T, np.float32)
    T_ref = body_T[0].astype(np.float32)
    inv = np.linalg.inv(T_ref).astype(np.float32)
    cap_p, cap_R = _apply_inv(inv, cap_T[:, :3, 3], cap_T[:, :3, :3])
    body_p, body_R = _apply_inv(inv, body_T[:, :3, 3], body_T[:, :3, :3])
    obj_feat = np.concatenate([cap_p, rotmat_to_6d(cap_R), body_p, rotmat_to_6d(body_R)], axis=1).astype(np.float32)
    return obj_feat, T_ref


def build_features(npz):
    """npz(prepare_dataset 출력) → (obj_feat(N,18), wrist_feat(N,18), T_ref(4,4))."""
    cap_T, body_T = npz["cap_T"], npz["body_T"]
    N = len(cap_T)
    T_ref = body_T[0].astype(np.float32)
    inv = np.linalg.inv(T_ref).astype(np.float32)

    cap_p, cap_R = _apply_inv(inv, cap_T[:, :3, 3], cap_T[:, :3, :3])
    body_p, body_R = _apply_inv(inv, body_T[:, :3, 3], body_T[:, :3, :3])
    rh_R = Rotation.from_rotvec(npz["rh_waa"]).as_matrix().astype(np.float32)
    lh_R = Rotation.from_rotvec(npz["lh_waa"]).as_matrix().astype(np.float32)
    rh_p, rh_R = _apply_inv(inv, npz["rh_wpos"], rh_R)
    lh_p, lh_R = _apply_inv(inv, npz["lh_wpos"], lh_R)

    obj_feat = np.concatenate([cap_p, rotmat_to_6d(cap_R), body_p, rotmat_to_6d(body_R)], axis=1).astype(np.float32)
    wrist_feat = np.concatenate([rh_p, rotmat_to_6d(rh_R), lh_p, rotmat_to_6d(lh_R)], axis=1).astype(np.float32)
    return obj_feat, wrist_feat, T_ref


def wrist_feat_to_world(wrist_feat, T_ref):
    """출력 특징(N,18) + T_ref → gym/world 프레임 (rh_pos, rh_aa, lh_pos, lh_aa)."""
    def part(o):
        p = wrist_feat[:, o:o + 3]
        R = sixd_to_rotmat(wrist_feat[:, o + 3:o + 9])
        # 되돌리기: W_world = T_ref @ W_canon
        W = np.tile(np.eye(4, dtype=np.float32), (len(p), 1, 1))
        W[:, :3, :3] = R
        W[:, :3, 3] = p
        W = np.einsum("ij,tjk->tik", T_ref, W)
        aa = Rotation.from_matrix(W[:, :3, :3]).as_rotvec().astype(np.float32)
        return W[:, :3, 3].astype(np.float32), aa
    rh_p, rh_aa = part(0)
    lh_p, lh_aa = part(9)
    return rh_p, rh_aa, lh_p, lh_aa


class Normalizer:
    """열별 mean/std 정규화."""
    def __init__(self, x):
        self.mean = x.mean(0)
        self.std = x.std(0) + 1e-6

    def norm(self, x):
        return (x - self.mean) / self.std

    def denorm(self, x):
        return x * self.std + self.mean

    def state(self):
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state(cls, s):
        o = cls.__new__(cls)
        o.mean = s["mean"]; o.std = s["std"]
        return o
