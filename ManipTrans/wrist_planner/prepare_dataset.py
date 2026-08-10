"""양손(bimanual) 손목 플래너 학습용 '정렬된' 데이터셋 생성.

문제: RH/LH 리타겟 pkl은 각자 다른 원본 프레임 구간(RH range, LH range)을
      각각 T프레임으로 리샘플한 것이라, 같은 인덱스가 서로 다른 실제 시각이다.
해결: 두 구간의 겹치는 공통 구간에서 N프레임을 뽑아, 양손 손목/DOF와
      양 객체(뚜껑/몸통) SE(3) 궤적을 모두 공통 타임라인으로 보간 정렬한다.

출력: wrist_planner/datasets/<idx>.npz  (모두 gym/world 프레임, 공통 타임라인 정렬)
  common_t       (N,)        공통 원본 프레임 인덱스
  cap_T, body_T  (N,4,4)     뚜껑/몸통 SE(3) (gym 프레임)
  rh_wpos,rh_waa (N,3)(N,3)  오른손목 위치 / axis-angle
  lh_wpos,lh_waa (N,3)(N,3)  왼손목 위치 / axis-angle
  rh_dof,lh_dof  (N,20)      손가락 관절각
  cap_urdf, body_urdf, idx, rh_range, lh_range  메타

기존 maniptrans 코드는 읽기만 하며 수정하지 않는다.
실행:
  cd /home/leegyuwon/Documents/ManipTrans
  PATH=<env>/bin:$PATH LD_LIBRARY_PATH=<env>/lib:$LD_LIBRARY_PATH \
  python wrist_planner/prepare_dataset.py --data_idx 97fc3@1
"""
import isaacgym  # noqa: F401  (torch보다 먼저)
import os
import json
import glob
import pickle
import argparse
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

from main.dataset.factory import ManipDataFactory
from main.dataset.mano2dexhand import pack_data
from main.dataset.transform import aa_to_rotmat
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory

HERE = os.path.dirname(os.path.abspath(__file__))


def mujoco2gym_transf():
    """play_retarget.py와 동일한 데이터->gym 변환."""
    M = np.eye(4)
    M[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(np.array([np.pi / 2, 0, 0]))
    M[:3, 3] = np.array([0, 0, 0.4 + 0.015])
    return M.astype(np.float32)


def program_ranges(idx):
    """program_info에서 ((rh_range),(lh_range)) 반환."""
    h, st = idx.split("@")
    prog_dir = "data/OakInk-v2/program/program_info"
    h2p = {}
    for f in glob.glob(prog_dir + "/*.json"):
        b = os.path.basename(f)
        try:
            hh = b.split("++seq__")[1].split("__")[0][:5]
        except IndexError:
            continue
        h2p[hh] = f
    prog = json.load(open(h2p[h]))
    key = list(prog.keys())[int(st)]
    rng = eval(key)  # ((rh0,rh1),(lh0,lh1))
    return rng[0], rng[1]


def load_side(idx, side, M, device="cuda:0"):
    """한쪽 손의 (객체 SE(3), 손목 pos/aa, dof) + 객체 urdf 로드. 모두 gym 프레임."""
    dh = DexHandFactory.create_hand("dg5fs", side)
    dtype = ManipDataFactory.dataset_type(idx)
    dd = ManipDataFactory.create_data(
        manipdata_type=dtype, side=side, device=device,
        mujoco2gym_transf=torch.eye(4, device=device), dexhand=dh, verbose=False,
    )
    demo = pack_data([dd[idx]], dh)
    obj = demo["obj_trajectory"].view(-1, 4, 4).detach().cpu().numpy()          # (T,4,4) data 프레임
    obj_gym = np.einsum("ij,tjk->tik", M, obj).astype(np.float32)                # gym 프레임
    obj_urdf = demo["obj_urdf_path"][0]
    T = obj_gym.shape[0]

    tag = "rh" if side == "right" else "lh"
    pk = glob.glob(f"data/retargeting/OakInk-v2/mano2dg5fs_{tag}/*{idx.split('@')[0]}*@{idx.split('@')[1]}.pkl")
    assert pk, f"리타겟 pkl 없음: {tag} {idx}"
    ret = pickle.load(open(pk[0], "rb"))
    wpos = ret["opt_wrist_pos"].astype(np.float32)     # (Tr,3) gym
    waa = ret["opt_wrist_rot"].astype(np.float32)      # (Tr,3) axis-angle
    dof = ret["opt_dof_pos"].astype(np.float32)        # (Tr,20)
    Tr = wpos.shape[0]
    assert Tr == T, f"[{side}] demo T={T} != retarget Tr={Tr} (정렬 가정 불일치)"
    return dict(obj=obj_gym, obj_urdf=obj_urdf, wpos=wpos, waa=waa, dof=dof, T=T)


def interp_pos(src_t, src_v, q_t):
    """(T,D) 선형보간 -> (N,D)."""
    return np.stack([np.interp(q_t, src_t, src_v[:, d]) for d in range(src_v.shape[1])], axis=1).astype(np.float32)


def interp_rot(src_t, src_R, q_t):
    """(T,3,3) 회전 Slerp -> (N,3,3)."""
    slerp = Slerp(src_t, Rotation.from_matrix(src_R))
    return slerp(q_t).as_matrix().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_idx", default="97fc3@1")
    ap.add_argument("--n_frames", type=int, default=300, help="공통 타임라인 샘플 수")
    args = ap.parse_args()
    idx = args.data_idx

    M = mujoco2gym_transf()
    rh_rng, lh_rng = program_ranges(idx)
    print(f"[range] RH={rh_rng} LH={lh_rng}")

    R = load_side(idx, "right", M)   # 뚜껑(cap) 조작
    L = load_side(idx, "left", M)    # 몸통(body) 조작
    print(f"[frames] RH T={R['T']} LH T={L['T']}")
    print(f"[obj] cap={os.path.basename(os.path.dirname(R['obj_urdf']))}  body={os.path.basename(os.path.dirname(L['obj_urdf']))}")

    # 각 side의 실제 프레임 타임스탬프
    t_r = np.linspace(rh_rng[0], rh_rng[1], R["T"])
    t_l = np.linspace(lh_rng[0], lh_rng[1], L["T"])
    # 공통(겹치는) 구간
    c0, c1 = max(rh_rng[0], lh_rng[0]), min(rh_rng[1], lh_rng[1])
    common_t = np.linspace(c0, c1, args.n_frames)
    print(f"[common] 겹침 구간=[{c0},{c1}] len={c1-c0} -> {args.n_frames} 프레임")

    # 뚜껑(RH side), 몸통(LH side) SE(3) 보간
    cap_pos = interp_pos(t_r, R["obj"][:, :3, 3], common_t)
    cap_R = interp_rot(t_r, R["obj"][:, :3, :3], common_t)
    body_pos = interp_pos(t_l, L["obj"][:, :3, 3], common_t)
    body_R = interp_rot(t_l, L["obj"][:, :3, :3], common_t)

    def to_T(pos, Rm):
        Tmat = np.tile(np.eye(4, dtype=np.float32), (len(pos), 1, 1))
        Tmat[:, :3, :3] = Rm
        Tmat[:, :3, 3] = pos
        return Tmat

    cap_T = to_T(cap_pos, cap_R)
    body_T = to_T(body_pos, body_R)

    # 양손 손목 pos/rot, dof 보간
    rh_wpos = interp_pos(t_r, R["wpos"], common_t)
    rh_R = interp_rot(t_r, aa_to_rotmat(R["waa"]), common_t)
    rh_waa = Rotation.from_matrix(rh_R).as_rotvec().astype(np.float32)
    rh_dof = interp_pos(t_r, R["dof"], common_t)

    lh_wpos = interp_pos(t_l, L["wpos"], common_t)
    lh_R = interp_rot(t_l, aa_to_rotmat(L["waa"]), common_t)
    lh_waa = Rotation.from_matrix(lh_R).as_rotvec().astype(np.float32)
    lh_dof = interp_pos(t_l, L["dof"], common_t)

    out = os.path.join(HERE, "datasets", f"{idx}.npz")
    np.savez(
        out,
        common_t=common_t.astype(np.float32),
        cap_T=cap_T, body_T=body_T,
        rh_wpos=rh_wpos, rh_waa=rh_waa, rh_dof=rh_dof,
        lh_wpos=lh_wpos, lh_waa=lh_waa, lh_dof=lh_dof,
        cap_urdf=R["obj_urdf"], body_urdf=L["obj_urdf"],
        idx=idx, rh_range=np.array(rh_rng), lh_range=np.array(lh_rng),
    )
    # 정합성 로그
    print(f"[저장] {out}")
    print(f"  cap 이동범위={np.linalg.norm(cap_pos.max(0)-cap_pos.min(0))*100:.1f}cm  "
          f"body 이동범위={np.linalg.norm(body_pos.max(0)-body_pos.min(0))*100:.1f}cm")
    # 뚜껑의 몸통대비 상대회전(=풀림 회전) 총량
    rel = np.einsum("tij,tjk->tik", np.transpose(body_R, (0, 2, 1)), cap_R)
    ang = np.degrees(np.linalg.norm(Rotation.from_matrix(rel).as_rotvec(), axis=1))
    print(f"  뚜껑-몸통 상대회전: {ang.min():.0f}~{ang.max():.0f}deg (풀림 진행량 {ang.max()-ang.min():.0f}deg)")


if __name__ == "__main__":
    main()
