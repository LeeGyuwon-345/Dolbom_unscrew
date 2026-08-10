"""텀블러 배치 최적화: Al9 손목 궤적을 RB5 IK 로 추종 가능한 위치 탐색.

입력: wrist_traj.csv (캡홈 상대 손목 자세, Al9 재생 기록)
방법: 텀블러 후보 (x,y) 그리드마다, 궤적 웨이포인트를 순차 IK(워름스타트)로
추종하며 채점 -- IK 잔차 / 관절한계 여유 / 중간링크 바닥높이 / 구성 연속성.
"""
import csv
import json
import math
import sys

import torch
import pytorch_kinematics as pk
import xml.dom.minidom as minidom

TRAJ = "/tmp/claude-1000/-home-leegyuwon-Documents/c6d021c0-e2b6-4d39-9d88-21f873dadf99/scratchpad/wrist_traj.csv"
CAP_Z = 0.225
SUB = 6            # 웨이포인트 서브샘플 간격
XS = [0.35, 0.45, 0.55, 0.60, 0.65, 0.75]
YS = [-0.20, -0.10, 0.0, 0.10, 0.20]

rows = list(csv.DictReader(open(TRAJ)))
# 첫 에피소드만 (step 이 다시 0 으로 떨어지기 전까지)
traj = []
prev = -1
for r in rows:
    s = int(r["step"])
    if s < prev:
        break
    prev = s
    traj.append([float(r[k]) for k in ("px", "py", "pz", "qx", "qy", "qz", "qw")])
traj = traj[::SUB]
print(f"웨이포인트 {len(traj)}개 (서브샘플 {SUB})")

urdf = "/home/leegyuwon/Documents/task1/assets/rb5_850e/rb5_850e.urdf"
chain = pk.build_serial_chain_from_urdf(open(urdf, "rb").read(), "tcp")
names = chain.get_joint_parameter_names()
d = minidom.parse(urdf)
lim = {}
for j in d.getElementsByTagName("joint"):
    if j.getAttribute("type") == "revolute":
        l = j.getElementsByTagName("limit")[0]
        lim[j.getAttribute("name")] = (float(l.getAttribute("lower")), float(l.getAttribute("upper")))
lo = torch.tensor([lim[n][0] for n in names])
hi = torch.tensor([lim[n][1] for n in names])

R_m = torch.tensor([[1., 0, 0], [0, 0., -1.], [0, 1., 0.]])   # Rx(+90)
t_m = torch.tensor([0.0, -0.125, 0.0])

def quat_to_R(q):
    x, y, z, w = q
    return torch.tensor([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

def solve_from(q0, p_t, R_t, iters=120, lr=0.06):
    q = q0.clone().requires_grad_(True)
    opt = torch.optim.Adam([q], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        ms = chain.forward_kinematics(q.unsqueeze(0), end_only=False)
        m = ms["tcp"].get_matrix()[0]
        # 관절 범위 ±360도(공식 스펙) -> wrist 여행 720도라 완전 자세 추종 재시도
        loss = ((m[:3, 3] - p_t) ** 2).sum() * 4.0 + ((m[:3, :3] - R_t) ** 2).sum() * 0.5 \
               + ((q - q.clamp(lo, hi)) ** 2).sum() * 10.0 + (q ** 2).sum() * 1e-3 \
               + ((q - q0) ** 2).sum() * 5e-3
        for ln in ("link3", "link4", "link5"):
            lz = ms[ln].get_matrix()[0, 2, 3]
            loss = loss + torch.relu(0.15 - lz) ** 2 * 50.0
        loss.backward()
        opt.step()
    with torch.no_grad():
        qc = q.clamp(lo, hi)
        ms = chain.forward_kinematics(qc.unsqueeze(0), end_only=False)
        m = ms["tcp"].get_matrix()[0]
        e = float((m[:3, 3] - p_t).norm())
        cos = ((torch.trace(m[:3, :3].T @ R_t) - 1) / 2).clamp(-1, 1)
        eo = float(torch.arccos(cos))
        minz = min(float(ms[l].get_matrix()[0, 2, 3]) for l in ("link3", "link4", "link5"))
        margin = float(torch.minimum(qc - lo, hi - qc).min())
    return qc.detach(), e, eo, minz, margin

seed = json.load(open("/home/leegyuwon/Documents/task1/assets/rb5_allegro/arm_ik_grasp.json"))
q_seed = torch.tensor(seed["q"])

results = []
for X in XS:
    for Y in YS:
        cap = torch.tensor([X, Y, CAP_Z])
        q = q_seed.clone()
        # 첫 웨이포인트는 넉넉히 수렴시킨다
        worst = dict(e=0.0, eo=0.0, minz=1e9, margin=1e9, jump=0.0)
        ok = True
        for i, w in enumerate(traj):
            p_bl = cap + torch.tensor(w[:3])
            R_bl = quat_to_R(w[3:])
            R_t = R_bl @ R_m.T
            p_t = p_bl - R_t @ t_m
            q_new, e, eo, minz, margin = solve_from(q, p_t, R_t, iters=300 if i == 0 else 80)
            jump = float((q_new - q).abs().max()) if i > 0 else 0.0
            worst["e"] = max(worst["e"], e)
            worst["eo"] = max(worst["eo"], eo)
            worst["minz"] = min(worst["minz"], minz)
            worst["margin"] = min(worst["margin"], margin)
            worst["jump"] = max(worst["jump"], jump)
            q = q_new
            if worst["e"] > 0.05:      # 5cm 이상 실패 -> 조기 탈락
                ok = False
                break
        feas = ok and worst["e"] < 0.01 and worst["minz"] > 0.10 and worst["margin"] > 0.15
        results.append((X, Y, feas, worst))
        print(f"({X:.2f},{Y:+.2f})  최대IK오차 {worst['e']*1000:6.1f}mm  자세 {math.degrees(worst['eo']):5.1f}도  "
              f"최저링크z {worst['minz']*1000:4.0f}mm  한계여유 {math.degrees(worst['margin']):5.1f}도  "
              f"점프 {math.degrees(worst['jump']):5.1f}도  {'가능' if feas else '탈락'}", flush=True)

good = [r for r in results if r[2]]
if good:
    best = max(good, key=lambda r: (math.degrees(r[3]["margin"]) - r[3]["e"] * 1e4))
    print(f"\n최적 텀블러 위치: ({best[0]:.2f}, {best[1]:+.2f})  "
          f"한계여유 {math.degrees(best[3]['margin']):.1f}도, IK오차 {best[3]['e']*1000:.1f}mm")
else:
    print("\n가능 위치 없음 -- 그리드/조건 재검토 필요")
