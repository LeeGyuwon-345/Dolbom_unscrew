"""slip2 재현: 앵커 기반 순간편차의 분포/축분해/dead 스윕 (CSV+FK)."""
from isaacgym import gymapi
import csv, sys, math
import numpy as np
URDF_DIR="/home/leegyuwon/Documents/task1/assets/rb5_allegro"; URDF="rb5_allegro_vmount_siltip2.urdf"
CAPC=np.array([0.4,0.0]); CAPR=0.044; PAD=0.0115; PITCH=3.18309886e-03
gym=gymapi.acquire_gym(); sp=gymapi.SimParams(); sp.up_axis=gymapi.UP_AXIS_Z; sp.gravity=gymapi.Vec3(0,0,0)
sim=gym.create_sim(0,0,gymapi.SIM_PHYSX,sp)
o=gymapi.AssetOptions(); o.fix_base_link=True; o.disable_gravity=True; o.collapse_fixed_joints=False
asset=gym.load_asset(sim,URDF_DIR,URDF,o)
env=gym.create_env(sim,gymapi.Vec3(-1,-1,0),gymapi.Vec3(1,1,1),1)
act=gym.create_actor(env,asset,gymapi.Transform(),"r",0,0)
names=gym.get_actor_rigid_body_names(env,act)
TIPS=[(i,n) for i,n in enumerate(names) if n.endswith("_tip")]
LBL={"link_15.0_tip":"엄지","link_3.0_tip":"검지","link_7.0_tip":"중지","link_11.0_tip":"약지"}
rows=list(csv.DictReader(open(sys.argv[1])))
eps=[];cur=[];prev=-1
for r in rows:
    s=int(float(r["step"]))
    if s<prev and cur: eps.append(cur); cur=[]
    prev=s; cur.append(r)
if cur: eps.append(cur)
S_all=[]; Sphi=[]; Sz=[]
anch={}  # tip -> (phi,z)
for ep in eps:
    if len(ep)<300: continue
    anch={}
    for r in ep:
        q=[float(r[f"arm_q_j{j+1}"]) for j in range(6)]+[float(r[f"hand_q_{j}"]) for j in range(16)]
        st=gym.get_actor_dof_states(env,act,gymapi.STATE_ALL); st["pos"]=np.array(q,dtype=np.float32); st["vel"]=0
        gym.set_actor_dof_states(env,act,st,gymapi.STATE_ALL); gym.simulate(sim); gym.fetch_results(sim,True)
        bs=gym.get_actor_rigid_body_states(env,act,gymapi.STATE_POS)
        ang=math.radians(float(r["unscrew_deg"])); capz=PITCH*max(ang,0.0)
        for bi,nm in TIPS:
            if nm not in LBL: continue
            p=bs["pose"]["p"][bi]; xy=np.array([p[0],p[1]])-CAPC
            rad=np.linalg.norm(xy)
            near = (rad - PAD - CAPR) < 0.001   # fingerep 동일 기준 (반경만)
            if not near:
                anch.pop(nm,None); continue
            phi=math.atan2(xy[1],xy[0])-ang
            z=p[2]-capz                          # 캡 상승 보정 상대 z (zslip 동일)
            if nm not in anch: anch[nm]=(phi,z); continue
            dphi=phi-anch[nm][0]; dphi=math.atan2(math.sin(dphi),math.cos(dphi))
            dz=z-anch[nm][1]
            s2=math.hypot(dphi*CAPR,dz)
            S_all.append(s2*1000); Sphi.append(abs(dphi*CAPR)*1000); Sz.append(abs(dz)*1000)
S=np.array(S_all); P=np.array(Sphi); Z=np.array(Sz)
print(f"접촉스텝 표본 {len(S)}")
print(f"편차 s   : 평균 {S.mean():.1f}mm  중앙 {np.median(S):.1f}  p75 {np.percentile(S,75):.1f}  p90 {np.percentile(S,90):.1f}  p99 {np.percentile(S,99):.1f}  최대 {S.max():.1f}")
print(f"축 분해  : |φ·R| 평균 {P.mean():.1f}mm (p90 {np.percentile(P,90):.1f})   |z| 평균 {Z.mean():.1f}mm (p90 {np.percentile(Z,90):.1f})")
for d in (5,8,10,15):
    print(f"dead {d:2d}mm 초과 스텝 비율: 전체 {100*(S>d).mean():.1f}%   z단독 {100*(Z>d).mean():.1f}%   φ단독 {100*(P>d).mean():.1f}%")
