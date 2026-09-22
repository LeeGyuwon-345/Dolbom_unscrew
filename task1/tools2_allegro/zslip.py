"""팁의 캡밴드 상대 z 운동: 진폭·왕복(방향전환) 정량. 캡 상승분 보정."""
from isaacgym import gymapi
import csv, sys, math
import numpy as np
URDF_DIR="/home/leegyuwon/Documents/task1/assets/rb5_allegro"; URDF="rb5_allegro_vmount_siltip2.urdf"
PITCH=3.18309886e-03
gym=gymapi.acquire_gym(); sp=gymapi.SimParams(); sp.up_axis=gymapi.UP_AXIS_Z; sp.gravity=gymapi.Vec3(0,0,0)
sim=gym.create_sim(0,0,gymapi.SIM_PHYSX,sp)
o=gymapi.AssetOptions(); o.fix_base_link=True; o.disable_gravity=True; o.collapse_fixed_joints=False
asset=gym.load_asset(sim,URDF_DIR,URDF,o)
env=gym.create_env(sim,gymapi.Vec3(-1,-1,0),gymapi.Vec3(1,1,1),1)
act=gym.create_actor(env,asset,gymapi.Transform(),"r",0,0)
names=gym.get_actor_rigid_body_names(env,act)
TIPS=[(i,n) for i,n in enumerate(names) if n.endswith("_tip")]
LBL={"link_3_0_tip":"검지","link_15_0_tip":"엄지","link_7_0_tip":"중지","link_11_0_tip":"약지",
     "link_3.0_tip":"검지","link_15.0_tip":"엄지","link_7.0_tip":"중지","link_11.0_tip":"약지"}
def episodes(path):
    rows=list(csv.DictReader(open(path))); cur=[]; prev=-1
    for r in rows:
        s=int(float(r["step"]))
        if s<prev and cur: yield cur; cur=[]
        prev=s; cur.append(r)
    if cur: yield cur
path=sys.argv[1]; nmax=int(sys.argv[2]) if len(sys.argv)>2 else 3
done=0
for ep in episodes(path):
    if len(ep)<300: continue
    if done>=nmax: break
    done+=1
    Z=[[] for _ in TIPS]; caps=[]
    for r in ep:
        q=[float(r[f"arm_q_j{j+1}"]) for j in range(6)]+[float(r[f"hand_q_{j}"]) for j in range(16)]
        st=gym.get_actor_dof_states(env,act,gymapi.STATE_ALL); st["pos"]=np.array(q,dtype=np.float32); st["vel"]=0
        gym.set_actor_dof_states(env,act,st,gymapi.STATE_ALL); gym.simulate(sim); gym.fetch_results(sim,True)
        bs=gym.get_actor_rigid_body_states(env,act,gymapi.STATE_POS)
        for t,(i,_) in enumerate(TIPS): Z[t].append(float(bs["pose"]["p"][i][2]))
        caps.append(float(r["unscrew_deg"]))
    lift=np.array(caps)*math.pi/180*PITCH  # 캡 상승(m)
    print(f"\nep(캡 {caps[-1]-caps[0]:+.0f}°): 캡상승 {lift[-1]*1000:.1f}mm")
    for t,(i,n) in enumerate(TIPS):
        rel=(np.array(Z[t])-lift)*1000  # 캡밴드 상대 z (mm)
        d=np.diff(rel)
        # 0.3mm 이상 움직임의 방향전환 수 = 왕복 지표
        sig=d[np.abs(d)>0.05]
        rev=int(np.sum(np.sign(sig[1:])*np.sign(sig[:-1])<0)) if len(sig)>2 else 0
        span=rel.max()-rel.min()
        print(f"  {LBL.get(n,n)}: 상대z 범위 {span:5.1f}mm  방향전환 {rev:3d}회  순변위 {rel[-1]-rel[0]:+6.1f}mm")
