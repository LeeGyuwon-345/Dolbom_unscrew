"""팔 구성 비교: elbow(link3)·손목(base_link) 월드 위치와 엘보우업 여부."""
import json, math, sys
import numpy as np
from isaacgym import gymapi
RD="/home/leegyuwon/Documents/task1/assets/rb5_allegro"; RU="rb5_allegro_vmount_siltip2.urdf"
gym=gymapi.acquire_gym(); sp=gymapi.SimParams(); sp.up_axis=gymapi.UP_AXIS_Z; sp.gravity=gymapi.Vec3(0,0,0)
sim=gym.create_sim(0,0,gymapi.SIM_PHYSX,sp)
o=gymapi.AssetOptions(); o.fix_base_link=True; o.disable_gravity=True; o.collapse_fixed_joints=False
asset=gym.load_asset(sim,RD,RU,o)
env=gym.create_env(sim,gymapi.Vec3(-1,-1,0),gymapi.Vec3(1,1,1),1)
act=gym.create_actor(env,asset,gymapi.Transform(),"r",0,0)
names=gym.get_actor_rigid_body_names(env,act)
IDX={n:i for i,n in enumerate(names)}
for path in sys.argv[1:]:
    q=np.array(json.load(open(path))['dof_pos'],dtype=np.float32)
    st=gym.get_actor_dof_states(env,act,gymapi.STATE_ALL); st["pos"]=q; st["vel"]=0
    gym.set_actor_dof_states(env,act,st,gymapi.STATE_ALL); gym.simulate(sim); gym.fetch_results(sim,True)
    bs=gym.get_actor_rigid_body_states(env,act,gymapi.STATE_POS)
    out={}
    for n in ("link3","link4","tcp","base_link"):
        p=bs["pose"]["p"][IDX[n]]; out[n]=(round(float(p[0]),3),round(float(p[1]),3),round(float(p[2]),3))
    eb="엘보우업" if out["link3"][2] > out["tcp"][2]+0.03 else "엘보우 낮음!"
    print(f"{path.split('/')[-1]}: J1={math.degrees(q[0]):.0f}도  elbow(link3)={out['link3']}  tcp={out['tcp']}  손목={out['base_link']}  [{eb}]")
