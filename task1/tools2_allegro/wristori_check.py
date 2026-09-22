"""손목(base_link) 자세 비교: 두 자세의 손목 회전 차이(각도)와 팜 법선 방향."""
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
names=gym.get_actor_rigid_body_names(env,act); WI=names.index("base_link")
def wrist_R(path):
    q=np.array(json.load(open(path))['dof_pos'],dtype=np.float32)
    st=gym.get_actor_dof_states(env,act,gymapi.STATE_ALL); st["pos"]=q; st["vel"]=0
    gym.set_actor_dof_states(env,act,st,gymapi.STATE_ALL); gym.simulate(sim); gym.fetch_results(sim,True)
    r=gym.get_actor_rigid_body_states(env,act,gymapi.STATE_POS)["pose"]["r"][WI]
    x,y,z,w=float(r[0]),float(r[1]),float(r[2]),float(r[3])
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
A=wrist_R('/home/leegyuwon/Documents/task1/tools2_allegro/poses/rhinit2_rot115.json')
B=wrist_R(sys.argv[1])
Rd=A.T@B
ang=math.degrees(math.acos(max(-1,min(1,(np.trace(Rd)-1)/2))))
axis=np.array([Rd[2,1]-Rd[1,2],Rd[0,2]-Rd[2,0],Rd[1,0]-Rd[0,1]])
axis=axis/ (np.linalg.norm(axis)+1e-9)
print(f"rot115 대비 손목 회전 차이: {ang:.1f}도  (축 {axis.round(2)})")
for nm,R in (("rot115",A),("az135",B)):
    print(f"  {nm} 손목 z축(월드): {R[:,2].round(2)}  x축: {R[:,0].round(2)}")
