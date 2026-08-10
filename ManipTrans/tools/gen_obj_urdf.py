"""CoACD 완료된 각 객체에 대해 obj URDF 생성 (README step 3).
coacd_object_preview/align_ds/<obj>/<name>.<ext> (분해 메시) 를 참조하는 <name>.urdf 를 같은 폴더에 생성.
멱등: 이미 있으면 덮어씀. CoACD가 더 완료되면 다시 실행하면 됨."""
import os
import glob

COACD_ROOT = "data/OakInk-v2/coacd_object_preview/align_ds"

TEMPLATE = '''<?xml version="1.0"?>
<robot name="{name}">
  <material name="obj_color">
      <color rgba="1.0 0.423529411765 0.0392156862745 1.0"/>
  </material>
  <link name="base">
    <visual>
      <origin xyz="0.0 0.0 0.0"/>
      <geometry>
        <mesh filename="{mesh}" scale="1 1 1"/>
      </geometry>
      <material name="obj_color"/>
    </visual>
    <collision>
      <origin xyz="0.0 0.0 0.0"/>
      <geometry>
        <mesh filename="{mesh}" scale="1 1 1"/>
      </geometry>
    </collision>
  </link>
</robot>
'''


def main():
    made = 0
    skipped = 0
    for d in sorted(glob.glob(os.path.join(COACD_ROOT, "*"))):
        if not os.path.isdir(d):
            continue
        # 분해 메시 파일: .done/.failed/.urdf 제외한 mesh(.ply/.obj)
        meshes = [f for f in os.listdir(d)
                  if f.lower().endswith((".ply", ".obj")) and not f.endswith((".done", ".failed"))]
        if not meshes:
            skipped += 1
            continue
        mesh = meshes[0]
        name = os.path.splitext(mesh)[0]
        urdf_path = os.path.join(d, name + ".urdf")
        with open(urdf_path, "w") as f:
            f.write(TEMPLATE.format(name=name, mesh=mesh))
        made += 1
    print(f"obj URDF 생성: {made}개, mesh 없어 skip: {skipped}개")


if __name__ == "__main__":
    main()
