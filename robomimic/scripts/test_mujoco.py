import mujoco

file_path = '/home/zhangel9/.local/lib/python3.10/site-packages/robosuite-1.4.1-py3.10.egg/robosuite/models/assets/robots/panda/obj_meshes/link0_vis/link0_vis_10.obj'
model = mujoco.MjModel.from_xml_string(open(file_path).read())
print("Model loaded successfully")
