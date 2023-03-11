import taichi as ti
from Rigidbody import Rigidbody
from SceneManager import SceneManager
from ParticleSystem import ParticleSystem
from Params import params

ti.init(arch=ti.gpu, device_memory_GB=4)

window = ti.ui.Window("mpm snow 3d", (1024, 1024), vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((0, 0, 0))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

if params.gui:
    result_dir = "./results"
    # if os.path.exists(result_dir):
    #     os.rmdir(result_dir)
    video_manager = ti.tools.VideoManager(output_dir=result_dir, framerate=30, automatic_build=False)

point = []
for i in range(0, 2):
    for j in range(0, 2):
        for k in range(0, 2):
            point.append([-1+i*2,-1+j*2,-1+k*2])

print(point)

meshIndex = [
    [0,1,2],[1,2,3],
    [4,5,6],[5,6,7],
    [0,1,4],[1,4,5],
    [2,3,6],[3,6,7],
    [0,2,4],[2,4,6],
    [1,3,5],[3,5,7]
]

meshNormal = [
    [-1,0,0],[-1,0,0],
    [1,0,0],[1,0,0],
    [0,-1,0],[0,-1,0],
    [0,1,0],[0,1,0],
    [0,0,-1],[0,0,-1],
    [0,0,1],[0,0,1]
]

camera.position(0.0, 16.0, 0.0)
camera.lookat(0.0, 0.0, -4.0)
camera.up(0.0, 0.0, 1.0)
rigidbody = Rigidbody(4, point, meshIndex, meshNormal)

scene.set_camera(camera)

plane = ti.Vector.field(3, dtype=float, shape=2*3)
planeZ = -5
plane[0] = ti.Vector([5,  5, planeZ])
plane[1] = ti.Vector([-5, 5, planeZ])
plane[2] = ti.Vector([5, -5, planeZ])
plane[3] = ti.Vector([-5,-5, planeZ])
plane[4] = ti.Vector([-5, 5, planeZ])
plane[5] = ti.Vector([5, -5, planeZ])

for frame in range(240):
    print("frame{} start".format(frame))
    for i in range(params.subFrameNum):
        # rigidbody.step(frame * params.subFrameNum+i)
        rigidbody.step()
    frame += 1

    if params.gui:
        # Draw a smaller ball to avoid visual penetration
        scene.mesh(rigidbody.MeshPoint, color=(1, 1, 1), two_sided=True)
        scene.mesh(plane, color=(0,1,1), two_sided=True)
        scene.point_light(pos=(4.0, 0.0, 5.0), color=(1, 1, 1))
        canvas.scene(scene)

        pixels_img = window.get_image_buffer_as_numpy()
        video_manager.write_frame(pixels_img)

        window.show()

if params.gui:
    video_manager.make_video(mp4=True)