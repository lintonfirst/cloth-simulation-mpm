import taichi as ti
from Simulator import Simulator
from SceneManager import SceneManager
from ParticleSystem import ParticleSystem
from Params import params
from TimeBench import timeBench
import os

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

sceneManager = SceneManager()
print("Prepare Scene")
sceneManager.TwoWayCouple(camera)

print("Init Particle System")
particleSystem = ParticleSystem(sceneManager)

print("Init Simulator")
sim = Simulator(particleSystem)

scene.set_camera(camera)

print("Start Simulation")
totalFrame = 0
for frame in range(240):
    print("frame{} start".format(frame))
    subFrame = 0
    frameTime = 0.0
    # while frameTime < params.frameDt:
    #     # print("subframe{} start".format(frame*20+i))
    #     sim.step(totalFrame)
    #     frameTime += params.dt[None]
    #     totalFrame += 1
    # timeBench.report()
    if params.gui:
        # Draw a smaller ball to avoid visual penetration
        scene.particles(sim.particleSystem.position, radius=params.particleRadius, per_vertex_color=sim.particleSystem.color)
        # if params.ifUseRigidbody:
        #     scene.mesh(sim.rigidbody.MeshPoint, color=(1, 1, 1), two_sided=True)
        scene.mesh(sim.collideBodySystem.meshPoint, per_vertex_color=sim.collideBodySystem.meshColor, two_sided=True)
        scene.point_light(pos=(0.0, 0.0, 5.0), color=(1, 1, 1))
        # canvas.scene(scene)

        window.show()
    frame += 1