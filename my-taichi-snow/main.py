import taichi as ti
from ParticleManager import ParticleManager
from RigidManager import RigidManager

ti.init(arch=ti.gpu)  # Alternatively, ti.init(arch=ti.cpu)

window = ti.ui.Window("Taichi Cloth Simulation on GGUI", (512, 512),
                      vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((0, 0, 0))
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camaraPos={
    'x':8,
    'y':8,
    'z':20,
}

rigidManager=RigidManager()
particleManager=ParticleManager(rigidManager)

print("begin frame")
while window.running:
    if window.get_event(ti.ui.PRESS):
        if window.event.key == 'w': camaraPos['y']+=0.5
        elif window.event.key == 's': camaraPos['y']-=0.5
        elif window.event.key == 'a': camaraPos['x']-=0.5
        elif window.event.key == 'd': camaraPos['x']+=0.5
        elif window.event.key == 'q': camaraPos['z']+=0.5
        elif window.event.key == 'e': camaraPos['z']-=0.5
    camera.position(camaraPos['x'],camaraPos['y'], camaraPos['z'])
    camera.lookat(camaraPos['x'], camaraPos['y'], 0)
    scene.set_camera(camera)

    scene.point_light(pos=(8, 9, 10), color=(1, 1, 1))
    # Draw a smaller ball to avoid visual penetration
    
    rigidManager.step()
    particleManager.step()
    scene.particles(rigidManager.rigids, radius=0.3, color=(0.7, 0, 0))
    scene.particles(particleManager.particles, radius=0.05, color=(0.9, 0.9, 0.9))
    canvas.scene(scene)
    window.show()