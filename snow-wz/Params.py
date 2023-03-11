import taichi as ti

ti.init(arch=ti.gpu, device_memory_GB=4)
vec3r = ti.types.vector(3, ti.f32)


@ti.data_oriented
class Params:
    def __init__(self):
        self.gui = True
        self.subFrameNum = 60
        self.dt = ti.field(dtype=ti.f32, shape=())
        self.frameDt = 1/60
        self.dt[None] = self.frameDt/self.subFrameNum
        self.maxDt = self.frameDt/self.subFrameNum
        self.minDt = self.frameDt/self.subFrameNum*0.2

        self.discretizeRange = 4
        self.discretizeStart = -1
        self.cellSize = 0.1
        self.gridLength = 20
        self.particleRadius = 0.05

        self.youngsModulus = 1.4e5
        self.poissionRate = 0.2
        self.hardenCoef = 10
        self.frictionCoef = 0.2
        self.thetaC = 2.5e-2 # 2.5e-2
        self.thetaS = 7.5e-3 # 7.5e-3
        self.density0 = 4e2

        self.alpha = 0.95

        self.windowWidth = 720
        self.ifRandom = False
        self.ifUseRigidbody = False
        self.ifTwoWayCouple = False

        self.rigidBodyInitX = [0, 0, 0]
        self.rigidBodyInitV = [0, 0, 0]
        self.rigidBodyOmega = 0
        self.rigidBodyScale = 1

params = Params()
