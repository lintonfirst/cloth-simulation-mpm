import taichi as ti
from SceneManager import SceneManager
from Params import params, vec3r
from Rigidbody import Rigidbody
import numpy as np

@ti.data_oriented
class ParticleSystem:
    def __init__(self, scene: SceneManager = None):
        self.particleNum = 0
        self.fluidParticleNum = 0

        self.lambda0 = params.youngsModulus * params.poissionRate / (1 + params.poissionRate) / (
                1 - 2 * params.poissionRate)
        self.mu0 = params.youngsModulus / (2 * (1 + params.poissionRate))

        self.position = ti.Vector.field(3, dtype=float)
        self.mass = ti.field(dtype=float)
        self.volume = ti.field(dtype=float)
        self.material = ti.field(dtype=ti.int8)
        self.color = ti.Vector.field(3, dtype=float)
        self.velocity = ti.Vector.field(3, dtype=float)

        self.FE = ti.Matrix.field(3, 3, dtype=float)
        self.FP = ti.Matrix.field(3, 3, dtype=float)

        if SceneManager is not None:
            self.init_field(scene)

    def init_field(self, scene: SceneManager):
        self.particleNum = scene.particleNum
        self.fluidParticleNum = scene.fluidParticleNum

        ti.root.dense(ti.i, self.particleNum).place(self.position)
        ti.root.dense(ti.i, self.particleNum).place(self.mass)
        ti.root.dense(ti.i, self.particleNum).place(self.volume)
        ti.root.dense(ti.i, self.particleNum).place(self.material)
        ti.root.dense(ti.i, self.particleNum).place(self.color)
        ti.root.dense(ti.i, self.particleNum).place(self.velocity)

        ti.root.dense(ti.i, self.fluidParticleNum).place(self.FE)
        ti.root.dense(ti.i, self.fluidParticleNum).place(self.FP)

        self.position.from_numpy(scene.position)
        self.velocity.from_numpy(scene.velocity)
        self.init_field_gpu()

    @ti.kernel
    def init_field_gpu(self):
        identity = ti.Matrix.identity(float, 3)
        color = vec3r([1,1,1])

        for i in self.color:
            self.FE[i] = identity
            self.FP[i] = identity
            self.color[i] = color