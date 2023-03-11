import taichi as ti
import RigidManager as RigidManager
from Config import Config
import taichi as ti
import random

@ti.data_oriented
class ParticleManager:
    @ti.kernel
    def initPos(self):
        for x in range(self.numParticles):
            self.particles[x]=[ti.random(float) * 12.8 + 1.6,ti.random(float)  * 12.8+ 1.6,ti.random(float)  * 0.5]
        
    def __init__(self,rigidmanager):
        self.rigidManager : RigidManager=rigidmanager
        self.numGrids=Config['numGrids']
        self.numParticles=Config['numParticles']
        self.particles=ti.Vector.field(3,dtype=float,shape=self.numParticles)
        self.initPos()
    
    @ti.kernel
    def step(self):
        pass