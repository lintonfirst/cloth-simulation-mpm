import taichi as ti

@ti.data_oriented
class RigidManager:
    @ti.kernel
    def initPos(self):
        for x in self.rigids:
            self.rigids[x]= [8,8,0.5]
            
    def __init__(self):
        self.numRigids=1
        self.rigids=ti.Vector.field(3,dtype=float,shape=self.numRigids)
        self.initPos()
        
    
            
    @ti.kernel
    def step(self):
        for x in self.rigids:
            self.rigids[x]+=ti.Vector([0.01,0.0,0.0])