import taichi as ti
from Params import params, vec3r


@ti.data_oriented
class Grid:
    def __init__(self, edgeGridNum, minBoundary, edgeLength, particleSystem):
        self.edgeGridNum = edgeGridNum
        self.gridR = edgeLength / edgeGridNum
        self.invGridR = edgeGridNum / edgeLength
        self.particleSystem = particleSystem

        self.minBoundary = vec3r(minBoundary)
        self.maxBoundary = vec3r(minBoundary)+vec3r([edgeLength, edgeLength, edgeLength])

        self.gridVelocity = ti.Vector.field(3, dtype=ti.f32, shape=(edgeGridNum+1, edgeGridNum+1, edgeGridNum+1))
        self.gridLastVelocity = ti.Vector.field(3, dtype=ti.f32, shape=(edgeGridNum+1, edgeGridNum+1, edgeGridNum+1))
        self.gridMass = ti.field(dtype=ti.f32, shape=(edgeGridNum+1, edgeGridNum+1, edgeGridNum+1))
        self.gridForce = ti.Vector.field(3, dtype=ti.f32, shape=(edgeGridNum+1, edgeGridNum+1, edgeGridNum+1))

    @ti.func
    def N(self, x) -> ti.f32:
        x = ti.abs(x * self.invGridR)
        returnValue = 0.0
        if x < 1:
            returnValue = x**2*(x/2-1)+2/3
        elif x < 2:
            returnValue = x*(x*(-x/6+1)-2)+4/3
        if returnValue < 1e-4:
            returnValue = 0.0
        return returnValue

    @ti.func
    def dN(self, x) -> ti.f32:
        x = x * self.invGridR
        t = ti.abs(x)
        returnValue = 0.0
        if t < 1:
            returnValue = 1.5*x*t - 2*x
        elif t < 2:
            returnValue = -x * t/2 + 2*x-2*x/t
        return returnValue


    @ti.func
    def weight(self, positionOffset) -> ti.f32:
        returnValue = self.N(positionOffset.x)*self.N(positionOffset.y)*self.N(positionOffset.z)
        if returnValue < 1e-4:
            returnValue = 0.0
        return returnValue

    @ti.func
    def dWeight(self, positionOffset) -> vec3r:
        retVec = ti.Vector([0.0, 0.0, 0.0])
        retVec.x = self.dN(positionOffset.x) * self.N(positionOffset.y) * self.N(positionOffset.z)
        retVec.y = self.N(positionOffset.x) * self.dN(positionOffset.y) * self.N(positionOffset.z)
        retVec.z = self.N(positionOffset.x) * self.N(positionOffset.y) * self.dN(positionOffset.z)
        retVec.x = self.dN(positionOffset.x) * self.N(positionOffset.y) * self.N(positionOffset.z)
        retVec.y = self.N(positionOffset.x) * self.dN(positionOffset.y) * self.N(positionOffset.z)
        retVec.z = self.N(positionOffset.x) * self.N(positionOffset.y) * self.dN(positionOffset.z)
        return retVec * self.invGridR


