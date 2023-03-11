import taichi as ti
from Params import params, vec3r

@ti.data_oriented
class CollideBodySystem:
    def __init__(self):
        self.meshNum = 0
        self.meshPoint = ti.Vector.field(3, dtype=float)
        self.meshVelocity = ti.Vector.field(3, dtype=float)
        self.meshNormal = ti.Vector.field(3, dtype=float)
        self.meshColor = ti.Vector.field(3, dtype=float)

    def addGround(self):
        useMoveMesh = False
        if useMoveMesh:
            self.meshNum = 4
        else:
            self.meshNum = 2

        ti.root.dense(ti.i, self.meshNum*3).place(self.meshPoint)
        ti.root.dense(ti.i, self.meshNum).place(self.meshVelocity)
        ti.root.dense(ti.i, self.meshNum).place(self.meshNormal)
        ti.root.dense(ti.i, self.meshNum*3).place(self.meshColor)

        point1 = vec3r([-params.gridLength/2, -params.gridLength/2, 0])
        point2 = vec3r([params.gridLength/2, -params.gridLength/2, 0])
        point3 = vec3r([-params.gridLength/2, params.gridLength/2, 0])
        point4 = vec3r([params.gridLength/2, params.gridLength/2, 0])

        self.meshPoint[0*3+0] = point1
        self.meshPoint[0*3+1] = point2
        self.meshPoint[0*3+2] = point3
        self.meshPoint[1*3+0] = point3
        self.meshPoint[1*3+1] = point2
        self.meshPoint[1*3+2] = point4

        for i in range(0, 2):
            self.meshVelocity[i] = vec3r([0, 0, 0])
            self.meshNormal[i] = vec3r([0, 0, 1])
            for j in range(0, 3):
                self.meshColor[i * 3 + j] = vec3r([0, 1, 1])

        if useMoveMesh:
            point1 = vec3r([-3, -1,  -5.5])
            point2 = vec3r([-3, 1, -5.5])
            point3 = vec3r([-5, -1, 0])
            point4 = vec3r([-5, 1, 0])

            self.meshPoint[2*3+0] = point1
            self.meshPoint[2*3+1] = point2
            self.meshPoint[2*3+2] = point3
            self.meshPoint[3*3+0] = point3
            self.meshPoint[3*3+1] = point2
            self.meshPoint[3*3+2] = point4

            for i in range(2, 4):
                self.meshVelocity[i] = vec3r([2, 0, 0])
                self.meshNormal[i] = vec3r([0, 1, 0]).cross(point3-point1).normalized()
                for j in range(0, 3):
                    self.meshColor[i * 3 + j] = vec3r([1, 1, 0])

    @ti.kernel
    def meshMove(self):
        for index in self.meshVelocity:
            for i in ti.ndrange(3):
                # if self.meshPoint[index * 3 + i].x + self.meshVelocity[index].x * params.dt[None] > 5.0:
                #     self.meshVelocity[index] = ti.Vector([0, 0, 0])
                self.meshPoint[index*3+i] += self.meshVelocity[index]*params.dt[None]

    @ti.func
    def detectCollision(self, index: int, pos: vec3r, vel: vec3r) -> bool:
        flag = False
        p0 = self.meshPoint[index*3+0]
        p1 = self.meshPoint[index*3+1]
        p2 = self.meshPoint[index*3+2]

        velRel = vel-self.meshVelocity[index]
        norm = self.meshNormal[index]
        if velRel.norm() != 0:
            t = (p1-pos).dot(norm)/velRel.dot(norm)
            if 0 <= t <= params.dt[None]:
                e1 = p1 - p0
                e2 = p2 - p0
                s = pos - p0
                s1 = velRel.cross(e2)
                s2 = s.cross(e1)

                s1e1 = s1.dot(e1)
                t = s2.dot(e2)/s1e1
                b1 = s1.dot(s)/s1e1
                b2 = s2.dot(velRel)/s1e1
                if 0 <= t <= params.dt[None] and 0 < b1 < 1 and 0 < b2 < 1 and 0 < b1+b2 < 1:
                    flag = True
        return flag

    @ti.func
    def resolveCollision(self, index: int, pos: vec3r, vel: vec3r) -> vec3r:
        norm = self.meshNormal[index]
        vRel = vel - self.meshVelocity[index]
        vN = vRel.dot(norm)
        vTangent = vRel - vN * norm
        vRelResolve = ti.Vector.zero(float, 3)
        if vTangent.norm() > ti.abs(params.frictionCoef*vN):
            vRelResolve = vTangent + params.frictionCoef*vN*vTangent.normalized()
        return vRelResolve + self.meshVelocity[index]





