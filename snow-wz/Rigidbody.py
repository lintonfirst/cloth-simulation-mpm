import taichi as ti
from Params import params, vec3r


@ti.func
def updateRotation(w, q):
    wv = ti.Vector([w.x, w.y, w.z])
    qv = ti.Vector([q.x, q.y, q.z])
    cw = w.w * q.w - wv.dot(qv)
    cv = wv.cross(qv) + q.w * wv + w.w * qv

    cv = ti.Vector([cv.x + q.x, cv.y + q.y, cv.z + q.z])
    cw = cw + q.w
    if cv.norm() > 0:
        cv = cv.normalized() * ti.math.sqrt(1 - cw ** 2)
    c = ti.Vector([cv.x, cv.y, cv.z, cw])
    return c


@ti.func
def getCrossMatrix(v):
    tmp = ti.Matrix.zero(float, 3, 3)
    tmp[0, 1] = -v.z
    tmp[0, 2] = v.y
    tmp[1, 0] = v.z
    tmp[1, 2] = -v.x
    tmp[2, 0] = -v.y
    tmp[2, 1] = v.x
    return tmp


@ti.func
def getRotateMatrix(q):
    return ti.Matrix([
        [1 - 2 * q.y ** 2 - 2 * q.z ** 2, 2 * q.x * q.y - 2 * q.z * q.w, 2 * q.x * q.z + 2 * q.y * q.w],
        [2 * q.x * q.y + 2 * q.z * q.w, 1 - 2 * q.x ** 2 - 2 * q.z ** 2, 2 * q.y * q.z - 2 * q.x * q.w],
        [2 * q.x * q.z - 2 * q.y * q.w, 2 * q.y * q.z + 2 * q.x * q.w, 1 - 2 * q.x ** 2 - 2 * q.y ** 2]
    ])


@ti.data_oriented
class Rigidbody:
    def __init__(self, mass, pointList, meshIndex, meshNormal):
        self.centroid = ti.Vector.field(3, dtype=float, shape=())
        self.pointNum = len(pointList)
        self.meshNum = len(meshIndex)

        self.Point = ti.Vector.field(3, dtype=float, shape=self.pointNum)
        self.actualPoint = ti.Vector.field(3, dtype=float, shape=self.pointNum)
        self.MeshIndex = ti.field(dtype=int, shape=(self.meshNum, 3))
        self.MeshNormal = ti.Vector.field(3, dtype=float, shape=self.meshNum)
        self.MeshPoint = ti.Vector.field(3, dtype=float, shape=self.meshNum * 3)

        self.mass = mass
        self.inertiaRef = ti.Matrix.field(3, 3, dtype=float, shape=())
        self.inertia = ti.Matrix.field(3, 3, dtype=float, shape=())
        self.v = ti.Vector.field(3, dtype=float, shape=())
        self.w = ti.Vector.field(3, dtype=float, shape=())
        self.x = ti.Vector.field(3, dtype=float, shape=())
        self.q = ti.Vector.field(4, dtype=float, shape=())
        self.torque = ti.Vector.field(3, dtype=float, shape=())
        self.force = ti.Vector.field(3, dtype=float, shape=())

        self.muN = 0.3
        self.muT = 0.2

        self.centroid[None] = ti.Vector([0.0, 0.0, 0.0])
        for index in range(0, self.pointNum):
            self.Point[index] = ti.Vector(pointList[index])*params.rigidBodyScale
            # ti.atomic_add(self.centroid[None], self.Point[index])
            self.centroid[None] += self.Point[index]
        self.centroid[None] /= self.pointNum

        for index in range(0, self.meshNum):
            for j in range(0, 3):
                self.MeshIndex[index, j] = meshIndex[index][j]
            self.MeshNormal[index] = ti.Vector(meshNormal[index])

        self.x[None] = ti.Vector(params.rigidBodyInitX)
        self.v[None] = ti.Vector(params.rigidBodyInitV)

        self.initialize()
        self.reset()

    @ti.kernel
    def initialize(self):


        # self.x[None] = ti.Vector([0, 0, -3])
        # self.v[None] = ti.Vector([0, 0, 0])

        omega = params.rigidBodyOmega
        self.q[None] = ti.Vector([0, ti.sin(omega / 2), 0, ti.cos(omega / 2)])
        rotateMat = getRotateMatrix(self.q[None])

        self.inertiaRef[None] = ti.Matrix.zero(float, 3, 3)
        for index in range(0, self.pointNum):
            r = self.Point[index] - self.centroid[None]
            ti.atomic_add(self.inertiaRef[None],
                          (r.dot(r) * ti.Matrix.identity(float, 3) - r.outer_product(r)) * self.mass / self.pointNum)

        self.inertia[None] = rotateMat @ self.inertiaRef[None] @ rotateMat.transpose()

        pos = self.centroid[None] + self.x[None]
        for index in range(0, self.pointNum):
            self.actualPoint[index] = pos + rotateMat @ self.Point[index]
            # print(self.actualPoint[index])

        for index in range(0, self.meshNum):
            self.MeshPoint[index * 3 + 0] = self.actualPoint[self.MeshIndex[index, 0]]
            self.MeshPoint[index * 3 + 1] = self.actualPoint[self.MeshIndex[index, 1]]
            self.MeshPoint[index * 3 + 2] = self.actualPoint[self.MeshIndex[index, 2]]

    def collideWithPlane(self):
        self.collisionImpulse(ti.Vector([0, 0, -5]), ti.Vector([0, 0, 1]))

    def step(self):
        self.updateXandQ()
        self.reset()

    @ti.kernel
    def reset(self):
        self.force[None] = ti.Vector([0, 0, -9.8 * self.mass])
        self.torque[None] = ti.Vector.zero(float, 3)

    @ti.kernel
    def collisionImpulse(self, p: vec3r, normal: vec3r):
        avgCollisionPoint = ti.Vector([0.0, 0.0, 0.0])
        averageNum = 0

        pos = self.centroid[None] + self.x[None]
        rotateMat = getRotateMatrix(self.q[None])
        for index in self.Point:
            flag = True
            ri = self.Point[index] - self.centroid[None]
            xi = pos + rotateMat @ ri

            if (xi - p).dot(normal) >= 0.0:
                flag = False
            else:
                vi = self.v[None] + self.w[None].cross(rotateMat @ ri)
                if vi.dot(normal) >= 0.0:
                    flag = False

            if flag:
                avgCollisionPoint += self.Point[index]
                ti.atomic_add(avgCollisionPoint, self.Point[index])
                ti.atomic_add(averageNum, 1)

        if averageNum > 0:
            ri = avgCollisionPoint / averageNum - self.centroid[None]
            Rri = rotateMat @ ri
            xi = pos + Rri
            vi = self.v[None] + self.w[None].cross(Rri)

            viN = vi.dot(normal) * normal
            viT = vi - viN

            a = max(1 - self.muT * (1 + self.muN) * viN.norm() / viT.norm(), 0)
            viN_new = - self.muN * viN
            viT_new = a * viT

            vi_new = viN_new + viT_new

            # print(Rri)
            RriMatrix = getCrossMatrix(Rri)
            K = ti.Matrix.identity(float, 3) / self.mass - RriMatrix @ self.inertia[None].inverse() @ RriMatrix
            impulse = K.inverse() @ (vi_new - vi)
            force = impulse / params.dt[None]
            self.force[None] += force
            self.torque[None] += Rri.cross(force)
            # self.v[None] += impulse/self.mass
            # self.w[None] += self.inertia[None].inverse() @ (RriMatrix @ impulse)
            # print(self.inertia[None].inverse(), self.w[None], RriMatrix @ impulse)

    @ti.kernel
    def updateVandW(self):
        self.v[None] += params.dt[None] * self.force[None] / self.mass
        # self.w[None] += params.dt[None] * self.inertia[None].inverse() @ self.torque[None]
        # print(self.v[None])

    @ti.kernel
    def updateXandQ(self):
        self.x[None] += params.dt[None] * self.v[None]
        # print("velocity", self.v[None], self.force[None]/self.mass)

        # wt = 0.5 * params.dt[None] * self.w[None]
        # self.q[None] = updateRotation(ti.Vector([wt.x, wt.y, wt.z, 0.0]), self.q[None])

        rotateMat = getRotateMatrix(self.q[None])
        pos = self.centroid[None] + self.x[None]
        # print(self.q[None], rotateMat)

        self.inertia[None] = rotateMat @ self.inertiaRef[None] @ rotateMat.transpose()

        # print(rotateMat)
        for index in range(0, self.pointNum):
            self.actualPoint[index] = pos + rotateMat @ (self.Point[index] - self.centroid[None])
            # print(self.actualPoint[index]-pos)

        for index in range(0, self.meshNum):
            self.MeshPoint[index * 3 + 0] = self.actualPoint[self.MeshIndex[index, 0]]
            self.MeshPoint[index * 3 + 1] = self.actualPoint[self.MeshIndex[index, 1]]
            self.MeshPoint[index * 3 + 2] = self.actualPoint[self.MeshIndex[index, 2]]

    @ti.func
    def resolveCollision(self, particleIndex: int, pos: vec3r, vel: vec3r, mass: float) -> vec3r:
        tMin = ti.math.inf
        velRelMin = ti.Vector.zero(float, 3)
        meshVelMin = ti.Vector.zero(float, 3)
        indexMin = 0

        rotateMat = getRotateMatrix(self.q[None])
        ifCollide = False
        for index in range(0, self.meshNum):
            p0 = self.actualPoint[self.MeshIndex[index, 0]]
            p1 = self.actualPoint[self.MeshIndex[index, 1]]
            p2 = self.actualPoint[self.MeshIndex[index, 2]]

            avgRi = (self.Point[self.MeshIndex[index, 0]] \
                     + self.Point[self.MeshIndex[index, 1]] \
                     + self.Point[self.MeshIndex[index, 2]]) / 3 - self.centroid[None]

            # meshVelocity = self.v[None] + self.w[None].cross(rotateMat @ avgRi)
            meshVelocity = self.v[None] + self.w[None].cross(pos - self.centroid[None]-self.x[None])
            norm = rotateMat @ self.MeshNormal[index]
            velRel = vel - meshVelocity

            if velRel.norm() != 0:
                t = (p1 - pos).dot(norm) / velRel.dot(norm)
                if 0 <= t <= params.dt[None]:
                    e1 = p1 - p0
                    e2 = p2 - p0
                    s = pos - p0
                    s1 = velRel.cross(e2)
                    s2 = s.cross(e1)

                    s1e1 = s1.dot(e1)
                    t = s2.dot(e2) / s1e1
                    b1 = s1.dot(s) / s1e1
                    b2 = s2.dot(velRel) / s1e1
                    if 0 <= t <= params.dt[None] and 0 < b1 < 1 and 0 < b2 < 1 and 0 < b1 + b2 < 1:
                        ifCollide = True
                        if t < tMin:
                            tMin = t
                            velRelMin = velRel
                            meshVelMin = meshVelocity
                            indexMin = index
        ri = rotateMat.inverse() @ (pos - self.centroid[None] - self.x[None])
        # if ti.abs(ri.x) < 1 and ti.abs(ri.y) < 1 and ti.abs(ri.z) < 1:
        #     print(pos, vel)
        returnVel = vel
        if ifCollide:
            norm = rotateMat @ self.MeshNormal[indexMin]
            d = (params.dt[None] - tMin) * velRelMin.dot(norm)
            vN = velRelMin.dot(norm)
            vTangent = velRelMin - vN * norm
            # print("v", velRelMin, norm)
            vRelResolve = ti.Vector.zero(float, 3)
            if vTangent.norm() > ti.abs(params.frictionCoef * vN):
                vRelResolve = vTangent + params.frictionCoef * vN * vTangent.normalized()
            returnVel = vRelResolve + meshVelMin

            impulse = mass * (returnVel - vel)
            collidePoint = pos + vel * tMin
            Rri = collidePoint - self.centroid[None] - self.x[None]
            ti.atomic_add(self.force[None], -impulse / params.dt[None])
            # if d < 0:
            #     ti.atomic_add(self.force[None], 50 * d * norm)
            ti.atomic_add(self.torque[None], Rri.cross(-impulse / params.dt[None]))
            # print("addforce", -impulse/params.dt[None], pos, returnVel)
            # if -0.4 < pos.y < 0.4 and particleIndex > 0:
        return returnVel

    @ti.func
    def resolveParticleCollision(self, pos: vec3r, vel: vec3r, mass: float) -> vec3r:
        tMin = ti.math.inf
        velRelMin = ti.Vector.zero(float, 3)
        meshVelMin = ti.Vector.zero(float, 3)
        indexMin = 0

        rotateMat = getRotateMatrix(self.q[None])
        ifCollide = False
        for index in range(0, self.meshNum):
            p0 = self.actualPoint[self.MeshIndex[index, 0]]
            p1 = self.actualPoint[self.MeshIndex[index, 1]]
            p2 = self.actualPoint[self.MeshIndex[index, 2]]

            avgRi = (self.Point[self.MeshIndex[index, 0]] \
                     + self.Point[self.MeshIndex[index, 1]] \
                     + self.Point[self.MeshIndex[index, 2]]) / 3 - self.centroid[None]

            # meshVelocity = self.v[None] + self.w[None].cross(rotateMat @ avgRi)
            meshVelocity = self.v[None] + self.w[None].cross(pos - self.centroid[None]-self.x[None])

            velRel = vel - meshVelocity
            norm = rotateMat @ self.MeshNormal[index]
            if velRel.norm() != 0:
                t = (p1 - pos).dot(norm) / velRel.dot(norm)
                if 0 <= t <= params.dt[None]:
                    e1 = p1 - p0
                    e2 = p2 - p0
                    s = pos - p0
                    s1 = velRel.cross(e2)
                    s2 = s.cross(e1)

                    s1e1 = s1.dot(e1)
                    t = s2.dot(e2) / s1e1
                    b1 = s1.dot(s) / s1e1
                    b2 = s2.dot(velRel) / s1e1
                    if 0 <= t <= params.dt[None] and 0 < b1 < 1 and 0 < b2 < 1 and 0 < b1 + b2 < 1:
                        ifCollide = True
                        if t < tMin:
                            tMin = t
                            velRelMin = velRel
                            meshVelMin = meshVelocity
                            indexMin = index
        returnVel = vel
        if ifCollide:
            norm = rotateMat @ self.MeshNormal[indexMin]

            d = (params.dt[None] - tMin) * velRelMin.dot(norm)
            vN = velRelMin.dot(norm)
            vTangent = velRelMin - vN * norm
            # print("v", velRelMin, norm)
            vRelResolve = ti.Vector.zero(float, 3)
            if vTangent.norm() > ti.abs(params.frictionCoef * vN):
                vRelResolve = vTangent + params.frictionCoef * vN * vTangent.normalized()
            returnVel = vRelResolve + meshVelMin
        return returnVel
