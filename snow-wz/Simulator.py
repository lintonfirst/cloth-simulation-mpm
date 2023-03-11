import taichi as ti
from ParticleSystem import ParticleSystem
from Grid import Grid
from CollideBodySystem import CollideBodySystem
from Params import params
from TimeBench import timeBench
from Rigidbody import Rigidbody


@ti.data_oriented
class Simulator:
    def __init__(self, ps: ParticleSystem):
        self.particleNum = 0
        self.particleSystem = ps
        self.grid = Grid(edgeGridNum=int(params.gridLength / params.cellSize),
                         minBoundary=[-params.gridLength/2, -params.gridLength/2, -0.2],
                         edgeLength=params.gridLength,
                         particleSystem=ps)
        self.collideBodySystem = CollideBodySystem()
        self.collideBodySystem.addGround()

        point = []
        for i in range(0, 2):
            for j in range(0, 2):
                for k in range(0, 2):
                    point.append([-1 + i * 2, -1 + j * 2, -1 + k * 2])

        meshIndex = [
            [0, 1, 2], [1, 2, 3],
            [4, 5, 6], [5, 6, 7],
            [0, 1, 4], [1, 4, 5],
            [2, 3, 6], [3, 6, 7],
            [0, 2, 4], [2, 4, 6],
            [1, 3, 5], [3, 5, 7]
        ]

        meshNormal = [
            [-1, 0, 0], [-1, 0, 0],
            [1, 0, 0], [1, 0, 0],
            [0, -1, 0], [0, -1, 0],
            [0, 1, 0], [0, 1, 0],
            [0, 0, -1], [0, 0, -1],
            [0, 0, 1], [0, 0, 1]
        ]

        # meshIndex = [
        #     [0, 1, 2], [1, 2, 3],
        #     # [4, 5, 6], [5, 6, 7],
        #     [0, 1, 4], [1, 4, 5],
        #     [2, 3, 6], [3, 6, 7],
        #     # [0, 2, 4], [2, 4, 6],
        #     [1, 3, 5], [3, 5, 7]
        # ]
        #
        # meshNormal = [
        #     [-1, 0, 0], [-1, 0, 0],
        #     # [1, 0, 0], [1, 0, 0],
        #     [0, -1, 0], [0, -1, 0],
        #     [0, 1, 0], [0, 1, 0],
        #     # [0, 0, -1], [0, 0, -1],
        #     [0, 0, 1], [0, 0, 1]
        # ]

        self.rigidbody = Rigidbody(1000, point, meshIndex, meshNormal)

        self.initParticleMass()

    @ti.kernel
    def initParticleMass(self):
        for index in self.particleSystem.mass:
            self.particleSystem.mass[index] = 4.0/3.0 * ti.math.pi * params.particleRadius**3 * params.density0
            print(index, self.particleSystem.mass[index])

    def step(self, frame):

        timeBench.addPhase("rasterize particles")
        self.rasterizeParticles()
        if frame == 0:
            timeBench.addPhase("compute density&volume")
            self.computeDensityAndVolume()
        timeBench.addPhase("compute grid force")
        self.computeGridForce()
        timeBench.addPhase("update grid velocity")
        self.updateGridVelocity()
        # timeBench.addPhase("handle grid collision")
        # self.handleGridCollision()
        timeBench.addPhase("update F Matrix")
        self.updateF()
        timeBench.addPhase("update particle velocity")
        self.updateParticleVelocity()
        timeBench.addPhase("handle particle collision")
        self.handleParticleCollision()

        if params.ifTwoWayCouple:
            self.rigidbody.collideWithPlane()
            self.rigidbody.updateVandW()
            self.handleParticleCollision2()
        if params.ifUseRigidbody:
            self.rigidbody.step()

        self.computeCFLCondition()
        timeBench.addPhase("update particle position")
        self.updateParticlePosition()
        # self.collideBodySystem.meshMove()

    @ti.kernel
    def computeCFLCondition(self):
        maxVelocity = 0.0
        for index in self.particleSystem.velocity:
            ti.atomic_max(maxVelocity, self.particleSystem.velocity[index].norm())

        CFLCondition = maxVelocity * params.dt[None] * self.grid.invGridR
        # print("\tCFLCondition: ", CFLCondition)
        if CFLCondition > 1:
            params.dt[None] = self.grid.gridR / maxVelocity
            params.dt[None] = ti.max(params.dt[None], params.minDt)

    @ti.kernel
    def rasterizeParticles(self):
        for i, j, k in self.grid.gridMass:
            self.grid.gridMass[i, j, k] = 0.0
            self.grid.gridVelocity[i, j, k] = ti.Vector.zero(float, 3)
            self.grid.gridForce[i, j, k] = ti.Vector.zero(float, 3)

        for index in self.particleSystem.position:
            baseIndex = ti.cast((self.particleSystem.position[index] - self.grid.minBoundary) * self.grid.invGridR,
                                ti.i32)
            for i, j, k in ti.ndrange(params.discretizeRange, params.discretizeRange, params.discretizeRange):
                indexOffset = ti.Vector([i + params.discretizeStart, j + params.discretizeStart, k + params.discretizeStart])
                positionOffset = self.particleSystem.position[index] - self.grid.minBoundary \
                                 - ti.cast(baseIndex + indexOffset, ti.f32) * self.grid.gridR

                weight = self.grid.weight(positionOffset)
                # if index == 0:
                #     print(indexOffset, positionOffset, weight)
                ti.atomic_add(self.grid.gridMass[baseIndex + indexOffset], self.particleSystem.mass[index] * weight)
                ti.atomic_add(self.grid.gridVelocity[baseIndex + indexOffset],
                              self.particleSystem.velocity[index] * self.particleSystem.mass[index] * weight)

        for i, j, k in self.grid.gridVelocity:
            if self.grid.gridMass[i, j, k] > 0:
                self.grid.gridVelocity[i, j, k] /= self.grid.gridMass[i, j, k]

    @ti.kernel
    def computeDensityAndVolume(self):
        for index in self.particleSystem.position:
            baseIndex = ti.cast((self.particleSystem.position[index] - self.grid.minBoundary) * self.grid.invGridR, ti.i32)
            density = 0.0
            for i, j, k in ti.ndrange(params.discretizeRange, params.discretizeRange, params.discretizeRange):
                indexOffset = ti.Vector([i + params.discretizeStart, j + params.discretizeStart, k + params.discretizeStart])
                positionOffset = self.particleSystem.position[index] - self.grid.minBoundary \
                                 - ti.cast(baseIndex + indexOffset, ti.f32) * self.grid.gridR
                weight = self.grid.weight(positionOffset)

                ti.atomic_add(density, self.grid.gridMass[baseIndex + indexOffset] * weight * self.grid.invGridR ** 3)

            self.particleSystem.volume[index] = self.particleSystem.mass[index] / density
            # print(index, self.particleSystem.volume[index], 4.0/3.0 * ti.math.pi * params.particleRadius**3)

    @ti.kernel
    def computeGridForce(self):
        for index in self.particleSystem.position:
            baseIndex = ti.cast((self.particleSystem.position[index] - self.grid.minBoundary) * self.grid.invGridR,
                                ti.i32)

            FE = self.particleSystem.FE[index]
            FP = self.particleSystem.FP[index]
            RE, SE = ti.polar_decompose(FE)

            JE = FE.determinant()
            JP = FP.determinant()
            h = ti.exp(params.hardenCoef * (1 - JP))
            h = ti.max(ti.min(h, 10.0), 0.1)
            # if index % 100 == 0:
            #     print(index, JP, h)
            Mu = self.particleSystem.mu0 * h
            Lambda = self.particleSystem.lambda0 * h
            sigma = (2.0 * Mu * (FE - RE) @ FE.transpose() + Lambda * (JE - 1.0) * JE * ti.Matrix.identity(float, 3)) #JE or J or None
            V_sigma = self.particleSystem.volume[index] * sigma

            for i, j, k in ti.ndrange(params.discretizeRange, params.discretizeRange, params.discretizeRange):
                indexOffset = ti.Vector([i + params.discretizeStart, j + params.discretizeStart, k + params.discretizeStart])
                positionOffset = self.particleSystem.position[index] - self.grid.minBoundary \
                                 - ti.cast(baseIndex + indexOffset, ti.f32) * self.grid.gridR
                dWeight = self.grid.dWeight(positionOffset)
                ti.atomic_add(self.grid.gridForce[baseIndex + indexOffset], -V_sigma @ dWeight)

        for i, j, k in self.grid.gridForce:
            # if self.grid.gridMass[i, j, k] > 0 and ti.random(float) < 0.05:
            #     print(i, j, k, self.grid.gridForce[i, j, k].norm()/self.grid.gridMass[i, j, k])
            self.grid.gridForce[i, j, k] += self.grid.gridMass[i, j, k] * ti.Vector([0, 0, -9.8])

    @ti.kernel
    def updateGridVelocity(self):
        for i, j, k in self.grid.gridVelocity:
            if self.grid.gridMass[i, j, k] > 0:
                self.grid.gridLastVelocity[i, j, k] = self.grid.gridVelocity[i, j, k]
                self.grid.gridVelocity[i, j, k] += params.dt[None] * self.grid.gridForce[i, j, k] / self.grid.gridMass[i, j, k]

    @ti.kernel
    def handleGridCollision(self):
        for i, j, k in self.grid.gridVelocity:
            if self.grid.gridMass[i, j, k] > 0.0:
                position = self.grid.minBoundary + ti.Vector([i, j, k]) * self.grid.gridR
                velocity = self.grid.gridVelocity[i, j, k]
                for index in ti.ndrange(self.collideBodySystem.meshNum):
                    collideFlag = self.collideBodySystem.detectCollision(index, position, velocity)
                    if collideFlag:
                        self.grid.gridVelocity[i, j, k] = self.collideBodySystem.resolveCollision(index, position, velocity)
                    if params.ifUseRigidbody:
                        self.grid.gridVelocity[i, j, k] = self.rigidbody.resolveCollision(-1, position, self.grid.gridVelocity[i, j, k], self.grid.gridMass[i, j, k])


    @ti.kernel
    def updateF(self):
        for index in self.particleSystem.position:
            baseIndex = ti.cast((self.particleSystem.position[index] - self.grid.minBoundary) * self.grid.invGridR,
                                ti.i32)
            velocityGrad = ti.Matrix.zero(float, 3, 3)
            for i, j, k in ti.ndrange(params.discretizeRange, params.discretizeRange, params.discretizeRange):
                indexOffset = ti.Vector([i + params.discretizeStart, j + params.discretizeStart, k + params.discretizeStart])
                positionOffset = self.particleSystem.position[index] - self.grid.minBoundary \
                                 - ti.cast(baseIndex + indexOffset, ti.f32) * self.grid.gridR
                dWeight = self.grid.dWeight(positionOffset)
                ti.atomic_add(velocityGrad, self.grid.gridVelocity[baseIndex + indexOffset].outer_product(dWeight))

            FE = self.particleSystem.FE[index]
            FP = self.particleSystem.FP[index]
            I = ti.Matrix.identity(float, 3)

            FE = (I + params.dt[None] * velocityGrad) @ FE
            F = FE @ FP

            U, S, V = ti.svd(FE)
            for t in ti.static(range(3)):
                S[t, t] = ti.math.clamp(S[t, t], 1 - params.thetaC, 1 + params.thetaS)

            self.particleSystem.FE[index] = U @ S @ V.transpose()
            self.particleSystem.FP[index] = V @ S.inverse() @ U.transpose() @ F

    @ti.kernel
    def updateParticleVelocity(self):
        for index in self.particleSystem.position:
            baseIndex = ti.cast((self.particleSystem.position[index] - self.grid.minBoundary) * self.grid.invGridR,
                                ti.i32)
            vPIC = ti.Vector.zero(float, 3)
            vFLIP = self.particleSystem.velocity[index]

            for i, j, k in ti.ndrange(params.discretizeRange, params.discretizeRange, params.discretizeRange):
                indexOffset = ti.Vector([i + params.discretizeStart, j + params.discretizeStart, k + params.discretizeStart])
                positionOffset = self.particleSystem.position[index] - self.grid.minBoundary \
                                 - ti.cast(baseIndex + indexOffset, ti.f32) * self.grid.gridR
                weight = self.grid.weight(positionOffset)
                ti.atomic_add(vPIC, self.grid.gridVelocity[baseIndex + indexOffset] * weight)
                ti.atomic_add(vFLIP, (self.grid.gridVelocity[baseIndex + indexOffset] - self.grid.gridLastVelocity[baseIndex + indexOffset]) * weight)

            self.particleSystem.velocity[index] = (1 - params.alpha) * vPIC + params.alpha * vFLIP

    @ti.kernel
    def handleParticleCollision(self):
        for index in self.particleSystem.position:
            position = self.particleSystem.position[index]
            velocity = self.particleSystem.velocity[index]
            for collisionIndex in ti.ndrange(self.collideBodySystem.meshNum):
                collideFlag = self.collideBodySystem.detectCollision(collisionIndex, position, velocity)
                if collideFlag:
                    self.particleSystem.velocity[index] = \
                        self.collideBodySystem.resolveCollision(collisionIndex, position, velocity)
            if params.ifUseRigidbody:
                self.particleSystem.velocity[index] = self.rigidbody.resolveCollision(index, position, self.particleSystem.velocity[index], self.particleSystem.mass[index])

    @ti.kernel
    def handleParticleCollision2(self):
        for index in self.particleSystem.position:
            position = self.particleSystem.position[index]
            velocity = self.particleSystem.velocity[index]
            self.particleSystem.velocity[index] = self.rigidbody.resolveParticleCollision(position, velocity, self.particleSystem.mass[index])

    @ti.kernel
    def updateParticlePosition(self):
        for index in self.particleSystem.position:
            self.particleSystem.position[index] += params.dt[None] * self.particleSystem.velocity[index]
