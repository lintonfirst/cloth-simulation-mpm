import math

import numpy as np
from Params import params
import random


class SceneManager:
    def __init__(self):
        self.particleNum = 0
        self.fluidParticleNum = 0
        self.position = np.ndarray(shape=(0, 3))
        self.velocity = np.ndarray(shape=(0, 3))

    def addParticles(self, position: np.ndarray, velocity: np.ndarray):
        newParticleNum = position.shape[0]
        self.particleNum += newParticleNum
        self.fluidParticleNum += newParticleNum
        self.position = np.concatenate((self.position, position))
        self.velocity = np.concatenate((self.velocity, velocity))

    def CubeThrow(self, camera):
        params.frictionCoef = 0.6
        edgeNum = 25
        groupSize = edgeNum ** 3
        groupNum = 1
        particleNum = groupSize * groupNum
        position = np.ndarray(shape=(particleNum, 3))
        velocity = np.ndarray(shape=(particleNum, 3))
        for groupId in range(groupNum):
            for i in range(edgeNum):
                for j in range(edgeNum):
                    for k in range(edgeNum):
                        index = i * edgeNum ** 2 + j * edgeNum + k + groupSize * groupId
                        iValue, jValue, kValue = 0, 0, 0
                        if params.ifRandom:
                            iValue = random.random()
                            jValue = random.random()
                            kValue = random.random()
                        else:
                            iValue = i / edgeNum
                            jValue = j / edgeNum
                            kValue = k / edgeNum
                        position[index] = [iValue * 2.0 - 8.0 + 4.0 * groupId,
                                           jValue * 2.0 - 1.0,
                                           kValue * 2.0 - 1.0 + 1.0 * groupId]
                        velocity[index] = [5, 0, 3]
        self.addParticles(position, velocity)

        camera.position(0.0, 15.0, 4.0)
        camera.lookat(0.0, 0.0, -4.0)
        camera.up(0.0, 0.0, 1.0)

    def TwoCubesCollide(self, camera):
        params.gridLength = 10
        params.cellSize = params.gridLength / 128
        params.particleRadius = 0.05
        params.frictionCoef = 0.4
        params.youngsModulus = 1.4e5
        params.thetaC = 2.5e-1
        params.thetaS = 7.5e-2
        params.hardenCoef = 10

        XedgeLength, YedgeLength, ZedgeLength = 2.0, 2.0, 2.0
        XedgeNum, YedgeNum, ZedgeNum = int(XedgeLength / params.particleRadius/2), int(
            YedgeLength / params.particleRadius/2), int(ZedgeLength / params.particleRadius/2)
        groupSize = XedgeNum * YedgeNum * ZedgeNum
        groupNum = 2
        particleNum = groupSize * groupNum
        position = np.ndarray(shape=(particleNum, 3))
        velocity = np.ndarray(shape=(particleNum, 3))
        for groupId in range(groupNum):
            for k in range(ZedgeNum):
                for j in range(YedgeNum):
                    for i in range(XedgeNum):
                        index = k * XedgeNum * YedgeNum + j * XedgeNum + i + groupSize * groupId
                        iValue, jValue, kValue = 0, 0, 0
                        if params.ifRandom:
                            iValue = random.random()
                            jValue = random.random()
                            kValue = random.random()
                        else:
                            iValue = i / XedgeNum
                            jValue = j / YedgeNum
                            kValue = k / ZedgeNum
                        position[index] = [iValue * XedgeLength - 3.0 + 4.0 * groupId,
                                           jValue * YedgeLength - 1.0,
                                           kValue * ZedgeLength - 1.0]
                        if groupId == 0:
                            velocity[index] = [20, 0, 5]
                        else:
                            velocity[index] = [-20, 0, 5]
        self.addParticles(position, velocity)

        camera.position(0.0, 15.0, 4.0)
        camera.lookat(0.0, 0.0, -2.0)
        camera.up(0.0, 0.0, 1.0)

    def TwoBallsCollide(self, camera):
        params.gridLength = 2
        params.cellSize = 1/45

        params.particleRadius = params.cellSize/2/2
        params.frictionCoef = 0.4
        params.youngsModulus = 4.8e5
        params.thetaC = 1.9e-2
        params.thetaS = 7.5e-3
        params.hardenCoef = 10

        Radius = 0.3 * 0.5
        beishu = 1.0
        XedgeNum, YedgeNum, ZedgeNum = int(Radius * beishu / params.particleRadius), int(
            Radius * beishu / params.particleRadius), int(Radius * beishu / params.particleRadius)
        groupSize = XedgeNum * YedgeNum * ZedgeNum
        groupNum = 1
        positionList = []
        velocityList = []
        for groupId in range(groupNum):
            originX = Radius - 0.3 + 0.6 * groupId
            originY = Radius - 0.0
            originZ = Radius + 1.0
            for k in range(ZedgeNum):
                for j in range(YedgeNum):
                    for i in range(XedgeNum):
                        index = k * XedgeNum * YedgeNum + j * XedgeNum + i + groupSize * groupId
                        iValue, jValue, kValue = 0, 0, 0
                        if params.ifRandom:
                            iValue = random.random()
                            jValue = random.random()
                            kValue = random.random()
                        else:
                            iValue = i / XedgeNum
                            jValue = j / YedgeNum
                            kValue = k / ZedgeNum
                        x = iValue * Radius * 2 - 0.3 + 0.6 * groupId
                        y = jValue * Radius * 2 - 0.0
                        z = kValue * Radius * 2 + 1.0
                        if (x - originX)**2 + (y - originY)**2 + (z - originZ)**2 < Radius**2:
                            positionList.append([x, y, z])
                            if groupId == 0:
                                velocityList.append([0, 0, 0])
                            else:
                                velocityList.append([-10, 0, 4])
        position = np.array(positionList)
        velocity = np.array(velocityList)
        self.addParticles(position, velocity)
        print("particleNum: ", position.shape[0])

        camera.position(0.0, 3.0, 0.5)
        camera.lookat(0.0, 0.0, 0.0)
        camera.up(0.0, 0.0, 1.0)

    def CubeFall(self, camera):
        params.cellSize = 0.1
        params.gridLength = 8
        params.particleRadius = 0.02
        params.frictionCoef = 0.2
        params.thetaC = 1.9e-2
        self.Cube(-3.0)

        camera.position(0.0, 8.0, 0.0)
        camera.lookat(0.0, 0.0, -4.0)
        camera.up(0.0, 0.0, 1.0)

    def DecliningCubeFall(self, camera):
        params.cellSize = 0.1
        params.gridLength = 8
        params.particleRadius = 0.02
        params.frictionCoef = 0.2
        params.thetaC = 1.9e-2
        params.hardenCoef = 30
        self.DecliningCube(-2.0)

        camera.position(0.0, 8.0, 0.0)
        camera.lookat(0.0, 0.0, -4.0)
        camera.up(0.0, 0.0, 1.0)

        params.ifUseRigidbody = False
        params.ifTwoWayCouple = False

    def BreakOnWedge(self, camera):
        params.ifUseRigidbody = True
        params.ifTwoWayCouple = False
        params.rigidBodyInitX = [0, 0, -3]
        params.rigidBodyInitV = [0, 0, 0]
        params.rigidBodyOmega = math.pi / 4
        params.rigidBodyScale = 0.55

        params.cellSize = 0.06
        params.gridLength = 6
        params.particleRadius = 0.02
        params.frictionCoef = 1.0

        self.Cube(-2.0)
        camera.position(0.0, 8.0, 0.0)
        camera.lookat(0.0, 0.0, -4.0)
        camera.up(0.0, 0.0, 1.0)

    def PushRigidbody(self, camera):
        params.ifUseRigidbody = True
        params.rigidBodyInitX = [-6, 0, -3.5]
        params.rigidBodyInitV = [3, 0, 0]
        self.LongCube()

        camera.position(0.0, 15.0, 4.0)
        camera.lookat(0.0, 0.0, -4.0)
        camera.up(0.0, 0.0, 1.0)

    def TwoWayCouple(self, camera):
        params.ifUseRigidbody = True
        params.ifTwoWayCouple = True
        params.rigidBodyInitX = [-6, 0, -1]
        params.rigidBodyInitV = [8 ,0, 0]
        self.LongCube()

        camera.position(0,0,3)
        camera.lookat(0.0, 0.0, 0.0)
        camera.up(0.0, 0.0, 1.0)



    def BigCube(self, camera):
        self.BigCube()
        camera.position(-10.0, 5.0, 0.0)
        camera.lookat(0.0, 0.0, -4.0)
        camera.up(0.0, 0.0, 1.0)

    def Cube(self, zBoundary):
        XedgeLength, YedgeLength, ZedgeLength = 2.0, 1.0, 1.0
        XedgeNum, YedgeNum, ZedgeNum = int(XedgeLength / params.particleRadius), int(
            YedgeLength / params.particleRadius), int(ZedgeLength / params.particleRadius)
        groupSize = XedgeNum * YedgeNum * ZedgeNum
        particleNum = groupSize
        position = np.ndarray(shape=(particleNum, 3))
        velocity = np.ndarray(shape=(particleNum, 3))
        for k in range(ZedgeNum):
            for j in range(YedgeNum):
                for i in range(XedgeNum):
                    index = k * XedgeNum * YedgeNum + j * XedgeNum + i
                    iValue, jValue, kValue = 0, 0, 0
                    if params.ifRandom:
                        iValue = random.random()
                        jValue = random.random()
                        kValue = random.random()
                    else:
                        iValue = i / XedgeNum
                        jValue = j / YedgeNum
                        kValue = k / ZedgeNum
                    position[index] = [iValue * XedgeLength - XedgeLength / 2, jValue * YedgeLength - YedgeLength / 2,
                                       kValue * ZedgeLength + zBoundary]
                    velocity[index] = [0, 0, 0]
        self.addParticles(position, velocity)

    def DecliningCube(self, zBoundary):
        XedgeLength, YedgeLength, ZedgeLength = 1.0, 1.0, 2.0
        XedgeNum, YedgeNum, ZedgeNum = int(XedgeLength / params.particleRadius), int(
            YedgeLength / params.particleRadius), int(ZedgeLength / params.particleRadius)
        groupSize = XedgeNum * YedgeNum * ZedgeNum
        particleNum = groupSize
        position = np.ndarray(shape=(particleNum, 3))
        velocity = np.ndarray(shape=(particleNum, 3))
        omega = math.pi / 6
        for k in range(ZedgeNum):
            for j in range(YedgeNum):
                for i in range(XedgeNum):
                    index = k * XedgeNum * YedgeNum + j * XedgeNum + i
                    iValue, jValue, kValue = 0, 0, 0
                    if params.ifRandom:
                        iValue = random.random()
                        jValue = random.random()
                        kValue = random.random()
                    else:
                        iValue = i / XedgeNum
                        jValue = j / YedgeNum
                        kValue = k / ZedgeNum
                    x = iValue * XedgeLength - XedgeLength / 2
                    y = jValue * YedgeLength - YedgeLength / 2
                    z = kValue * ZedgeLength - ZedgeLength / 2
                    position[index] = [math.cos(omega) * x - math.sin(omega) * z, y,
                                       math.sin(omega) * x + math.cos(omega) * z + zBoundary]
                    velocity[index] = [0, 0, 0]
        self.addParticles(position, velocity)

    def BigCube(self):
        XedgeLength, YedgeLength, ZedgeLength = 4.0, 4.0, 2.0
        XedgeNum, YedgeNum, ZedgeNum = int(XedgeLength / 0.1), int(YedgeLength / 0.1), int(ZedgeLength / 0.1)
        groupSize = XedgeNum * YedgeNum * ZedgeNum
        particleNum = groupSize
        position = np.ndarray(shape=(particleNum, 3))
        velocity = np.ndarray(shape=(particleNum, 3))
        for k in range(ZedgeNum):
            for j in range(YedgeNum):
                for i in range(XedgeNum):
                    index = k * XedgeNum * YedgeNum + j * XedgeNum + i
                    iValue, jValue, kValue = 0, 0, 0
                    if params.ifRandom:
                        iValue = random.random()
                        jValue = random.random()
                        kValue = random.random()
                    else:
                        iValue = i / XedgeNum
                        jValue = j / YedgeNum
                        kValue = k / ZedgeNum
                    position[index] = [iValue * XedgeLength - XedgeLength / 2, jValue * YedgeLength - YedgeLength / 2,
                                       kValue * ZedgeLength - 4.9]
                    velocity[index] = [0, 0, 0]
        self.addParticles(position, velocity)

    def LongCube(self):
        XedgeLength, YedgeLength, ZedgeLength = 6.0, 4.0, 2.0
        XedgeNum, YedgeNum, ZedgeNum = int(XedgeLength / 0.1), int(YedgeLength / 0.1), int(ZedgeLength / 0.1)
        groupSize = XedgeNum * YedgeNum * ZedgeNum
        particleNum = groupSize
        position = np.ndarray(shape=(particleNum, 3))
        velocity = np.ndarray(shape=(particleNum, 3))
        for k in range(ZedgeNum):
            for j in range(YedgeNum):
                for i in range(XedgeNum):
                    index = k * XedgeNum * YedgeNum + j * XedgeNum + i
                    iValue, jValue, kValue = 0, 0, 0
                    if params.ifRandom:
                        iValue = random.random()
                        jValue = random.random()
                        kValue = random.random()
                    else:
                        iValue = i / XedgeNum
                        jValue = j / YedgeNum
                        kValue = k / ZedgeNum
                    position[index] = [iValue * XedgeLength - XedgeLength / 2, jValue * YedgeLength - YedgeLength / 2,
                                       kValue * ZedgeLength - 4.9]
                    velocity[index] = [0, 0, 0]
        self.addParticles(position, velocity)
