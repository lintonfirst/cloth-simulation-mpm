import taichi as ti
import numpy as np
import random


ti.init(arch = ti.cuda)

res = 512
E = 4000 #拉伸部分惩罚
gamma = 500 # shear部分惩罚
k = 6000 # 法向挤压惩罚
damping_C = 0
dim = 2
n_grid = 128
dx = 1 / n_grid
inv_dx = 1 / dx
dt =  7 * 1e-5



N = 7 # 线的条数
typeIII = 800 # 每条线的typeIII点数
typeII = typeIII + 1
typeI = 1000 # 每条线的typeI点数

n_particle_1 = typeI * N #总typeI点数
n_particle_2 = typeII * N 
n_particle_3 = typeIII * N

n_particle_12 = n_particle_1 + n_particle_2
total_particle = n_particle_1 + n_particle_2 + n_particle_3
iterator_1 = ti.field(dtype = ti.f32,shape=n_particle_1)
iterator_2 = ti.field(dtype = ti.f32,shape=n_particle_2)
iterator_3 = ti.field(dtype = ti.f32,shape=n_particle_3)
iterator_12 = ti.field(dtype = ti.f32,shape=n_particle_12)
rho = 1#密度

#x,v,volume中数据的储存
# 0,...,n_particle_1-1是所有纤维的typeI粒子
# n_particle_1, ..., n_particle_12 -1 是所有的 typeII粒子
# n_particle_12, ..., total_particle -1 是所有的 typeIII粒子
# 详情于reset()实现
x = ti.Vector.field(dim, dtype = ti.f32, shape=total_particle)
v = ti.Vector.field(dim, dtype = ti.f32, shape=total_particle)
volume = ti.field(dtype = ti.f32, shape=total_particle)


D_inv = ti.Matrix.field(dim, dim,dtype = ti.f32,shape = n_particle_3)
d = ti.Matrix.field(dim, dim,dtype = ti.f32,shape = n_particle_3)
C = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=total_particle)
FE = ti.Matrix.field(dim, dim, dtype = ti.f32, shape = (n_particle_1 + n_particle_3))


grid_v = ti.Vector.field(dim, dtype=ti.f32, shape=(n_grid, n_grid))
grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid))
grid_v_star = ti.Vector.field(dim,dtype = ti.f32, shape = (n_grid, n_grid))
f = ti.Vector.field(dim,dtype = ti.f32, shape = (n_grid, n_grid))
ROTATE = ti.Matrix([[0,-1.0],[1.0,0]])

@ti.func
def QR2(F):
    f1 = ti.Vector([F[0,0],F[1,0]])
    f2 = ti.Vector([F[0,1],F[1,1]])
    r11 = f1.norm(1e-9)
    q1 = f1/r11
    r12 = f2.dot(q1)
    q2 = f2 - r12 * q1
    r22 = q2.norm(1e-9)
    q2/=r22
    Q = ti.Matrix.cols([q1,q2])
    R = ti.Matrix([[r11,r12],[0,r22]])
    return Q,R


cf = 0.00
@ti.func
def RETURN_MAPPING(FEp):
    Q,R = QR2(FEp)
    r12 = R[0,1]
    r22 = R[1,1]

    #A = ti.Matrix([[E*r11*(r11-1.0)+gamma*r12**2, gamma * r12 * r22],[gamma * r12 * r22, k * ti.log(r22) * float(r22 <1)]])
    a = gamma  * ti.abs(r22) 
    b = cf * k * (1 - r22)**2 * r22
    c = ti.abs(r12)
    
    t=FEp

    if a * c - b * float(r22 < 1) > 0:
        if r22 > 1:
            r12 = 0.0
            r22 = 1.0
        elif r22 < 0:
            print("alert!!!!!!!!!!!!")
            r12 = 0.0
        else:
            r12 = (r12 * b)/(c * a)
            
        R[0,1] = r12
        R[1,1] = r22
        t=Q@R
    return t

@ti.func
def damping_affine(C):
    Ck = 0.5 * (C.transpose() + C)
    Cs = 0.5 * (C - C.transpose())
    return Ck + (1-damping_C) * Cs

@ti.func
def mesh_x_ind(mid,l):
    return n_particle_1 + mid + mid//typeIII + l

@ti.kernel
def FORCE_INCREMENT():
    #print("---------FORCE_INCREMENT-------")
    for p in iterator_1:#type 1 粒子贡献的力
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        FEp = FE[p]
        Q,R = QR2(FEp)
        r11 = R[0,0]
        r12 = R[0,1]
        r22 = R[1,1]

        #A = ti.Matrix([[E*r11*(r11-1)+gamma*r12**2,gamma * r12 * r22],[gamma * r12 * r22, k * ti.log(r22) * float(r22 < 1)]])

        A = ti.Matrix([[E*r11*(r11-1)+gamma*r12**2,gamma * r12 * r22],[gamma * r12 * r22, -k * (1 - r22)**2 * r22 * float(r22 < 1)]])

        dphi_dF = Q@A@(R.inverse().transpose())

        w = [
            0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2
        ]
        dwdx = [
            fx-1.5, 2.0*(1.0-fx), fx-0.5
        ]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i,j])
                dweight = ti.Vector([ dwdx[i][0] * w[j][1], w[i][0] * dwdx[j][1] ]) * inv_dx
                f[base + offset] += -volume[p] * dphi_dF@FEp.transpose()@dweight

    for q in iterator_3:
        p = q + n_particle_12
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        mesh_p_0 = mesh_x_ind(q,0)
        mesh_p_1 = mesh_x_ind(q,1)


        base_0 = (x[mesh_p_0] * inv_dx - 0.5).cast(int)
        fx_0 = x[mesh_p_0]*inv_dx - base_0.cast(float)
        base_1 = (x[mesh_p_1] * inv_dx - 0.5).cast(int)
        fx_1 = x[mesh_p_1]* inv_dx - base_1.cast(float)

        w = [
            0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2
        ]
        dwdx = [
            fx-1.5, 2*(1.0-fx), fx-0.5
        ]#这只是B样条导数，不是基函数的导数
        w_0 = [
            0.5 * (1.5 - fx_0) ** 2, 0.75 - (fx_0 - 1.0) ** 2, 0.5 * (fx_0 - 0.5) ** 2
        ]
        w_1 = [
            0.5 * (1.5 - fx_1) ** 2, 0.75 - (fx_1 - 1.0) ** 2, 0.5 * (fx_1 - 0.5) ** 2
        ]

        FEp = FE[q+n_particle_1]

        Q,R = QR2(FEp)
        r11 = R[0,0]
        r12 = R[0,1]
        r22 = R[1,1]

        A = ti.Matrix([[E*r11*(r11-1)+gamma*r12**2,gamma * r12 * r22],[gamma * r12 * r22, -k * (1 - r22)**2 * r22 * float(r22 < 1)]])
        dphi_dF = Q@A@(R.inverse().transpose())

        dp2 = ti.Vector([d[q][0,1],d[q][1,1]])
        dphi_dF2 = ti.Vector([dphi_dF[0,1],dphi_dF[1,1]])

        Dp_inv1 = ti.Vector([D_inv[q][0,0],D_inv[q][0,1]])

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i,j])
                dweight = ti.Vector([ dwdx[i][0] * w[j][1], w[i][0] * dwdx[j][1] ]) * inv_dx
                weight_1 = w_1[i][0] * w_1[j][1]
                weight_0 = w_0[i][0] * w_0[j][1]
                f[base + offset] += -volume[p] * dphi_dF2 * dweight.dot(dp2)
                f_vert = dphi_dF@Dp_inv1
                f[base_1 + offset] += -volume[p] * weight_1 * f_vert
                f[base_0 + offset] += volume[p] * weight_0 * f_vert

    return

bound = 3
@ti.kernel
def GRID_COLLISION():
    #print("---------GRID_COLLISION-------")
    for i, j in grid_m:

        if grid_m[i, j] > 0:


            grid_v_star[i,j] = grid_v[i,j] +  f[i,j] * dt
            inv_m = 1 / grid_m[i, j]
            grid_v_star[i, j] = inv_m * grid_v_star[i, j]
            grid_v_star[i, j].y -= dt * 9.80 * 30
            # center collision circle
            sc = ti.Vector([2.0,25.0])
            dist = ti.Vector([i * dx - 0.40, j * dx - 0.5])
            if sc.x * dist.x**2 + sc.y * dist.y**2 < 0.04 :
                dist = (dist*sc).normalized()
                grid_v_star[i, j] -= dist * ti.min(0, grid_v_star[i, j].dot(dist))
                grid_v_star[i,j] = 0.1 * grid_v_star[i, j] #降低最下层的速度


            # box
            if i < bound and grid_v_star[i, j].x < 0:
                grid_v_star[i, j].x = 0
            if i > n_grid - bound and grid_v_star[i, j].x > 0:
                grid_v_star[i, j].x = 0
            if j < bound and grid_v_star[i, j].y < 0:
                grid_v_star[i, j].y = 0
            if j > n_grid - bound and grid_v_star[i, j].y > 0:
                grid_v_star[i, j].y = 0

    return

#还没做和边界的摩擦
@ti.kernel
def FRICTION():
    #这里grid_v变成速度
    for i,j in grid_v:
        if grid_m[i, j] > 0:           
            grid_v[i,j] = grid_v_star[i,j]
    return

@ti.kernel
def TRANSFER_TO_GRID():
    #print("-----------------------p2g------------------------------")
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5) ** 2]
        affine = C[p]
        mass = volume[p] * rho
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i,j])
                weight = w[i][0]*w[j][1]
                grid_m[base + offset] += (weight * mass + 1e-7)
                dpos = (offset.cast(float) - fx) * dx
                grid_v[base + offset] += weight * mass * (v[p] +  affine@dpos)

def GRID_STEP():
    #print("---------------------------grid dynamic--------------------")
    FORCE_INCREMENT()
    GRID_COLLISION()
    FRICTION()

@ti.kernel
def TRANSFER_TO_PARTICLE(): 
    #print("---------------------g2p----------------------")
    for p in iterator_12:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        # Quadratic B-spline
        w = [
            0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2
        ]


        new_v = ti.Vector.zero(ti.f32, 2)
        new_C = ti.Matrix.zero(ti.f32, 2, 2)

        #收集grid上的速度
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i,j])
                weight = w[i][0] * w[j][1]
                dpos_pre = offset.cast(float) - fx
                ind = base + offset
                g_v = grid_v[ind]
                new_C += 4 * weight * g_v.outer_product(dpos_pre) * inv_dx
                new_v += weight * g_v


        v[p] = new_v 
        C[p] = damping_affine(new_C) 

    for q in iterator_3:
        p = q + n_particle_12
        mesh_p_0 = mesh_x_ind(q,0)
        mesh_p_1 = mesh_x_ind(q,1)

        v[p] = 0.5 * (v[mesh_p_0] + v[mesh_p_1])
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        # Quadratic B-spline
        w = [
            0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2
        ]

        new_C = ti.Matrix.zero(ti.f32, 2, 2)
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i,j])
                weight = w[i][0] * w[j][1]
                dpos_pre = offset.cast(float) - fx
                ind = base + ti.Vector([i, j])
                g_v = grid_v[ind] 
                new_C += 4 * weight * g_v.outer_product(dpos_pre) * inv_dx
        C[p] = damping_affine(new_C)         
    return

@ti.kernel
def UPDATE_PARTICLE_STATE():#使用APIC
    #print("---------------------update----------------------")
    for p in iterator_1:#粒子1 更新
        
        x[p]+=dt * v[p]
        
        FE[p]+=dt * C[p]@FE[p]

        # if( ti.abs(FE[p].determinant()-1) > 0.01 ):
        #     print("type1 deformation warning: ",FE[p])

    for q in iterator_2:#粒子2 更新
        p = q + n_particle_1
        x[p]+=dt * v[p]

    for q in iterator_3:#粒子3更新
        mesh_p_0 = mesh_x_ind(q,0)
        mesh_p_1 = mesh_x_ind(q,1)

        p = q + n_particle_12
        nabla_v = C[p]
        dp1 = x[mesh_p_1] - x[mesh_p_0]
        dp2 = ti.Vector([d[q][0,1],d[q][1,1]])
        dp2+=dt * nabla_v@dp2
        d[q] = ti.Matrix.cols([dp1,dp2])
        FE[q + n_particle_1] = d[q]@(D_inv[q])
        x[p] = 0.5 * (x[mesh_p_0] + x[mesh_p_1])

@ti.kernel
def PLASTICITY():
    for p in FE:
        FE[p] = RETURN_MAPPING(FE[p])

length = 0.75
width = 0.001
unitL = length/typeIII
v1 = length/typeI
v2 = unitL/3
gap = 0.01 #纤维之间的空隙

@ti.kernel
def reset():#初始化位置，体积，形变梯度，D

    for q in range(n_particle_2):
        p = q + n_particle_1


        line = q//typeII # 所在第line根纤维
        vert = q%typeII #所在纤维的第vert个顶点
        x[p] = ti.Vector([0.2+ vert * unitL,0.7 + line * gap]) #每根纤维起始位置为 (0.2,0.7 +line * gap)

        if vert==0 :volume[p] = v2
        elif vert+1 == typeII:volume[p] = v2
        else: volume[p] = 2 * v2
        C[p] = ti.Matrix([[0,0],[0,0]])

    for q in range(n_particle_3):

        mid = q + n_particle_12

        p0 = mesh_x_ind(q,0)
        p1 = mesh_x_ind(q,1)

        x[mid] = 0.5 * (x[p0] + x[p1])
        Dmid1 = x[p1] - x[p0]
        Dmid2 = (ROTATE@Dmid1)
        Dmid2/=Dmid2.norm(1e-8)
        d[q] = ti.Matrix.cols([Dmid1,Dmid2])
        D_inv[q] = d[q].inverse()

        FE[q + n_particle_1] = ti.Matrix([[1.0,0],[0.0,1.0]])
        volume[mid] = v2
        C[mid] = ti.Matrix([[0,0],[0,0]])

    for p in range(n_particle_1):
        #随机在每根纤维上撒typeI点
        line = p//typeI
        x[p] = [ti.random() * (length + 0.004) + 0.198, (ti.random()-0.5) * width + 0.70 + line * gap]
        FE[p] = ti.Matrix([[1.0,0],[0.0,1.0]])
        volume[p] = v1
        C[p] = ti.Matrix([[0,0],[0,0]])


    return

gui = ti.GUI("Curve", (res, res))

def main():
    #result_dir =  "./results"
    #video_manager = ti.VideoManager(output_dir=result_dir, framerate=32, automatic_build=False)
    reset()
    for i in range(28000):
        gui.clear(0xFFFAFA)
        for _ in range(10):
            grid_v.fill([0.0, 0.0])
            grid_m.fill(0.0)
            f.fill([0.0,0.0])
            grid_v_star.fill([0.0,0.0])
            TRANSFER_TO_GRID()
            GRID_STEP()
            TRANSFER_TO_PARTICLE()
            UPDATE_PARTICLE_STATE()
            PLASTICITY()
        # print("----------------end ",i," ---------------")
        a = x.to_numpy()
        gui.circles(a[n_particle_1:n_particle_12], radius=2, color = 0x1E90FF)
        gui.circles(a[n_particle_12:total_particle], radius=2, color=0x1E90FF)
        gui.circles(a[0:n_particle_1],radius=1.0,color=0xFF1493)
        #gui.circle((0.35, 0.43), radius=102, color=0x068587)
        #video_manager.write_frame(gui.get_image())
        gui.show()
    #video_manager.make_video(gif = True, mp4 = False)
    #print(f'GIF video is saved to {video_manager.get_output_filename(".gif")}')
if __name__ == '__main__': main()