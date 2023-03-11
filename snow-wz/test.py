import taichi as ti
ti.init(arch=ti.gpu)

a = ti.field(dtype=ti.int32)


ti.root.dense(ti.i, 2).place(a)
a[0] = 19
print(a[0])