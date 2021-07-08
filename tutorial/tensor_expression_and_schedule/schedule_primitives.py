# split: factor/npart, tile, fuse, reorder, bind
from __future__ import absolute_import, print_function

import tvm
from tvm import te
import numpy as np

# declare some variables for use later
n = te.var('n')
m = te.var('m')

A = te.placeholder((m, n), name='A')
B = te.placeholder((m, n), name='B')
C = te.compute((m, n), lambda i, j: A[i, j]*B[i, j], name='C')

s = te.create_schedule([C.op])

print(tvm.lower(s, [A, B, C], simple_mode=True))

A = te.placeholder((m, ), name='A')
B = te.compute((m,), lambda i: A[i] * 2, name='B')

s = te.create_schedule(B.op)
xo, xi = s[B].split(B.op.axis[0], factor=32)
print(tvm.lower(s, [A, B], simple_mode=True))

A = te.placeholder((m, ), name='A')
B = te.compute((m, ), lambda i: A[i], name='B')

s = te.create_schedule(B.op)
bx, tx = s[B].split(B.op.axis[0], nparts=32)
print(tvm.lower(s, [A, B], simple_mode=True))

A = te.placeholder((m, n), name='A')
B = te.compute((m, n), lambda i, j: A[i, j], name='B')

s = te.create_schedule(B.op)
xo, yo, xi, yi = s[B].tile(B.op.axis[0], B.op.axis[1], x_factor=10, y_factor=5)
print(tvm.lower(s, [A, B], simple_mode=True))

A = te.placeholder((m, n), name='A')
B = te.compute((m, n), lambda i, j: A[i, j], name='B')

s = te.create_schedule(B.op)
xo, yo, xi, yi = s[B].tile(B.op.axis[0], B.op.axis[1], x_factor=10, y_factor=5)
print(xo, yo, xi, yi)
fused = s[B].fuse(xi, yi)
print(tvm.lower(s, [A, B], simple_mode=True))

A = te.placeholder((m, n), name='A')
B = te.compute((m, n), lambda i,j : A[i, j], name='B')

s = te.create_schedule(B.op)
xo, yo, xi, yi = s[B].tile(B.op.axis[0], B.op.axis[1], x_factor=10, y_factor=5)
s[B].reorder(xi, yo, xo, yi)
print(tvm.lower(s, [A, B], simple_mode=True))

A = te.placeholder((n,), name="A")
B = te.compute(A.shape, lambda i: A[i] * 2, name="B")

s = te.create_schedule(B.op)
bx, tx = s[B].split(B.op.axis[0], factor=64)
s[B].bind(bx, te.thread_axis("blockIdx.x"))
s[B].bind(tx, te.thread_axis("threadIdx.x"))
print(tvm.lower(s, [A, B], simple_mode=True))

A = te.placeholder((m,), name="A")
B = te.compute((m,), lambda i: A[i] + 1, name="B")
C = te.compute((m,), lambda i: B[i] * 2, name="C")

s = te.create_schedule(C.op)
print(tvm.lower(s, [A, B, C], simple_mode=True))

A = te.placeholder((m,), name="A")
B = te.compute((m,), lambda i: A[i] + 1, name="B")
C = te.compute((m,), lambda i: B[i] * 2, name="C")

s = te.create_schedule(C.op)
s[B].compute_at(s[C], C.op.axis[0])
print(tvm.lower(s, [A, B, C], simple_mode=True))

A = te.placeholder((m, ), name='A')
B = te.compute((m, ), lambda i: A[i] + 1, name='B')
C = te.compute((m, ), lambda i: B[i] * 2, name='C')
D = te.compute((m, ), lambda i: C[i] - 3, name='D')

s = te.create_schedule(D.op)
s[C].compute_inline()
s[B].compute_inline()
print(tvm.lower(s, [A, B, C], simple_mode=True))

s = te.create_schedule(D.op)
s[B].compute_inline()
s[C].compute_inline()
print(tvm.lower(s, [A, B, C], simple_mode=True))

A = te.placeholder((m,), name="A")
B = te.compute((m,), lambda i: A[i] + 1, name="B")
C = te.compute((m,), lambda i: B[i] * 2, name="C")

s = te.create_schedule(C.op)
print(tvm.lower(s, [A, B, C], simple_mode=True))

A = te.placeholder((m,), name="A")
B = te.compute((m,), lambda i: A[i] + 1, name="B")
C = te.compute((m,), lambda i: B[i] * 2, name="C")

s = te.create_schedule(C.op)
s[B].compute_at(s[C], C.op.axis[0])
s[B].compute_root()
print(tvm.lower(s, [A, B, C], simple_mode=True))