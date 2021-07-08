from __future__ import absolute_import, print_function

import tvm
import tvm.testing
from tvm import te
import numpy as np

''' equivalent C code :
for (int i = 0; i < n; ++i) {
  B[i] = 0;
  for (int k = 0; k < m; ++k) {
    B[i] = B[i] + A[i][k];
  }
}
'''

n = te.var('n')
m = te.var('m')
A = te.placeholder((n, m), name='A')
k = te.reduce_axis((0, n), 'k')
B = te.compute((n,), lambda i: te.sum(A[i, k], axis=k), name='B')

s = te.create_schedule(B.op)
print(tvm.lower(s, [A, B], simple_mode=True))

ko, ki = s[B].split(B.op.reduce_axis[0], factor=16)
xo, xi = s[B].split(B.op.axis[0], factor=32)
print(tvm.lower(s, [A, B], simple_mode=True))

## gpu specific kernel
s[B].bind(xo, te.thread_axis("blockIdx.x"))
s[B].bind(xi, te.thread_axis("threadIdx.x"))
print(tvm.lower(s, [A, B], simple_mode=True))

s = te.create_schedule(B.op)
ko, ki = s[B].split(B.op.reduce_axis[0], factor=16)
BF = s.rfactor(B, ki)
print(tvm.lower(s, [A, B], simple_mode=True))

print(s[B].op.body)

## Cross Thread Reduction
xo, xi = s[B].split(s[B].op.axis[0], factor=32)
s[B].bind(xo, te.thread_axis('blockIdx.x'))
s[B].bind(xi, te.thread_axis('threadIdx.y'))
tx = te.thread_axis('threadIdx.x')
s[B].bind(s[B].op.reduce_axis[0], tx)
s[BF].compute_at(s[B], s[B].op.reduce_axis[0])
s[B].set_store_predicate(tx.var.equal(0))
fcuda = tvm.build(s, [A, B], 'cuda')
print(fcuda.imported_modules[0].get_source())

nn = 128
dev = tvm.cuda(0)
a = tvm.nd.array(np.random.uniform(size=(nn, nn)).astype(A.dtype), dev)
b = tvm.nd.array(np.zeros(nn, dtype=B.dtype), dev)
fcuda(a, b)
tvm.testing.assert_allclose(b.numpy(), np.sum(a.numpy(), axis=1), rtol=1e-4)

#2d convolution example
n = te.var("n")
Input = te.placeholder((n, n), name="Input")
Filter = te.placeholder((3, 3), name="Filter")
di = te.reduce_axis((0, 3), name="di")
dj = te.reduce_axis((0, 3), name="dj")
Output = te.compute(
    (n - 2, n - 2),
    lambda i, j: te.sum(Input[i + di, j + dj] * Filter[di, dj], axis=[di, dj]),
    name="Output",
)
s = te.create_schedule(Output.op)
print(tvm.lower(s, [Input, Filter, Output], simple_mode=True))

n = te.var("n")
m = te.var("m")
product = te.comm_reducer(lambda x, y: x * y, lambda t: tvm.tir.const(1, dtype=t), name="product")
A = te.placeholder((n, m), name="A")
k = te.reduce_axis((0, m), name="k")
B = te.compute((n,), lambda i: product(A[i, k], axis=k), name="B")