import numpy as np
import ot

a = np.ones(10)/10
b = np.ones(12)/12
C = np.random.rand(10, 12)

val = ot.unbalanced.sinkhorn_unbalanced2(a, b, C, reg=0.01, reg_m=0.1)
print("Unbalanced OT output:", val)

pi = ot.unbalanced.sinkhorn_unbalanced(a, b, C, reg=0.01, reg_m=0.1)
print("Pi sum:", pi.sum())
