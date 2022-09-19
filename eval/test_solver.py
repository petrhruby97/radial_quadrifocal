import numpy as np
import pyrqt

def make_tensor(P1,P2,P3,P4):
    T = np.zeros(16)
    for a in range(16):
        ix4 = a % 2
        ix3 = int(a / 2) % 2
        ix2 = int(a / 4) % 2
        ix1 = int(a / 8) % 2

        M = np.c_[P1[ix1,:], P2[ix2,:], P3[ix3,:], P4[ix4,:]]        
        sign = (ix1 + ix2 + ix3 + ix4) % 2
        if sign > 0:
            T[a] = -np.linalg.det(M)
        else:
            T[a] = np.linalg.det(M)
        
    T = T / np.linalg.norm(T)
    return T


def random_rot():
    A = np.random.randn(3,3)
    U,_,_ = np.linalg.svd(A)
    if np.linalg.det(U) < 0:
        U = -U
    return U

def lookat(point, center):
    r3 = point - center
    r3 = r3 / np.linalg.norm(r3)

    u = np.array([0,1,0])
    r1 = np.cross(u, r3)
    r1 = r1 / np.linalg.norm(r1)
    r2 = np.cross(r3, r1)
    R = np.c_[r1, r2, r3].T

    return R

    

def setup_synthetic_scene():
    X = np.random.rand(13,3)
    X = 2*(X - 0.5)

    

    c1 = np.random.randn(3)
    c2 = np.random.randn(3)
    c3 = np.random.randn(3)
    c4 = np.random.randn(3)

    c1 = 2.0 * c1 / np.linalg.norm(c1)
    c2 = 2.0 * c2 / np.linalg.norm(c2)
    c3 = 2.0 * c3 / np.linalg.norm(c3)
    c4 = 2.0 * c4 / np.linalg.norm(c4)

    R1 = lookat(2.0 * (np.random.rand(3) - 0.5), c1)
    R2 = lookat(2.0 * (np.random.rand(3) - 0.5), c2)
    R3 = lookat(2.0 * (np.random.rand(3) - 0.5), c3)
    R4 = lookat(2.0 * (np.random.rand(3) - 0.5), c4)

    t1 = -R1 @ c1
    t2 = -R2 @ c2
    t3 = -R3 @ c3
    t4 = -R4 @ c4

    x1 = (X @ R1.T + t1)
    x2 = (X @ R2.T + t2)
    x3 = (X @ R3.T + t3)
    x4 = (X @ R4.T + t4)

    x1 = x1[:,0:2] / x1[:,[2,2]]
    x2 = x2[:,0:2] / x2[:,[2,2]]
    x3 = x3[:,0:2] / x3[:,[2,2]]
    x4 = x4[:,0:2] / x4[:,[2,2]]

    P1 = np.c_[R1, t1]
    P2 = np.c_[R2, t2]
    P3 = np.c_[R3, t3]
    P4 = np.c_[R4, t4]

    xx = [x1,x2,x3,x4]
    PP = [P1,P2,P3,P4]
    
    return (xx, PP, X)


xx, PP, X = setup_synthetic_scene()


T_gt = make_tensor(PP[0], PP[1], PP[2], PP[3])

out = pyrqt.radial_quadrifocal_solver(xx[0], xx[1], xx[2], xx[3], {})

for T in out['QFs']:
    err = np.min([np.linalg.norm(T - T_gt), np.linalg.norm(T + T_gt)])
    print(err)


# Validate the synthetic instance
#eps = np.array([[0, -1], [1, 0]])
#for k in range(4):
#    proj = (PP[k][0:2,0:3] @ X.T).T + PP[k][0:2,3]
#
#    for i in range(13):
#        err = proj[i] @ eps @ xx[k][i].T
#        infront = np.dot(proj[i], xx[k][i]) > 0
#        print(err, infront)
#
