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
    u = np.array([0,1,0]) + 0.01*np.random.randn(3)    
    r1 = np.cross(u, r3)
    r1 = r1 / np.linalg.norm(r1)
    r2 = np.cross(r3, r1)
    R = np.c_[r1, r2, r3].T

    return R

def camera_error_modulo_flips(PP_est,PP_gt):
    err = 10000
    for z in [-1,1]:        
        H = np.diag([1,1,z,1])

        e1 = np.linalg.norm(PP_est[0] @ H - PP_gt[0][0:2,:])
        e2 = np.linalg.norm(PP_est[1] @ H - PP_gt[1][0:2,:])
        e3 = np.linalg.norm(PP_est[2] @ H - PP_gt[2][0:2,:])
        e4 = np.linalg.norm(PP_est[3] @ H - PP_gt[3][0:2,:])
        err = np.min([err, e1+e2+e3+e4])         
    return err
    

def setup_synthetic_scene():
    X = np.random.rand(13,3)
    X = 2*(X - 0.5)

    c1 = np.random.randn(3)
    c2 = np.random.randn(3)
    c3 = np.random.randn(3)
    c4 = np.random.randn(3)

    #c1 = 2.0 * c1 / np.linalg.norm(c1)
    #c2 = 2.0 * c2 / np.linalg.norm(c2)
    #c3 = 2.0 * c3 / np.linalg.norm(c3)
    #c4 = 2.0 * c4 / np.linalg.norm(c4)

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

    #x1 = x1[:,0:2] / x1[:,[2,2]]
    #x2 = x2[:,0:2] / x2[:,[2,2]]
    #x3 = x3[:,0:2] / x3[:,[2,2]]
    #x4 = x4[:,0:2] / x4[:,[2,2]]

    x1 = x1[:,0:2]
    x2 = x2[:,0:2]
    x3 = x3[:,0:2]
    x4 = x4[:,0:2]

    P1 = np.c_[R1, t1]
    P2 = np.c_[R2, t2]
    P3 = np.c_[R3, t3]
    P4 = np.c_[R4, t4]

    # transform coordinate system
    H = np.c_[R1.T, -R1.T @ t1]
    H = np.r_[H, np.array([[0,0,0,1]])]
    P1 = P1 @ H
    P2 = P2 @ H
    P3 = P3 @ H
    P4 = P4 @ H
    Hinv = np.linalg.inv(H)
    X = X @ Hinv[0:3,0:3].T + Hinv[0:3,3]

    # fix second camera translation
    alpha = -P2[1,3] / P2[1,2]
    H = np.c_[np.eye(3), np.array([0,0,alpha])]
    H = np.r_[H, np.array([[0,0,0,1]])]
    P1 = P1 @ H
    P2 = P2 @ H
    P3 = P3 @ H
    P4 = P4 @ H
    Hinv = np.linalg.inv(H)
    X = X @ Hinv[0:3,0:3].T + Hinv[0:3,3]

    
    # Fix scale
    sc = P2[0,3]
    if sc < 0:
        P2 = -P2
        sc = -sc
        x2 = -x2

    #P1[:,3] /= sc
    P2[:,3] /= sc
    P3[:,3] /= sc
    P4[:,3] /= sc
    X = X / sc
    
    xx = [x1,x2,x3,x4]
    PP = [P1[0:2,:],P2[0:2,:],P3[0:2,:],P4[0:2,:]]

    return (xx, PP, X)

def camera_rotation_error(PP_est,PP_gt):
    err = 10000
    for z in [-1,1]:
        H = np.diag([1,1,z,1])

        cur_err = 0
        for i in range(1,4):
            PPi = PP_est[i] @ H
            r1 = PPi[0,0:3]
            r2 = PPi[1,0:3]
            R = np.stack([r1, r2, np.cross(r1, r2)])
            if np.linalg.det(R) < 0:
                R = np.stack([r1, r2, -np.cross(r1, r2)])

            r1gt = PP_gt[i][0,0:3]
            r2gt = PP_gt[i][1,0:3]
            Rgt = np.stack([r1gt, r2gt, np.cross(r1gt, r2gt)])
            if np.linalg.det(Rgt) < 0:
                Rgt = np.stack([r1gt, r2gt, -np.cross(r1gt, r2gt)])


            Rdiff = Rgt.T @ R
            #print(R)
            #print(Rgt)
            cs = (Rdiff.trace()-1)/2
            if(cs > 1):
                cs = 1
            elif cs < -1:
                cs = -1
            #print(cs)
            #print()
            #print(Rdiff)
            #print((Rdiff.trace()-1)/2)
            cur_err = np.max([cur_err, np.arccos(cs)])
            #print((Rdiff.trace()-1)/2)
            #print(cur_err)
        #print(cur_err)


        err = np.min([err, cur_err])
    return err


def translation_dist(P1, P2):
        u1, s1, v1 = np.linalg.svd(P1)
        a1 = v1[2,0:3]/v1[2,3]
        a1_ = v1[3,0:3]/v1[3,3]
        b1 = a1_-a1
        b1 = b1/np.linalg.norm(b1)

        u2, s2, v2 = np.linalg.svd(P2)
        a2 = v2[2,0:3]/v2[2,3]
        a2_ = v2[3,0:3]/v2[3,3]
        b2 = a2_-a2
        b2 = b2/np.linalg.norm(b2)

        cr = np.cross(b1,b2)
        cr = cr/np.linalg.norm(cr)

        return np.dot(cr,(a2-a1))

def camera_translation_error(PP_est,PP_gt):
    err = 10000
    for z in [-1,1]:
        H = np.diag([1,1,z,1])

        d2 = translation_dist(PP_est[1], PP_gt[1])
        d3 = translation_dist(PP_est[2], PP_gt[2])
        d4 = translation_dist(PP_est[3], PP_gt[3])
        c_err = np.max(np.abs([d2,d3,d4]))

        err = np.min(np.abs([err, c_err]))
    return err



for x in range(100000):
    xx, PP_gt, X = setup_synthetic_scene()

    T_gt = make_tensor(PP_gt[0], PP_gt[1], PP_gt[2], PP_gt[3])

    out = pyrqt.calibrated_radial_quadrifocal_solver(xx[0], xx[1], xx[2], xx[3], {"solver": "NANSON2"})
    err_T = [np.min([np.linalg.norm(T - T_gt), np.linalg.norm(T + T_gt)]) for T in out['QFs']]


    err_P = []
    err_R = []
    err_T = []

    for i in range(out['valid']):
        P1 = out['P1'][i]
        P2 = out['P2'][i]
        P3 = out['P3'][i]
        P4 = out['P4'][i]

        err_P.append(camera_error_modulo_flips([P1,P2,P3,P4], PP_gt))
        err_R.append(camera_rotation_error([P1,P2,P3,P4], PP_gt))
        err_T.append(camera_translation_error([P1,P2,P3,P4], PP_gt))

    if(len(err_P)>0):
        print(str(min(err_P))+" "+str(min(err_R))+" "+str(min(err_T)))
    else:
        print("1 3.14 3.14")




## Validate the synthetic instance
#eps = np.array([[0, -1], [1, 0]])
#for k in range(4):
#    proj = (PP[k][0:2,0:3] @ X.T).T + PP[k][0:2,3]
#
#    for i in range(13):
#        err = proj[i] @ eps @ xx[k][i].T
#        infront = np.dot(proj[i], xx[k][i]) > 0
#        print(err, infront)

