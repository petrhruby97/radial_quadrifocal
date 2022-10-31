import utils.sfm_reader
import numpy as np
import random
import pyrqt

def compute_reprojection_error(xx, PP, X):
    repr_all = np.zeros((xx[0].shape[0], 0))
    for i in range(4):
        #print(X)
        z = X @ PP[i][0:2,0:3].T + PP[i][:,3].T
        nz = np.sqrt(z[:,0]*z[:,0] + z[:,1]*z[:,1])
        z /= np.c_[nz, nz]

        alpha = np.sum(xx[i] * z, axis=1)
        z_proj = z * np.c_[alpha, alpha]
        res = xx[i] - z_proj

        repr_all = np.c_[repr_all, np.sqrt(res[:,0]*res[:,0] + res[:,1]*res[:,1])]

    return repr_all

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

            Rgt = PP_gt[i][:,0:3]

            Rdiff = Rgt.T @ R
            cs = (Rdiff.trace()-1)/2
            if(cs > 1):
                cs = 1
            elif cs < -1:
                cs = -1
            cur_err = np.max([cur_err, np.arccos(cs)])

        err = np.min([err, cur_err])
    return err

def translation_dist(P1, P2):
        u1, s1, v1 = np.linalg.svd(P1)
        a1 = v1[2,0:3]/v1[2,3]
        a1_ = v1[3,0:3]/v1[3,3]
        b1 = a1_-a1
        b1 = b1/np.linalg.norm(b1)

        if(v1[2,3] < 1e-10) and (v1[2,3] > -1e-10):
            a1 = a1_
            b1 = v1[2,0:3]

        u2, s2, v2 = np.linalg.svd(P2)
        a2 = v2[2,0:3]/v2[2,3]
        a2_ = v2[3,0:3]/v2[3,3]
        b2 = a2_-a2
        b2 = b2/np.linalg.norm(b2)

        if(v2[2,3] < 1e-10) and (v2[2,3] > -1e-10):
            a2 = a2_
            b2 = v2[2,0:3]

        #print(a2)
        #print(b2)

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



dataset = 'grossmunster'
#dataset = 'kirchenge'
#dataset = 'pipes'
num_tuples = 20
min_shared_pts = 108

matches, poses, camera_dict = utils.sfm_reader.load_tuples(f'../data/{dataset}', num_tuples,
                                                              min_shared_pts=min_shared_pts,
                                                              verified_matches=False, uniform_images=False,
                                                              cycle_consistency=True)


pp = camera_dict['params'][[2,3]]
if dataset=='pipes':
    pp = camera_dict['params'][[1,2]]
#print(pp)

#STATISTICS
num_found_poses_13 = 0
avg_rep_err_13 = 0
avg_rot_err_13 = 0
avg_tran_err_13 = 0
avg_succ_rep_err_13 = 0
avg_succ_rot_err_13 = 0
avg_succ_tran_err_13 = 0
avg_succ_cam_err_13 = 0
AUC5_13 = 0
AUC10_13 = 0
AUC20_13 = 0
AUC_T1_13 = 0
AUC_T5_13 = 0
AUC_T10_13 = 0

num_found_poses_N = 0
avg_rep_err_N = 0
avg_rot_err_N = 0
avg_tran_err_N = 0
avg_succ_rep_err_N = 0
avg_succ_rot_err_N = 0
avg_succ_tran_err_N = 0
avg_succ_cam_err_N = 0
AUC5_N = 0
AUC10_N = 0
AUC20_N = 0
AUC_T1_N = 0
AUC_T5_N = 0
AUC_T10_N = 0

num_found_poses_N2 = 0
avg_rep_err_N2 = 0
avg_rot_err_N2 = 0
avg_tran_err_N2 = 0
avg_succ_rep_err_N2 = 0
avg_succ_rot_err_N2 = 0
avg_succ_tran_err_N2 = 0
avg_succ_cam_err_N2 = 0
AUC5_N2 = 0
AUC10_N2 = 0
AUC20_N2 = 0
AUC_T1_N2 = 0
AUC_T5_N2 = 0
AUC_T10_N2 = 0

num_found_poses_15 = 0
avg_rep_err_15 = 0
avg_rot_err_15 = 0
avg_tran_err_15 = 0
avg_succ_rep_err_15 = 0
avg_succ_rot_err_15 = 0
avg_succ_tran_err_15 = 0
avg_succ_cam_err_15 = 0
AUC5_15 = 0
AUC10_15 = 0
AUC20_15 = 0
AUC_T1_15 = 0
AUC_T5_15 = 0
AUC_T10_15 = 0

num_found_poses_7 = 0
avg_rep_err_7 = 0
avg_rot_err_7 = 0
avg_tran_err_7 = 0
avg_succ_rep_err_7 = 0
avg_succ_rot_err_7 = 0
avg_succ_tran_err_7 = 0
avg_succ_cam_err_7 = 0
AUC5_7 = 0
AUC10_7 = 0
AUC20_7 = 0
AUC_T1_7 = 0
AUC_T5_7 = 0
AUC_T10_7 = 0
pose_id = 0
for ((x1,x2,x3,x4), (P1,P2,P3,P4)) in zip(matches,poses):
    print(pose_id)
    pose_id += 1

    n = len(x1)    

    x1c = x1 - pp
    x2c = x2 - pp
    x3c = x3 - pp
    x4c = x4 - pp

    #do some stuff with GT pose
    P1r = P1[0:2,:]
    P2r = P2[0:2,:]
    P3r = P3[0:2,:]
    P4r = P4[0:2,:]
    PPgt = [P1r, P2r, P3r, P4r]
    dct = pyrqt.triangulate(x1c, x2c, x3c, x4c, P1r, P2r, P3r, P4r)
    
    
    opt = {
        'min_iterations': 10,
        'max_iterations': 100,
        'max_reproj': 2.0,
        'solver': 'MINIMAL'
    }

    out = pyrqt.ransac_quadrifocal(x1c,x2c,x3c,x4c,opt)

    if out['ransac']['num_inliers'] == 0:
        print(f'OUR: Found no reconstruction')
        print()
        avg_rep_err_13 += 50
        avg_rot_err_13 += 3.14
        avg_tran_err_13 += 3.14

    else:
        xx = [x1c,x2c,x3c,x4c]
        PP = [out['P1'],out['P2'],out['P3'],out['P4']]
        Ps = [out['P1'], out['P2'], out['P3'], out['P4']]
        Psgt = [P1, P2, P3, P4]

        num_found_poses_13 += 1
        rep_err = compute_reprojection_error(xx,PP,out['X'])
        repr_err = np.mean(rep_err[out['inliers'],:])
        rot_err = 180*camera_rotation_error(Ps, Psgt)/3.141592654
        tran_err = camera_translation_error(Ps, Psgt)
        avg_rep_err_13 += repr_err
        avg_rot_err_13 += rot_err
        avg_tran_err_13 += tran_err
        avg_succ_rep_err_13 += repr_err
        avg_succ_rot_err_13 += rot_err
        avg_succ_tran_err_13 += tran_err
        cam_err = camera_error_modulo_flips(Ps, Psgt)
        avg_succ_cam_err_13 += cam_err
        if(rot_err < 5):
            AUC5_13 += 1
        if(rot_err < 10):
            AUC10_13 += 1
        if(rot_err < 20):
            AUC20_13 += 1
        if(tran_err < 0.5):
            AUC_T1_13 += 1
        if(tran_err < 1.0):
            AUC_T5_13 += 1
        if(tran_err < 5.0):
            AUC_T10_13 += 1

        print("Cam err: "+str(cam_err))

        repr = compute_reprojection_error(xx,PP,out['X'])
        repr_GT = compute_reprojection_error(xx,PPgt,dct['Xs'])
        repr_GT_avg = np.mean(repr_GT[out['inliers'],:])
        print("GT repr. err.: "+str(repr_GT_avg))
        print(f'OUR: Found reconstruction with {out["ransac"]["num_inliers"]}/{len(x1)} inliers with {repr_err}px mean reprojection error.')
        print(f'{out["ransac"]["iterations"]} iterations, {out["ransac"]["refinements"]} refinements.')
        print(camera_rotation_error(Ps, Psgt))
        print("Translation "+str(camera_translation_error(Ps, Psgt)))
        print()
        
    n_opt = {
        'min_iterations': 10,
        'max_iterations': 100,
        'max_reproj': 2.0,
        'solver': 'NANSON'
    }

    n_out = pyrqt.ransac_quadrifocal(x1c,x2c,x3c,x4c,n_opt)

    if n_out['ransac']['num_inliers'] == 0:
        print(f'NANSON: Found no reconstruction')
        print()
        avg_rep_err_N += 50
        avg_rot_err_N += 3.14
        avg_tran_err_N += 3.14

    else:
        xx = [x1c,x2c,x3c,x4c]
        PP = [n_out['P1'],n_out['P2'],n_out['P3'],n_out['P4']]
        Ps = [n_out['P1'], n_out['P2'], n_out['P3'], n_out['P4']]
        Psgt = [P1, P2, P3, P4]

        num_found_poses_N += 1
        rep_err = compute_reprojection_error(xx,PP,n_out['X'])
        repr_err = np.mean(rep_err[n_out['inliers'],:])
        rot_err = 180*camera_rotation_error(Ps, Psgt)/3.141592654
        tran_err = camera_translation_error(Ps, Psgt)
        avg_rep_err_N += repr_err
        avg_rot_err_N += rot_err
        avg_tran_err_N += tran_err
        avg_succ_rep_err_N += repr_err
        avg_succ_rot_err_N += rot_err
        avg_succ_tran_err_N += tran_err
        cam_err = camera_error_modulo_flips(Ps, Psgt)
        avg_succ_cam_err_N += cam_err
        if(rot_err < 5):
            AUC5_N += 1
        if(rot_err < 10):
            AUC10_N += 1
        if(rot_err < 20):
            AUC20_N += 1
        if(tran_err < 0.5):
            AUC_T1_N += 1
        if(tran_err < 1.0):
            AUC_T5_N += 1
        if(tran_err < 5.0):
            AUC_T10_N += 1

        print("Cam err: "+str(cam_err))

        repr = compute_reprojection_error(xx,PP,n_out['X'])
        repr_GT = compute_reprojection_error(xx,PPgt,dct['Xs'])
        repr_GT_avg = np.mean(repr_GT[n_out['inliers'],:])
        print("GT repr. err.: "+str(repr_GT_avg))
        print(f'NANSON: Found reconstruction with {n_out["ransac"]["num_inliers"]}/{len(x1)} inliers with {repr_err}px mean reprojection error.')
        print(f'{n_out["ransac"]["iterations"]} iterations, {n_out["ransac"]["refinements"]} refinements.')
        print(camera_rotation_error(Ps, Psgt))
        print("Translation "+str(camera_translation_error(Ps, Psgt)))

    n2_opt = {
        'min_iterations': 10,
        'max_iterations': 100,
        'max_reproj': 2.0,
        'solver': 'NANSON2'
    }

    n2_out = pyrqt.ransac_quadrifocal(x1c,x2c,x3c,x4c,n2_opt)

    if n2_out['ransac']['num_inliers'] == 0:
        print(f'NANSON2: Found no reconstruction')
        print()
        avg_rep_err_N2 += 50
        avg_rot_err_N2 += 3.14
        avg_tran_err_N2 += 3.14

    else:
        xx = [x1c,x2c,x3c,x4c]
        PP = [n2_out['P1'],n2_out['P2'],n2_out['P3'],n2_out['P4']]
        Ps = [n2_out['P1'], n2_out['P2'], n2_out['P3'], n2_out['P4']]
        Psgt = [P1, P2, P3, P4]

        num_found_poses_N2 += 1
        rep_err = compute_reprojection_error(xx,PP,n2_out['X'])
        repr_err = np.mean(rep_err[n2_out['inliers'],:])
        rot_err = 180*camera_rotation_error(Ps, Psgt)/3.141592654
        tran_err = camera_translation_error(Ps, Psgt)
        avg_rep_err_N2 += repr_err
        avg_rot_err_N2 += rot_err
        avg_tran_err_N2 += tran_err
        avg_succ_rep_err_N2 += repr_err
        avg_succ_rot_err_N2 += rot_err
        avg_succ_tran_err_N2 += tran_err
        cam_err = camera_error_modulo_flips(Ps, Psgt)
        avg_succ_cam_err_N2 += cam_err
        if(rot_err < 5):
            AUC5_N2 += 1
        if(rot_err < 10):
            AUC10_N2 += 1
        if(rot_err < 20):
            AUC20_N2 += 1
        if(tran_err < 0.5):
            AUC_T1_N2 += 1
        if(tran_err < 1.0):
            AUC_T5_N2 += 1
        if(tran_err < 5.0):
            AUC_T10_N2 += 1

        print("Cam err: "+str(cam_err))

        repr = compute_reprojection_error(xx,PP,n2_out['X'])
        repr_GT = compute_reprojection_error(xx,PPgt,dct['Xs'])
        repr_GT_avg = np.mean(repr_GT[n2_out['inliers'],:])
        print("GT repr. err.: "+str(repr_GT_avg))
        print(f'NANSON2: Found reconstruction with {n2_out["ransac"]["num_inliers"]}/{len(x1)} inliers with {repr_err}px mean reprojection error.')
        print(f'{n2_out["ransac"]["iterations"]} iterations, {n2_out["ransac"]["refinements"]} refinements.')
        print(camera_rotation_error(Ps, Psgt))
        print("Translation "+str(camera_translation_error(Ps, Psgt)))

    l_opt = {
        'min_iterations': 10,
        'max_iterations': 100,
        'max_reproj': 2.0,
        'solver': 'LINEAR'
    }

    l_out = pyrqt.ransac_quadrifocal(x1c,x2c,x3c,x4c,l_opt)

    if l_out['ransac']['num_inliers'] == 0:
        print(f'LINEAR: Found no reconstruction')
        print()
        avg_rep_err_15 += 50
        avg_rot_err_15 += 3.14
        avg_tran_err_15 += 3.14

    else:
        xx = [x1c,x2c,x3c,x4c]
        PP = [l_out['P1'],l_out['P2'],l_out['P3'],l_out['P4']]
        Psgt = [P1, P2, P3, P4]
        Ps = PP

        num_found_poses_15 += 1
        rep_err = compute_reprojection_error(xx,PP,l_out['X'])
        repr_err = np.mean(rep_err[l_out['inliers'],:])
        rot_err = 180*camera_rotation_error(Ps, Psgt)/3.141592654
        tran_err = camera_translation_error(Ps, Psgt)
        avg_rep_err_15 += repr_err
        avg_rot_err_15 += rot_err
        avg_tran_err_15 += tran_err
        avg_succ_rep_err_15 += repr_err
        avg_succ_rot_err_15 += rot_err
        avg_succ_tran_err_15 += tran_err
        cam_err = camera_error_modulo_flips(Ps, Psgt)
        avg_succ_cam_err_15 += cam_err
        if(rot_err < 5):
            AUC5_15 += 1
        if(rot_err < 10):
            AUC10_15 += 1
        if(rot_err < 20):
            AUC20_15 += 1
        if(tran_err < 0.5):
            AUC_T1_15 += 1
        if(tran_err < 1):
            AUC_T5_15 += 1
        if(tran_err < 5):
            AUC_T10_15 += 1

        repr = compute_reprojection_error(xx,PP,l_out['X'])
        print(f'LINEAR: Found reconstruction with {l_out["ransac"]["num_inliers"]}/{len(x1)} inliers with {repr_err}px mean reprojection error.')
        print(f'{l_out["ransac"]["iterations"]} iterations, {l_out["ransac"]["refinements"]} refinements.')
        print(camera_rotation_error(PP, Psgt))
        print()

    u_opt = {
        'min_iterations': 10,
        'max_iterations': 100,
        'max_reproj': 2.0,
        'solver': 'UPRIGHT'
    }

    u_out = pyrqt.ransac_quadrifocal(x1c,x2c,x3c,x4c,u_opt)

    if u_out['ransac']['num_inliers'] == 0:
        print(f'UPRIGHT: Found no reconstruction')
        avg_rep_err_7 += 50
        avg_rot_err_7 += 3.14
        avg_tran_err_7 += 3.14
        print()
        print()
        print()

    else:
        xx = [x1c,x2c,x3c,x4c]
        PP = [u_out['P1'],u_out['P2'],u_out['P3'],u_out['P4']]
        Psgt = [P1, P2, P3, P4]
        Ps = PP

        num_found_poses_7 += 1
        rep_err = compute_reprojection_error(xx,PP,u_out['X'])
        repr_err = np.mean(rep_err[u_out['inliers'],:])
        rot_err = 180*camera_rotation_error(Ps, Psgt)/3.141592654
        tran_err = camera_translation_error(Ps, Psgt)
        avg_rep_err_7 += repr_err
        avg_rot_err_7 += rot_err
        avg_tran_err_7 += tran_err
        avg_succ_rep_err_7 += repr_err
        avg_succ_rot_err_7 += rot_err
        avg_succ_tran_err_7 += tran_err
        cam_err = camera_error_modulo_flips(Ps, Psgt)
        avg_succ_cam_err_7 += cam_err
        if(rot_err < 5):
            AUC5_7 += 1
        if(rot_err < 10):
            AUC10_7 += 1
        if(rot_err < 20):
            AUC20_7 += 1
        if(tran_err < 0.5):
            AUC_T1_7 += 1
        if(tran_err < 1.0):
            AUC_T5_7 += 1
        if(tran_err < 5.0):
            AUC_T10_7 += 1

        repr = compute_reprojection_error(xx,PP,u_out['X'])
        print(f'UPRIGHT: Found reconstruction with {u_out["ransac"]["num_inliers"]}/{len(x1)} inliers with {repr_err}px mean reprojection error.')
        print(f'{u_out["ransac"]["iterations"]} iterations, {u_out["ransac"]["refinements"]} refinements.')
        print(camera_rotation_error(PP, Psgt))
        print()
        print()
        print()

print("13 solver")
if(num_found_poses_13 > 0):
    print(str(num_found_poses_13/num_tuples))
    print("Old errors: "+str(num_found_poses_13/num_tuples)+" "+str(avg_succ_rep_err_13/num_found_poses_13)+" "+str(avg_succ_rot_err_13/num_found_poses_13)+" "+str(avg_succ_tran_err_13/num_found_poses_13))
    print("Cam error: "+str(avg_succ_cam_err_13/num_found_poses_13))
    print("Tran error: "+str(avg_succ_tran_err_13/num_found_poses_13))
    print("AUC: "+str(AUC5_13/num_tuples)+" "+str(AUC10_13/num_tuples)+" "+str(AUC20_13/num_tuples)+" | "+str(AUC_T1_13/num_tuples)+" "+str(AUC_T5_13/num_tuples)+" "+str(AUC_T10_13/num_tuples)+" ")
    print()
else:
    print(str(num_found_poses_13/num_tuples)+" "+str(50)+" "+str(180)+" "+str(100)+" "+str(0)+" "+str(0)+" "+str(0)+" "+str(0)+" "+str(0)+" "+str(0))

print("13 solver NANSON")
if(num_found_poses_N > 0):
    print(str(num_found_poses_N/num_tuples))
    print("Old errors: "+str(num_found_poses_N/num_tuples)+" "+str(avg_succ_rep_err_N/num_found_poses_N)+" "+str(avg_succ_rot_err_N/num_found_poses_N)+" "+str(avg_succ_tran_err_N/num_found_poses_N))
    print("Cam error: "+str(avg_succ_cam_err_N/num_found_poses_N))
    print("Tran error: "+str(avg_succ_tran_err_N/num_found_poses_N))
    print("AUC: "+str(AUC5_N/num_tuples)+" "+str(AUC10_N/num_tuples)+" "+str(AUC20_N/num_tuples)+" | "+str(AUC_T1_N/num_tuples)+" "+str(AUC_T5_N/num_tuples)+" "+str(AUC_T10_N/num_tuples)+" ")
    print()
else:
    print(str(num_found_poses_N/num_tuples)+" "+str(50)+" "+str(180)+" "+str(100)+" "+str(0)+" "+str(0)+" "+str(0)+" "+str(0)+" "+str(0)+" "+str(0))

print("13 solver NANSON2")
if(num_found_poses_N2 > 0):
    print(str(num_found_poses_N2/num_tuples))
    print("Old errors: "+str(num_found_poses_N2/num_tuples)+" "+str(avg_succ_rep_err_N2/num_found_poses_N2)+" "+str(avg_succ_rot_err_N2/num_found_poses_N2)+" "+str(avg_succ_tran_err_N2/num_found_poses_N2))
    print("Cam error: "+str(avg_succ_cam_err_N2/num_found_poses_N2))
    print("Tran error: "+str(avg_succ_tran_err_N2/num_found_poses_N2))
    print("AUC: "+str(AUC5_N2/num_tuples)+" "+str(AUC10_N2/num_tuples)+" "+str(AUC20_N2/num_tuples)+" | "+str(AUC_T1_N2/num_tuples)+" "+str(AUC_T5_N2/num_tuples)+" "+str(AUC_T10_N2/num_tuples)+" ")
    print()
else:
    print(str(num_found_poses_N2/num_tuples)+" "+str(50)+" "+str(180)+" "+str(100)+" "+str(0)+" "+str(0)+" "+str(0)+" "+str(0)+" "+str(0)+" "+str(0))

print("15 solver")
if(num_found_poses_15 > 0):
    print(str(num_found_poses_15/num_tuples))
    print("Old errors: "+str(num_found_poses_15/num_tuples)+" "+str(avg_succ_rep_err_15/num_found_poses_15)+" "+str(avg_succ_rot_err_15/num_found_poses_15)+" "+str(avg_succ_tran_err_15/num_found_poses_15))
    print("Cam error: "+str(avg_succ_cam_err_15/num_found_poses_15))
    print("Tran error: "+str(avg_succ_tran_err_15/num_found_poses_15))
    print("AUC: "+str(AUC5_15/num_tuples)+" "+str(AUC10_15/num_tuples)+" "+str(AUC20_15/num_tuples)+" | "+str(AUC_T1_15/num_tuples)+" "+str(AUC_T5_15/num_tuples)+" "+str(AUC_T10_15/num_tuples)+" ")
    #print(str(num_found_poses_15/num_tuples)+" "+str(avg_succ_rep_err_15/num_found_poses_15)+" "+str(avg_succ_rot_err_15/num_found_poses_15)+" "+str(avg_succ_tran_err_15/num_found_poses_15)+" "+str(AUC5_15/num_tuples)+" "+str(AUC10_15/num_tuples)+" "+str(AUC20_15/num_tuples)+" "+str(AUC_T1_15/num_tuples)+" "+str(AUC_T5_15/num_tuples)+" "+str(AUC_T10_15/num_tuples))
    print()
else:
    print(str(num_found_poses_15/num_tuples)+" "+str(50)+" "+str(180)+" "+str(100)+" "+str(0)+" "+str(0)+" "+str(0)+" "+str(0)+" "+str(0)+" "+str(0))

print("7 solver")
if(num_found_poses_7 > 0):
    print(str(num_found_poses_7/num_tuples))
    print("Old errors: "+str(num_found_poses_7/num_tuples)+" "+str(avg_succ_rep_err_7/num_found_poses_7)+" "+str(avg_succ_rot_err_7/num_found_poses_7)+" "+str(avg_succ_tran_err_7/num_found_poses_7))
    print("Cam error: "+str(avg_succ_cam_err_7/num_found_poses_7))
    print("Tran error: "+str(avg_succ_tran_err_7/num_found_poses_7))
    print("AUC: "+str(AUC5_7/num_tuples)+" "+str(AUC10_7/num_tuples)+" "+str(AUC20_7/num_tuples)+" | "+str(AUC_T1_7/num_tuples)+" "+str(AUC_T5_7/num_tuples)+" "+str(AUC_T10_7/num_tuples)+" ")
    #print(str(num_found_poses_7/num_tuples)+" "+str(avg_succ_rep_err_7/num_found_poses_7)+" "+str(avg_succ_rot_err_7/num_found_poses_7)+" "+str(avg_succ_tran_err_7/num_found_poses_7)+" "+str(AUC5_7/num_tuples)+" "+str(AUC10_7/num_tuples)+" "+str(AUC20_7/num_tuples)+" "+str(AUC_T1_7/num_tuples)+" "+str(AUC_T5_7/num_tuples)+" "+str(AUC_T10_7/num_tuples))
else:
    print(str(num_found_poses_7/num_tuples)+" "+str(50)+" "+str(180)+" "+str(100)+" "+str(0)+" "+str(0)+" "+str(0)+" "+str(0)+" "+str(0)+" "+str(0))

