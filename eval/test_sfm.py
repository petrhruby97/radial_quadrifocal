import utils.sfm_reader
import numpy as np
import random
import pyrqt

def compute_reprojection_error(xx, PP, X):
    repr_all = np.zeros((xx[0].shape[0], 0))
    for i in range(4):
        z = X @ PP[i][0:2,0:3].T + PP[i][:,3].T
        nz = np.sqrt(z[:,0]*z[:,0] + z[:,1]*z[:,1])
        z /= np.c_[nz, nz]

        alpha = np.sum(xx[i] * z, axis=1)
        z_proj = z * np.c_[alpha, alpha]
        res = xx[i] - z_proj

        repr_all = np.c_[repr_all, np.sqrt(res[:,0]*res[:,0] + res[:,1]*res[:,1])]

    return repr_all


dataset = 'grossmunster'
num_tuples = 20

matches, poses, camera_dict = utils.sfm_reader.load_tuples(f'../data/{dataset}', num_tuples,
                                                              min_shared_pts=200,
                                                              verified_matches=True, uniform_images=False,
                                                              cycle_consistency=True)


pp = camera_dict['params'][[2,3]]

for ((x1,x2,x3,x4), (P1,P2,P3,P4)) in zip(matches,poses):

    n = len(x1)    

    x1c = x1 - pp
    x2c = x2 - pp
    x3c = x3 - pp
    x4c = x4 - pp

    
    
    opt = {
        'min_iterations': 10,
        'max_iterations': 100,
        'max_reproj': 2.0
    }

    out = pyrqt.ransac_quadrifocal(x1c,x2c,x3c,x4c,opt)

    if out['ransac']['num_inliers'] == 0:
        print(f'Found no reconstruction')

    else:
        xx = [x1c,x2c,x3c,x4c]
        PP = [out['P1'],out['P2'],out['P3'],out['P4']]
        repr = compute_reprojection_error(xx,PP,out['X'])
        repr_avg = np.mean(repr[out['inliers'],:])
        print(f'Found reconstruction with {out["ransac"]["num_inliers"]}/{len(x1)} inliers with {repr_avg}px mean reprojection error.')
