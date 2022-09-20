import utils.sfm_reader
import numpy as np
import random
import pyrqt


dataset = 'grossmunster'
num_tuples = 20

matches, poses, camera_dict = utils.sfm_reader.load_tuples(f'../data/{dataset}', num_tuples,
                                                              min_shared_pts=200,
                                                              verified_matches=True, uniform_images=True,
                                                              cycle_consistency=True)


pp = camera_dict['params'][[2,3]]

for ((x1,x2,x3,x4), (P1,P2,P3,P4)) in zip(matches,poses):

    n = len(x1)
    ind = np.random.choice(range(n), 13, replace=False)

    x1s = x1[ind] - pp
    x2s = x2[ind] - pp
    x3s = x3[ind] - pp
    x4s = x4[ind] - pp
    
    out = pyrqt.calibrated_radial_quadrifocal_solver(x1s,x2s,x3s,x4s,{})

    print(f'GT = \n{P2}')
    for k in range(out['valid']):
        print(out['P2'][k])

    import ipdb
    ipdb.set_trace()