import utils.sfm_reader
import numpy as np
import random
import pyrqt


dataset = 'grossmunster'
num_tuples = 1

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
        'max_reproj': 4.0
    }

    out = pyrqt.ransac_quadrifocal(x1c,x2c,x3c,x4c,opt)

    

    import ipdb
    ipdb.set_trace()