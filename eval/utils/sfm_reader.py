import sys

from . import read_write_model
import sqlite3
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
#import pytsamp

def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % 2147483647
    image_id1 = (pair_id - image_id2) / 2147483647
    return image_id1, image_id2

def image_ids_to_pair(image_id1, image_id2):
    pair_id = image_id2 + 2147483647 * image_id1
    return pair_id


def extract_keyp(cursor, im):
    im = int(im)    
    cursor.execute("SELECT * FROM keypoints WHERE image_id = ?;", (im,))
    fetched = cursor.fetchone()
    if fetched is None:
        print(f'Strange, no image1 found for id={im}')
        return None
    image_idx, n_rows, n_columns, raw_data = fetched
    keyp = np.frombuffer(raw_data, dtype=np.float32).reshape(n_rows, n_columns).copy()
    keyp = keyp[:,0:2] 
    return keyp

def extract_matches(cursor, im1, im2, verified_matches):
    im1 = int(im1)
    im2 = int(im2)
   
    if im1 < im2:
        switch_cams = False
        pair_id = image_ids_to_pair(im1, im2)
    else:
        switch_cams = True
        pair_id = image_ids_to_pair(im2, im1)

    if verified_matches:
        cursor.execute("SELECT pair_id, rows, cols, data FROM two_view_geometries WHERE pair_id == ?;", (pair_id,))            
    else:
        cursor.execute("SELECT pair_id, rows, cols, data FROM matches WHERE pair_id == ?;", (pair_id,)) # for raw matches (without verification)

    fetched = cursor.fetchone()
    if fetched is None:
        # print(f'No matches found for {pair_id=}')
        return [], []
    pair_idx, n_rows, n_columns, raw_data = fetched
    if raw_data is None:
        return [], []
    matches = np.frombuffer(raw_data, dtype=np.uint32).reshape(n_rows, n_columns)

    #if verified_matches:
    #    print("Found " + str(n_rows) + " geometrically verified matches between image " + str(im1) + " and " + str(im2))
    #else:
    #    print("Found " + str(n_rows) + " matches between image " + str(im1) + " and " + str(im2))

    if switch_cams:
        ind1 = matches[:,1]
        ind2 = matches[:,0]
    else:
        ind1 = matches[:,0]
        ind2 = matches[:,1]

    return list(ind1), list(ind2)

def extract_tuple_matches(cursor, tuple, cycle_consistency=False, verified_matches=False):
    x1 = extract_keyp(cursor, tuple[0])
    x2 = extract_keyp(cursor, tuple[1])
    x3 = extract_keyp(cursor, tuple[2])
    x4 = extract_keyp(cursor, tuple[3])

    m12a, m12b = extract_matches(cursor, tuple[0], tuple[1], verified_matches)
    m13a, m13b = extract_matches(cursor, tuple[0], tuple[2], verified_matches)
    m14a, m14b = extract_matches(cursor, tuple[0], tuple[3], verified_matches)

    m23a, m23b = extract_matches(cursor, tuple[1], tuple[2], verified_matches)
    m24a, m24b = extract_matches(cursor, tuple[1], tuple[3], verified_matches)
    m34a, m34b = extract_matches(cursor, tuple[2], tuple[3], verified_matches)


    ind = list(set(m12a).intersection(m13a).intersection(m14a))

    if len(ind) == 0:
        return [], [], [], []

    x1m = []
    x2m = []
    x3m = []
    x4m = []

    for i1 in ind:
        i2 = m12b[m12a.index(i1)]
        i3 = m13b[m13a.index(i1)]
        i4 = m14b[m14a.index(i1)]

        if cycle_consistency:
            # check that (i2,i3) is matched            
            if not i2 in m23a or m23b[m23a.index(i2)] != i3:
                continue
            # check that (i2,i4) is matched
            if not i2 in m24a or m24b[m24a.index(i2)] != i4:
                continue
            # check that (i3,i4) is matched
            if not i3 in m34a or m34b[m34a.index(i3)] != i4:
                continue

        x1m.append(x1[i1])
        x2m.append(x2[i2])
        x3m.append(x3[i3])
        x4m.append(x4[i4])

    return np.array(x1m), np.array(x2m), np.array(x3m), np.array(x4m)





def load_tuples(sfm_path, num_tuples, min_shared_pts=20, database_name='database.db',
               verified_matches=False, cycle_consistency=False, uniform_images=True, return_pairs=False):
    sfm_path = Path(sfm_path)
    try:
        cameras, images, points3D = read_write_model.read_model(sfm_path, ext=".bin")
    except:
        cameras, images, points3D = read_write_model.read_model(sfm_path, ext=".txt")

    connection = sqlite3.connect(sfm_path / database_name)
    cursor = connection.cursor()

    # To generate the pairs we select a random 3D point and take two random cameras which see this point
    point3D_ids = list(points3D.keys())
    image_ids = list(images.keys())
    #print(images)
    #print(image_ids)
    print(f'{sfm_path} contains {len(image_ids)} images')

    tuple_set = set()
    tuples = []
    poses = []
    matches = []
    num_trials = 0

    with tqdm(total=num_tuples, desc='Number of pairs selected') as pbar:
        while len(tuples) < num_tuples:
            num_trials += 1
            #print(num_trials)
            if num_trials > 40000:
            	break
            if uniform_images:
                candidate_tuple = tuple(sorted(np.random.choice(image_ids, 4, replace=False)))
            else: # Uniform random 3D points; this may bias selection towards images with lots of points
                p3d_id = np.random.choice(point3D_ids)
                p3d = points3D[p3d_id]
                im_ids = p3d.image_ids
                if len(im_ids) < 4:
                    continue
                candidate_tuple = np.random.choice(im_ids, 4, replace=False)
                candidate_tuple = tuple(sorted(candidate_tuple))


            if candidate_tuple in tuple_set:
                continue

            #print(candidate_tuple)


            # check number of shared points
            im1 = images[candidate_tuple[0]]
            im2 = images[candidate_tuple[1]]
            im3 = images[candidate_tuple[2]]
            im4 = images[candidate_tuple[3]]

            overlap = set(im1.point3D_ids).intersection(im2.point3D_ids).intersection(im3.point3D_ids).intersection(im4.point3D_ids)

            if len(overlap) < min_shared_pts:
                continue

            # Okay we have a good pair, time to get the raw matches from the database
            x1,x2,x3,x4 = extract_tuple_matches(cursor, candidate_tuple, cycle_consistency, verified_matches)

            if len(x1) < min_shared_pts:
                continue

            P1 = np.c_[im1.qvec2rotmat(), im1.tvec]
            P2 = np.c_[im2.qvec2rotmat(), im2.tvec]
            P3 = np.c_[im3.qvec2rotmat(), im3.tvec]
            P4 = np.c_[im4.qvec2rotmat(), im4.tvec]

            H = np.r_[P1, np.array([[0.0, 0.0, 0.0, 1.0]])]
            H = np.linalg.inv(H)
            P1 = P1 @ H
            P2 = P2 @ H
            P3 = P3 @ H
            P4 = P4 @ H

            matches.append((x1, x2, x3, x4))
            poses.append((P1,P2,P3,P4))
            tuples.append(candidate_tuple)
            tuple_set.add(candidate_tuple)

            pbar.update()

    camera = cameras[1] # assumes there is a single camera
    camera_dict = {
        'model': camera.model,
        'width': camera.width,
        'height': camera.height,
        'params': camera.params
    }

    # No idea what is going on here....
    # Sometimes camera.params[0] gets overwritten somehow?
    assert(np.abs(camera.params[0]) > 1e-5)

    connection.close()

    if return_pairs:
        return tuples
    else:
        return matches, poses, camera_dict


def load_tuples_from_indices(sfm_path, tuples, database_name='database.db', verified_matches=False, cycle_consistency=False):
    sfm_path = Path(sfm_path)
    cameras, images, points3D = read_write_model.read_model(sfm_path, ext=".bin")

    connection = sqlite3.connect(sfm_path / database_name)
    cursor = connection.cursor()

    matches = []
    poses = []

    for tuple in tqdm(tuples):
        im1 = images[tuple[0]]
        im2 = images[tuple[1]]
        im3 = images[tuple[2]]
        im4 = images[tuple[3]]

        x1,x2,x3,x4 = extract_tuple_matches(cursor, tuple, cycle_consistency, verified_matches)
        P1 = np.c_[im1.qvec2rotmat(), im1.tvec]
        P2 = np.c_[im2.qvec2rotmat(), im2.tvec]
        P3 = np.c_[im3.qvec2rotmat(), im3.tvec]
        P4 = np.c_[im4.qvec2rotmat(), im4.tvec]

        H = np.r_[P1, np.array([[0.0, 0.0, 0.0, 1.0]])]
        H = np.linalg.inv(H)
        P1 = P1 @ H
        P2 = P2 @ H
        P3 = P3 @ H
        P4 = P4 @ H

        matches.append((x1, x2, x3, x4))
        poses.append((P1,P2,P3,P4))        

    camera = cameras[1]  # assumes there is a single camera
    camera_dict = {
        'model': camera.model,
        'width': camera.width,
        'height': camera.height,
        'params': camera.params
    }

    connection.close()

    return matches, poses, camera_dict
