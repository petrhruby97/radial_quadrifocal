# Four-view geometry with unknown radial distortion

This is the implementation of paper [Four-view geometry with unknown radial distortion](https://openaccess.thecvf.com/content/CVPR2023/papers/Hruby_Four-View_Geometry_With_Unknown_Radial_Distortion_CVPR_2023_paper.pdf)



## Bibtex
If you use the code in your project, please cite:
```
@inproceedings{Hruby_2023,
  author       = {Petr Hruby and
                  Viktor Korotynskiy and
                  Timothy Duff and
                  Luke Oeding and
                  Marc Pollefeys and
                  Tom{\'{a}}s Pajdla and
                  Viktor Larsson},
  title        = {Four-view geometry with unknown radial distortion},
  booktitle    = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year         = {2023},

```

In our code, we use efficient homotopy continuation implementation [MINUS](https://github.com/rfabbri/minus). If you use that in your project, please also cite:

```
@inproceedings{Fabbri_2020_TRPLP,
  author       = {Ricardo Fabbri and
                  Timothy Duff and
                  Hongyi Fan and
                  Margaret H. Regan and
                  David da Costa de Pinho and
                  Elias P. Tsigaridas and
                  Charles W. Wampler and
                  Jonathan D. Hauenstein and
                  Peter J. Giblin and
                  Benjamin B. Kimia and
                  Anton Leykin and
                  Tom{\'{a}}s Pajdla},
  title        = {{TRPLP} - Trifocal Relative Pose From Lines at Points},
  booktitle    = {2020 {IEEE/CVF} Conference on Computer Vision and Pattern Recognition,
                  {CVPR} 2020, Seattle, WA, USA, June 13-19, 2020},
  year         = {2020},
}
```


## THE REPOSITORY

### ./rqt/	THE SOLVERS

	13 Explicit solver:
		BODY: radial_quadrifocal_solver.cc
		HEADER: radial_quadrifocal_solver.h
		HC: homotopy.h
	13 Implicit solver:
		BODY: nanson2_radial_quadrifocal_solver.cc
		HEADER: nanson2_radial_quadrifocal_solver.h
		HC: nanson2_homotopy.h
	15 Linear solver:
		BODY: linear_radial_quadrifocal_solver.cc
		HEADER: linear_radial_quadrifocal_solver.h
	7 Upright Explicit solver:
		BODY: upright_radial_quadrifocal_solver.cc
		HEADER: upright_radial_quadrifocal_solver.h
		HC: upright_homotopy.h
	7 Upright Implicit solver:
		BODY: upright_nanson_radial_quadrifocal_solver.cc
		HEADER: upright_nanson_radial_quadrifocal_solver.cc
		HC: upright_nanson_homotopy.h
	Others:
		INTERFACE FOR RANSAC: quadrifocal_estimator.cc, quadrifocal_estimator.h
		METRIC UPGRADE + CHEIRALITY FILTER: metric_upgrade.cc, metric_upgrade.h
		RANSAC: ransac_impl.h
		SOLVER FOR det(A + alpha*B) = 0 (used to construct the Radial Quadrifocal Tensor): solver_det4.cc, solver_det4.h
		SETTINGS AND STARTING SOLUTIONS FOR HC: types.cc, types.h
		CHEIRALITY FILTER FOR UPRIGHT SOLVERS: upright_filter_cheirality.cc, upright_filter_cheirality.h

### ./pybind/ PYTHON BINDINGS

	pyrqt.cc: Contains python bindings for the functions defined in folder ./rqt/


### ./eval/
	Noiseless stability tests:
		stability_test_13explicit.py
		stability_test_13implicit.py
		stability_test_15linear.py
		stability_test_7explicit.py
		stability_test_7implicit.py
	These scripts are run without arguments, they print out camera, rotation, and translation error of 100000 randomly generated problems, each on a single line.
	
	Robustness towards noise tests:
		noise_test_13explicit.py
		noise_test_13implicit.py
		noise_test_15linear.py
		noise_test_7explicit.py
		noise_test_7implicit.py
	These scripts are run without arguments, they iteratively increase the noise level from 0 to 10px, and for every noise level, they print the following values:
		noise in px
		percentage of problems, for which the solver returned a solution
		average camera error
		average rotation error (in degrees)
		average translation error (in world units)
		average camera error calculated over solutions, where the solver returned a solution
		average rotation error (in degrees) calculated over solutions, where the solver returned a solution
		average translation error (in world units) calculated over solutions, where the solver returned a solution
		percentage of problems with rotation error below 1 degree
		percentage of problems with rotation error below 5 degrees
		percentage of problems with rotation error below 10 degrees
		percentage of problems with translation error below 0.01 world units
		percentage of problems with translation error below 0.05 world units
		percentage of problems with translation error below 0.1 world units
	
	Test with real data:
		test_ransac.py
	This script runs without arguments. It extracts camera quadruplets from a COLMAP model, whose address has to be specified in the script. For every solver, it produces the following values:
		Name of the solver
		Percentage of quadruplets, for which the solver generated some pose
		Average camera error
		Average translation error
		Percentage of poses, whose rotation error is below 5, 10, 20 degrees
		Percentage of poses, whose translation error is below 0.5, 1, 5 meters

## INSTALLATION

1. clone the repository
2. mkdir build
3. cd build
4. cmake ..
5. make
6. there is a compiled python library "./build/pybind/pyrqt.cpython-38-x86_64-linux-gnu.so"
7. If you want to run the scripts in ./eval/, copy the file "./build/pybind/pyrqt.cpython-38-x86_64-linux-gnu.so" to ./eval/


## RUNNING

The tests are located in ./eval/ <br>
They are run without arguments as python3 name_of_test.py <br>
The list of the tests and their output is given above.



