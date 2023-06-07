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
		BODY: linear_radial_quadrifocal_solver.cc <br>
		HEADER: linear_radial_quadrifocal_solver.h <br>
	7 Upright Explicit solver: <br> <br>
		BODY: upright_radial_quadrifocal_solver.cc <br>
		HEADER: upright_radial_quadrifocal_solver.h <br>
		HC: upright_homotopy.h <br> <br>
	7 Upright Implicit solver: <br>
		BODY: upright_nanson_radial_quadrifocal_solver.cc <br>
		HEADER: upright_nanson_radial_quadrifocal_solver.cc <br>
		HC: upright_nanson_homotopy.h <br> <br>
	Others: <br>
		INTERFACE FOR RANSAC: quadrifocal_estimator.cc, quadrifocal_estimator.h <br>
		METRIC UPGRADE + CHEIRALITY FILTER: metric_upgrade.cc, metric_upgrade.h <br>
		RANSAC: ransac_impl.h <br>
		SOLVER FOR det(A + alpha*B) = 0 (used to construct the Radial Quadrifocal Tensor): solver_det4.cc, solver_det4.h <br>
		SETTINGS AND STARTING SOLUTIONS FOR HC: types.cc, types.h <br>
		CHEIRALITY FILTER FOR UPRIGHT SOLVERS: upright_filter_cheirality.cc, upright_filter_cheirality.h <br> <br>

### ./pybind/ PYTHON BINDINGS

	pyrqt.cc: Contains python bindings for the functions defined in folder ./rqt/ <br> <br>


### ./eval/
	Noiseless stability tests: <br>
		stability_test_13explicit.py <br>
		stability_test_13implicit.py <br>
		stability_test_15linear.py <br>
		stability_test_7explicit.py <br>
		stability_test_7implicit.py <br>
	These scripts are run without arguments, they print out camera, rotation, and translation error of 100000 randomly generated problems, each on a single line. <br> <br>
	
	Robustness towards noise tests: <br>
		noise_test_13explicit.py <br>
		noise_test_13implicit.py <br>
		noise_test_15linear.py <br>
		noise_test_7explicit.py <br>
		noise_test_7implicit.py <br>
	These scripts are run without arguments, they iteratively increase the noise level from 0 to 10px, and for every noise level, they print the following values: <br>
		noise in px <br>
		percentage of problems, for which the solver returned a solution <br>
		average camera error <br>
		average rotation error (in degrees) <br>
		average translation error (in world units) <br>
		average camera error calculated over solutions, where the solver returned a solution <br>
		average rotation error (in degrees) calculated over solutions, where the solver returned a solution <br>
		average translation error (in world units) calculated over solutions, where the solver returned a solution <br>
		percentage of problems with rotation error below 1 degree <br>
		percentage of problems with rotation error below 5 degrees <br>
		percentage of problems with rotation error below 10 degrees <br>
		percentage of problems with translation error below 0.01 world units <br>
		percentage of problems with translation error below 0.05 world units <br>
		percentage of problems with translation error below 0.1 world units <br> <br>
	
	Test with real data: <br>
		test_ransac.py <br> <br>
	This script runs without arguments. It extracts camera quadruplets from a COLMAP model, whose address has to be specified in the script. For every solver, it produces the following values: <br>
		Name of the solver <br>
		Percentage of quadruplets, for which the solver generated some pose <br>
		Average camera error <br>
		Average translation error <br>
		Percentage of poses, whose rotation error is below 5, 10, 20 degrees <br>
		Percentage of poses, whose translation error is below 0.5, 1, 5 meters <br>

## INSTALLATION

1. clone the repository
2. mkdir build
3. cd build
4. cmake ..
5. make
6. wait until the compillation finishes There will be some warnings but [that's life](https://www.youtube.com/watch?v=TnlPtaPxXfc).
7. there is a compiled python library "./build/pybind/pyrqt.cpython-38-x86_64-linux-gnu.so"
8. If you want to run the scripts in ./eval/, copy the file "./build/pybind/pyrqt.cpython-38-x86_64-linux-gnu.so" to ./eval/


## RUNNING

TODO WRITE DOWN THE LIST OF THE TESTS
TODO WRITE DOWN THE INPUT AND OUTPUT FORMAT FOR EACH TEST



