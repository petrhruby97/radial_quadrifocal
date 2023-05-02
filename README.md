# Four-view geometry with unknown radial distortion

This is the implementation of paper [Four-view geometry with unknown radial distortion](https://www.youtube.com/watch?v=TnlPtaPxXfc) (TODO ADD THE PROPER LINK). (TODO ADD SOME BRIEF DESCRIPTION OF THE PROJECT)

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

./rqt/	THE SOLVERS
	TODO LIST
	TODO READ THROUGH THE CODE AND DO SOME CLEANING

./pybind/ PYTHON BINDINGS

./eval/	PYTHON SCRIPTS WITH REAL AND SYNTHETIC TESTS
	TODO THIS PART NEEDS THE MOST CLEANING
	1. look into the folder in the second disk of the computer and see what is there
	2. write down the purpose of each of the tests
	3. sort out the test scripts into subfolders, remove those that are not necessary, and unify the input/output
	4. clean up the code, write down headers to each file


## INSTALLATION

1. clone the repository
2. mkdir build
3. cd build
4. cmake ..
5. make
6. wait until the compillation finishes There will be some warnings but [that's life](https://www.youtube.com/watch?v=TnlPtaPxXfc).
7. there is a compiled python library "./build/pybind/pyrqt.cpython-38-x86_64-linux-gnu.so"


## RUNNING

TODO WRITE DOWN THE LIST OF THE TESTS
TODO WRITE DOWN THE INPUT AND OUTPUT FORMAT FOR EACH TEST



