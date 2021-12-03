Directory Structure —

HW2_cs5180098
  Q1.sh                   (script file for runing Q1)
  Q2.sh                   (script file for runing Q2)
  readme_cs5180098.txt    (description on how to run)
  report_cs5180098.pdf    (implenataion details and observation)
  install.sh              (for running on hpc)

  Q1         (directory)
	  |-> gspan       (directory)
	  |-> fsg         (directory)
	  |-> gaston      (directory)
	  |-> output      (directory)
	  |-> format.py   (preprocess the input)
	  |-> plot.py     (run the algrithms and plot the runtime comparision graph)
  Q2         (directory)
    |-> mtree_library.py       (library taken form https://github.com/tburette/mtree/blob/master/mtree.py)
    |-> knn.py         (run kdtree,mtree and sequential search and plot the graph)
    |-> kdtree.py  (for running kdtree only)
    |-> mtree.py   (for runing mtree only)
    |-> algorithms.txt (contains the pseudocde for finding k-NN in kdtree and mtree)


How to run this code on HPC?
  submission follows the instruction given in problem statement.
  "source install.sh" clone our repositary, move to the location cataining our hw2 submission and loads the required modules to run this program on hpc.
  run "unzip HW2_cs5180098.zip" will produce a folder named HW2_cs5180098 containing our submision 
  In HW2_cs5180098 run the following command to get the plots.
  |-> sh Q1.sh <data> <plot name>    (for running Q1)
  |-> sh Q2.sh <data> <plot name>    (for running Q2)

  to load all the required modules, we can also run below commands —
    |-> module load compiler/gcc/9.1.0
    |-> module load compiler/python/3.6.0/ucs4/gnu/447
    |-> module load pythonpackages/3.6.0/matplotlib/3.0.2/gnu
    |-> module load pythonpackages/3.6.0/numpy/1.16.1/gnu
    |-> module load pythonpackages/3.6.0/scikit-learn/0.21.2/gnu
    |-> module load pythonpackages/3.6.0/pandas/0.23.4/gnu
    |-> module load pythonpackages/3.6.0/scipy/1.1.0/gnu

  
  for preprocessing individual function following command can be used — 
    |-> python format.py <data> output/<processed data> gspan
    |-> python format.py <data> output/<processed data> fsg
    |-> python format.py <data> output/<processed data> gaston

  above three commands process data for corresponding mentioned subgraph
  mining tools i.e. gspan, fsg, and gaston respectively

  once preprocessing is done, the next and the below command will 
  run these subgraph mining tools on thier respective processed 
  data with different supports and plot the graph. the command is —
    |-> python plot.py output/<data gspan> output/<data fsg> output/<gaston> <plot name>

  The following command is sufficient to run Q2 for ploting average query time
    |-> python knn.py <data> <plot name>

  The following commands can be used to run the individual parts in Q2 i.e. kdtree and mtree.
    |-> python kdtree.py <data> <reduced_dimension> <no_of_nearest_neighbour>
    |-> python kdtree.py <data> <reduced_dimension> <no_of_nearest_neighbour>  
  
  in above command if second and third are not provided then the program will take default
  values as: reduced_dimension = 2 and no_of_nearest_neighbour=5.
