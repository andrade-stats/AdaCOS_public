Adaptive Covariate Acquisition for Minimizing Total Cost of Classification
==

Python 3 implementation for adaptive (dynamic) acquisition of covariates based on the cost of each covariate and potential decrease of misclassification costs

 
Steps for reproducing the experimental results of proposed method and baselines
==

**1. Initialization**: Set all path constants in "constants.py" and "commonVisualizationDefinitions.py"

 	$ python syntheticDataExperiments.py smallContra 100 proposed

**2. Training**: run all evaluation measures and save results, e.g.:

	$ python /data/intelpython3/bin/python precalculateAllNonLinearL1FeatureSets.py pima_5foldCV
	$ python /opt/intel/intelpython3/bin/python precalculateAllNonLinearL1FeatureSets.py breastcancer_5foldCV
	$ python /opt/intel/intelpython3/bin/python precalculateAllNonLinearL1FeatureSets.py heartDiseaseWithMissing_5foldCV

**2. Run clustering evaluation**: run all evaluation measures and save results, e.g.:

	$ python evalAndSaveResults.py SYNTHETIC_DATA onlyNu 10 all smallContra 1000

**3. Visualize Results**: The results are save in the folder "plots".

	$ python analyzeClustering.py


ADMM Implementation
==
 
 ADMM.py
 parallelADMM(dataFeatures, dataLabels, covariateSims, origNu, origGamma, optParams, warmStartAuxilliaryVars, paramId)
 
 bStep.py
 optimizeLBFGSNew(dataFeatures, dataLabels, edgesZ, singleZ, edgesU, singleU, rho, B0, beta0, allAdjacentNodes, MAX_ITERATIONS)
 
 zStep.py
 updateZ_fast(edgesZ, singleZ, edgesU, singleU, B, rho, S, nu, gamma, allAdjacentNodes)
 
 uStep.py
 updateU_fast(edgesZ, singleZ, edgesU, singleU, B, allAdjacentNodes)
 
 identify connected components:
 getClusteringAndSelectedFeatures(edgesU, singleU, B, rho, S, nu, gamma, allAdjacentNodesPrior)
 in zStep.py


Visualization Details
==

the visualization in "analyzeClustering.py" uses the colors extracted from:
http://tools.medialab.sciences-po.fr/iwanthue/
and the graphviz package 
https://www.graphviz.org/

