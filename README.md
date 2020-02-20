Adaptive Covariate Acquisition for Minimizing Total Cost of Classification
==

Python 3 implementation for adaptive (dynamic) acquisition of covariates based on the cost of each covariate and potential decrease of misclassification costs.  
For convenience 4 preprocessed datasets are accompanied:  
Pima Diabetes (pima_5foldCV),  Breast Cancer (breastcancer_5foldCV), Heartdisease (heartDiseaseWithMissing_5foldCV), PhysioNet (pyhsioNetWithMissing_5foldCV)
 
Steps for reproducing the experimental results of proposed method and baselines
==

**1. Initialization**: Set all path constants in "constants.py" and "commonVisualizationDefinitions.py"

**2. Training**: run all evaluation measures and save results.
 For using the group lasso penalty (default) use, e.g.

	$ python precalculateAllNonLinearL1FeatureSets.py pima_5foldCV
	
 For using forward-selection (with asymmetric costs) use, e.g.
 
	$ python precalculateAllGreedyFeatureSets.py pima_5foldCV asymmetricCost
	
**2. Run evaluation**: run all evaluation measures and save results ("greedy" = forward-selection method, 
"nonLinearL1" = group lasso method), e.g.:

	$ python getOptimalSequence_totalCosts.py pima_5foldCV asymmetricCost nonLinearL1

**3. Visualize Results**: Specify the dataset and experimental setting in the beginning of showAllResults.py and run:

	$ python showAllResults.py


Implementation Details
==

For learning the non-linear classifiers we use part of the implementation from the pygam package:
https://pygam.readthedocs.io/en/latest/

