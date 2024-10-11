# AMAT
In direct exoplanet detection, existing algorithms use techniques based on a low-rank approximation to separate the rotating planet signal from the quasi-static speckles. We present a novel approach that iteratively finds the planet’s flux and the low-rank approximation of quasi-static signals, strengthening the existing models. 


### CONTENTS:

* README: this file
* amat.py: the main code for AMAT algorithm
* l1lracd.py: the functions for calculating l1 norm LRA 
* util.py: the utilized functions for our proposed algorithm
* test_AMAT.ipynb: test of L1 and L2 norm for exoplanet detection as a detection map comparison.



### CITE:
Please cite "An Alternating Minimization Algorithm with Trajectory for Direct Exoplanet Detection -- The AMAT Algorithm" 
https://doi.org/10.48550/arXiv.2410.06310
and "An Alternating Minimization Algorithm with Trajectory for Direct Exoplanet Detection" https://doi.org/10.14428/esann/2023.ES2023-137.  

Please also provide a link to this webpage in your paper (https://github.com/hazandaglayan/amat)

### Dependencies:
You need to install VIP_HCI, numpy, and joblib. 
