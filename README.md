# **Repository for pyLatte package** #
### - A Python package for computing diagnostics and verifications using the CAFE system output 

--------------------------
Contains codes/documents associated with the pyLatte package (name courtesy of Vassili)
* verif.py - functions for assessing one data set relative to another (usually model output relative to observation).
* utils.py - general support functions for the pyLatte package. 
* **tests** - contains tests for various modules of the pyLatte package.
* **tutorials** - contains tutorials on using the pyLatte package.

--------------------------
#### To use the pyLatte package:
1)   Clone the pyLatte Bitbucket repository to your local machine: 
 
`git clone https://<userid>@bitbucket.csiro.au/scm/df/pylatte.git`
 
2)   Within your .bashrc (or equivalent), add the pyLatte location to your PYTHONPATH:

`PYTHONPATH="${PYTHONPATH}:/location/of/pyLatte/clone"`

`export PYTHONPATH`
 
3)   Replicate the pylatte_env environment, containing all necessary packages:

In `pyLatte/requirements`, run `conda env create -f pylatte_env.yml`

Contact: Dougie Squire

