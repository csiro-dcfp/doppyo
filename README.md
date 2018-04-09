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
**1)** Install anaconda if you have not already done so:

a) Make your own directory in '/OSM/CBR/OA_DCFP/apps/'

`mkdir /OSM/CBR/OA_DCFP/apps/<userid>`

b) A recent version of the anaconda install is available in '/OSM/CBR/OA_DCFP/apps/anaconda_installer'. Run this and follow the prompts to install anaconda in the directory you have just created.

c) The anaconda installer will add the anaconda path to your .bashrc (or equivalent). You will need to source this for the changes to be made:

`source ~/.bashrc`

d) Check that anaconda python is now your default:

`which python`

**2)**   Clone the pyLatte Bitbucket repository to your local machine: 
 
`git clone https://<userid>@bitbucket.csiro.au/scm/df/pylatte.git`
 
**3)**   Within your .bashrc (or equivalent), add the pyLatte location to your PYTHONPATH:
`PYTHONPATH="${PYTHONPATH}:/location/of/pyLatte/clone"`

`export PYTHONPATH`
 
**4)**   Replicate the pylatte_env environment locally:

In 'pyLatte/requirements' run `conda env create -f pylatte_env.yml`

Or, if updating existing pylatte_env, activate the environment, then in 'pyLatte/requirements' run `conda env update --file=pylatte_env.yml`

**5)**   Activate the pylatte_env environment:

`source activate pylatte_env`

Contact: Dougie Squire

