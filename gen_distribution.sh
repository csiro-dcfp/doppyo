#!/bin/bash

rm -r dist
rm -r build
rm -r doppyo.egg-info 
python3 setup.py sdist bdist_wheel 
