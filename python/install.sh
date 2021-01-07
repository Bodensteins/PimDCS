#!/bin/bash

echo y | pip uninstall pimtorch
python setup.py clean
python setup.py install