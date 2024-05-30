#!/bin/bash

pip install --upgrade pip
pip install wheel
pip install modelcif
pip install "fair-esm[esmfold]"
pip install "dllogger @ git+https://github.com/NVIDIA/dllogger.git"
pip install "openfold @ git+https://github.com/aqlaboratory/openfold.git@v1.0.1"
pip install --upgrade deepspeed