#!/usr/bin/env bash
set -euo pipefail

test -x $(which conda) || (echo "Please install conda first" && exit 1)

# clone repo
git clone git@gitlab.com:DeepFrench/deep-learning-project.git

cd deep-learning-project/

# create conda env from environment file
conda env create -f environment.yml

