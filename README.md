# Deep Learning

### Project for the [Deep Learning](https://www.cc.gatech.edu/classes/AY2019/cs7643_spring/#project) class

**Public page:** [https://deepfrench.gitlab.io/deep-learning-project](https://deepfrench.gitlab.io/deep-learning-project) 
</br>
**Repo page:** [https://gitlab.com/DeepFrench/deep-learning-project](https://gitlab.com/DeepFrench/deep-learning-project)

## Training

There are two ways to train the model: either use the training 
script, or use Google Cloud ~~ML-Engine~~ AI Platform.

### Downloading and running the training script

From a fresh AWS/Google Cloud Compute instance, 
run the following to install everything:

```bash
bash <(curl -s https://deepfrench.gitlab.io/deep-learning-project/install.sh)
```

Then, just run the following:

```bash
conda activate python3-dl
python trainer.py
```

Feel free to modify at your convenience the training configuration located in `trainer.py`.

### Training on Google Cloud AI Platform

Just edit and run the `train-on-google-cloud.sh` script.
You'll need to update `PROJECT_ID` and `BUCKET_ID`. 

**Make sure the bucket exists before running the script!**

The script will create a Docker container with the code, 
push it to Google Cloud Container Registry, and submit a 
training task on AI Platform to tune hyperparameters according 
to the configuration file you select in `google_cloud_jobs/`.

Documentation:

- [Using Hyperparameter Tuning](https://cloud.google.com/ml-engine/docs/tensorflow/using-hyperparameter-tuning)
- [HyperParameter Spec Documentation](https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#HyperparameterSpec)
- [Hyperparameter tuning in Cloud Machine Learning Engine using Bayesian Optimization](https://cloud.google.com/blog/products/gcp/hyperparameter-tuning-cloud-machine-learning-engine-using-bayesian-optimization) (Blog post) 

## Installing

### Cloning

[Install `git lfs`](https://git-lfs.github.com) (on macOS: `brew install git-lfs`), **and then** run:

```bash
git lfs install
```

If you installed `git-lfs` after cloning the repo, you can use the following command to download LFS files:

```bash
git lfs fetch
git lfs pull
```

### Setting up an environment *(Optional)*

If you set up a virtual environment and store it in the root folder, make sure 
not to add it to git to name it like one of those options in the `.gitignore`:

```
env/
venv/
ENV/
env.bak/
venv.bak/
```

### Installing packages

```bash
pip install -r requirements.txt
```

## Running tests

To run all tests with test discovery:

```bash
python -m unittest
```