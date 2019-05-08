#!/usr/bin/env bash
set -euo pipefail

# PROJECT_ID: your project's id. Use the PROJECT_ID that matches your Google Cloud Platform project.
# BUCKET_ID: the bucket in which the HyperParameter Tuning models will be stored.
export PROJECT_ID=georgia-tech-ajouandin3
export BUCKET_ID=transformer_trained_model

export DATE_ID=$(date +%Y%m%d_%H%M%S)
export IMAGE_REPO_NAME=dl_gatech_transformer_pytorch_container
export IMAGE_TAG=transformer_pytorch
export IMAGE_URI=gcr.io/${PROJECT_ID}/${IMAGE_REPO_NAME}:${IMAGE_TAG}
export REGION=us-east1

# Check bucket exists
if ! [[ $(gsutil ls gs://) =~ "gs://${BUCKET_ID}/" ]]; then
    echo "Bucket gs://${BUCKET_ID}/ does not exist. Make sure it exists, or change its name."
    exit 1
fi

# Build, test, and push Docker container
docker build -f Dockerfile -t ${IMAGE_URI} ./
docker run ${IMAGE_URI} --epochs 1 --is-test
docker push ${IMAGE_URI}

# Select config file in google_cloud_jobs/*
config_file=$1
CONFIG_PREFIX=_config.yaml
while [[ -z ${config_file} ]]; do
    echo "Pick a config file number to submit job: "
    PS3="Config file number: "
    select config_file in google_cloud_jobs/*${CONFIG_PREFIX}; do
        if [[ -z ${config_file} ]]; then
            echo "\nInvalid choice.\n"
        fi
        break;
    done
done
config_suffix=$(basename ${config_file})
config_suffix=${config_suffix%"$CONFIG_PREFIX"}
JOB_DIR=gs://${BUCKET_ID}/${config_suffix}/${DATE_ID}
JOB_NAME=${config_suffix}_job_${DATE_ID}

# Submit training job
echo
echo "Submitting Job with configuration '${config_file}'..."
gcloud beta ai-platform jobs submit training $JOB_NAME \
  --job-dir=${JOB_DIR} \
  --region=${REGION} \
  --master-image-uri ${IMAGE_URI} \
  --config="${config_file}"
