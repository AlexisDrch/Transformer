# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.\n",
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Install the latest version of pytorch
FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime
WORKDIR /root
# Installs pandas, google-cloud-storage, and cloudml-hypertune
RUN pip install pandas google-cloud-storage cloudml-hypertune


# Install requirements
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Set up the entry point to invoke the trainer.
ENTRYPOINT ["python", "task.py"]

# Install TensorboardX with Google Cloud Storage support
RUN pip install tensorflow "git+https://github.com/CatalinVoss/tensorboardX.git@patch-1"

# Copies the trainer code to the docker image.
COPY . ./
