# Segmentation tracking in video

Project Description: The goal is to label an object in one frame of a video and let the algorithm track the exact shape of the object in subsequent frames.

## Setting Up a Python Environment

Optional: Create a Python virtual environment for the project to manage dependencies.

```bash
cd <project>
python -m venv myenv
source myenv/bin/activate
```

## Installing Dependencies

Optional: Install the required Python packages from the requirements.txt file.

```bash
pip install -r requirements.txt
```

## Download the DAVIS2016 Dataset

Instructions to download the dataset required for training the model:

```bash
git clone https://github.com/davisvideochallenge/davis.git
./davis/data/get_davis.sh
```

We will only need the images.

### Source:

https://davischallenge.org/davis2016/code.html
https://github.com/davisvideochallenge/davis

## Train model

Run the training script with optional arguments for the dataset root directory and number of epochs. By default, the dataset root directory is set to ./davis, and epochs are set to 8.

```bash
python train.py <dataset_root_dir> <epochs>
```

During training, you can monitor hyperparameters by watching loss.png, which is updated after each epoch. The output.png will show the model's predicted mask every 10 training iterations.
