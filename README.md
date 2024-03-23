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

## DAVIS2016 Dataset

For training, validation and testing we are using A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation\* [DAVIS](https://davischallenge.org/index.html), which is a widely recognized dataset in the field of computer vision, particularly in the area of video object segmentation. It provides a set of high-quality, full-resolution video sequences that are densely annotated for the task of video object segmentation, serving as a benchmark for evaluating algorithms in this domain.

## Train model

Run the training script with optional arguments for the dataset root directory and number of epochs. By default, the dataset root directory is set to ./DAVIS, and epochs are set to 8.

```bash
python train.py <dataset_root_dir> <epochs>
```

During training, you can monitor hyperparameters by watching loss.png, which is updated after each epoch. The output.png will show the model's predicted mask every 10 training iterations.
