# Archery Score Counter

## Overview

---

## Introduction

Welcome to the **Archery Target Scoring Automation** project! This repository contains a Python script designed to automate the process of scoring archery targets. By leveraging advanced image processing techniques and the powerful YOLO (You Only Look Once) object detection model, this project accurately detects hit points on an archery target, calculates distances from the center, and assigns scores based on predefined radii.

### Key Features:
- **Target Board Detection**: Uses a pre-trained YOLO model to detect and isolate the target board from the image.
- **Hit Point Identification**: Detects and locates hit points on the target using color detection and contour analysis.
- **Score Calculation**: Measures distances from the center and assigns scores according to standard archery scoring rules.
- **Visual Output**: Displays the processed image with scores annotated for each hit point and the total score.

### Getting Started:
To get started, follow the installation instructions and run the provided script on your target images. Check out the detailed documentation and code explanations in this repository for a comprehensive understanding of how the system works.

Feel free to explore, contribute, and enhance this project. Happy coding and scoring!

---

This introduction provides a clear overview of your project, highlighting its key features and guiding users on how to get started.
## Requirements

- **Python 3.10**
- The required libraries are specified in the `environment.yaml` file.

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/abdullah0307/Archery-Score-Counter.git
cd Archery-Score-Counter
```

### 2. Create the Conda Environment

Ensure you have Anaconda or Miniconda installed on your system. Then, create the environment using the provided `environment.yaml` file:

```bash
conda env create -f environment.yaml
```

### 3. Activate the Environment

```bash
conda activate archery
```

### 4. Verify Installation

Run the following command to ensure all packages are installed correctly:

```bash
python -m pip check
```

## Running the Code

### Step-by-Step Instructions

1. **Prepare Your Data**:
   - Ensure your input data is placed in the appropriate directory (e.g., `data/input`).

2. **Run the Script**:
   - Execute the main script to start the process.
   
   ```bash
   python start.py
   ```

3. **View Results**:
   - The output will be saved in the `output` directory or displayed on the screen as specified.

## References

- [OpenCV Documentation](https://docs.opencv.org/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Ultralytics Documentation](https://github.com/ultralytics/yolov5)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

## Contact

If you have any questions or need further assistance, feel free to contact me at abdullahjavaid0307@gmail.com

---

This README file provides a comprehensive guide for setting up and running your project, including cloning the repository, creating the environment, running the code, and references for further reading.