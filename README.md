# ML_LP_Hybrid
This repository contains the implementation of a novel hybrid approach that integrates linear programming (LP) into the loss function of an unsupervised machine learning model. By leveraging the strengths of both optimization techniques and machine learning, this method provides a robust solution for solving complex optimization problems.

# Unsupervised Machine Learning Hybrid Approach with Linear Programming

This repository contains the implementation of a novel hybrid approach that integrates linear programming (LP) into the loss function of an unsupervised machine learning model. By leveraging the strengths of both optimization techniques and machine learning, this method provides a robust solution for solving complex optimization problems.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Model Overview](#model-overview)
- [Training the Model](#training-the-model)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Linear programming (LP) is a widely used optimization technique, but it has limitations in dealing with non-linear, high-dimensional, and dynamic environments. This repository implements a hybrid approach that integrates LP constraints directly into the loss function of a machine learning model, particularly in an unsupervised setting.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/andrew-jeremy/ML_LP_Hybrid.git

2. Navigate to the project directory:
   ```bash
   cd ML_LP_Hybrid

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

## Usage
4. To train the model and predict resource allocation, simply run the following command:
   ```bash
   python ml_lp_hybrid.py

## Model Overview
The model uses an autoencoder architecture with a custom loss function that integrates LP constraints. The loss function includes both a reconstruction loss and a penalty term derived from the LP constraints, guiding the model to adhere to these constraints while learning the underlying structure of the data.

## Training the Model
The training process involves the following steps:

## Generating synthetic data that represents resource allocation scenarios.
Training an autoencoder with the custom LP loss function.
Evaluating the model's performance based on constraint satisfaction and reconstruction error.

## Examples
Example usage of the trained model with new data is included in the if __name__ == "__main__": block in the script.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License.


