SentimentNet: End-to-End Neural Network for Text Review Classification
Overview
This repository showcases a complete sentiment analysis pipeline where I implement a custom neural network using only Python and NumPy. The goal is to classify text reviews into positive or negative sentiments, demonstrating not just technical fluency in building neural architectures, but also deep understanding of NLP preprocessing and end-to-end machine learning workflows.

All methodology, explanations, and code in this repository are authored and thoroughly documented by me. This project is suitable as a portfolio piece for data science, business analytics, and technical interview contexts.

Table of Contents
Project Motivation

Key Features

Workflow

Getting Started

Requirements

Usage

Results & Discussion

Customization & Extensions

Author

License

Project Motivation
Sentiment analysis is an important application in business analytics, customer feedback, marketing, and product research. Many industry solutions use “black box” libraries—my objective here is to deconstruct and rebuild the core mechanics, ensuring full transparency and learning at every stage.

This project demonstrates:

How to process and encode textual data for machine learning

The mechanics of neural networks (forward/backward propagation, gradient descent) implemented manually

The importance of vocabulary selection and feature engineering in NLP

Iterative model optimization from basic to efficient networks

Key Features
Manual Neural Network Implementation: Direct coding of all neural layers, weights, activations (no frameworks like TensorFlow or PyTorch).

Custom Backpropagation & Training Loop: All weight updates and learning rates are computed and tuned explicitly.

Advanced Text Preprocessing: Vocabulary is built from real reviews, filtered for importance and polar sentiment, and encoded for input into the model.

Stepwise Optimization: Each section illustrates how tweaks in data and model design influence accuracy and computational efficiency.

Thorough Documentation: Every step, from data import to evaluation, is explained in first-person to maximize clarity and reusability.

Workflow
Dataset Import: Text reviews and corresponding sentiment labels are loaded, inspected, and cleaned.

Vocabulary Engineering: The review text is processed to select the most valuable features for sentiment prediction, minimizing noise and redundancy.

Model Creation: I define and initialize a neural network class with configurable input, hidden, and output layers.

Training: The network is trained using the training set. Gradients are calculated and weights are updated using backpropagation.

Evaluation: The model’s accuracy is computed on a test set, and iterative improvements are discussed.

Refinement: Learning rate, hidden node count, and vocabulary size are analyzed and optimized to balance speed and performance.

Result Interpretation: Test cases and results are interpreted to highlight strengths, limitations, and next steps.

Getting Started
Clone this repository and open ML_Project_1.ipynb in your preferred Jupyter Notebook environment.
All code is ready to execute cell by cell, with explanatory markdown for every major step.

Requirements
Python 3.x

NumPy

Install dependencies via:

bash
pip install numpy
Usage
Open the Notebook:
Start ML_Project_1.ipynb in Jupyter or Google Colab.

Run Cells Sequentially:
Follow the notebook blocks to read the workflow, understand the methodology, and execute the code.

Analyze Outputs:
Each step prints intermediate results and final accuracy, clarifying the impact of improvements.

Experiment:
Adjust vocabulary selection, hyperparameters (learning rate, hidden nodes), or add your own dataset to explore more.

Results & Discussion
The model demonstrates end-to-end sentiment classification on textual data with accuracy improving as noisy or irrelevant words are filtered and network parameters are tuned.

Manually implementing the neural network provides transparency and learning depth unavailable in “black box” solutions.

The approach illustrates practical skills in both machine learning engineering and real-world NLP applications relevant to business analytics.

Customization & Extensions
Feel free to:

Replace the dataset with your own reviews or text corpus.

Modify the preprocessing logic to suit domain-specific sentiment features.

Extend the neural network architecture or integrate additional layers/functions for experimentation.

Use the workflow as a basis for classification in other NLP applications (topic modeling, spam detection, etc).

Author
All code and documentation authored by Nimish Abraham.
For questions, collaborations, or feedback, please contact me through GitHub.

License
This project is open-source and licensed under the MIT License.
