# Survival Analysis with Cox Proportional Hazards Model for correlation survival outcomes

This project implements a Cox Proportional Hazards model with frailty terms using Pycox and PyTorch. The code is designed for survival analysis on datasets with random effects, focusing on predicting survival times and handling non-linear effects.

## Table of Contents
- [Project Overview](#project-overview)
- [Dependencies](#dependencies)
- [Acknowledgements](#acknowledgements)

## Project Overview
This code trains a Cox Proportional Hazards model with frailty terms on survival data, using the `Pycox` library along with neural network extensions in `PyTorch`. The model is tested on datasets with varying covariates and evaluates performance using metrics such as the C-index and Integrated Brier Score.
## Dependencies
Python 3.6+
PyTorch
Pycox for survival analysis models
numpy, pandas, scikit-learn for data processing
matplotlib, seaborn for plotting
## Acknowledgements
This project uses the Pycox library for survival analysis models and torchtuples for additional PyTorch utility functions.
