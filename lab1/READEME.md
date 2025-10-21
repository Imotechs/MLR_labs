# Machine Learning Regression and Data-Driven Modeling lab work

This repository contains practical implementations of **Singular Value Decomposition (SVD)**, linear regression, LASSO, Robust PCA, and dimensionality reduction techniques on real datasets. The lab demonstrates how matrix decomposition can be applied to image compression, housing price prediction, facial recognition, and denoising.

## Overview

The lab explores fundamental matrix decomposition methods and their applications:
- Image compression using SVD
- Linear regression for housing prices
- Facial recognition with eigenfaces
- Sparse modeling via LASSO
- Robust PCA for separating structured and noisy components
- Optimal truncation of singular values for noisy data

## Dataset Information

### California Housing Dataset
- **Source**: 1990 US Census
- **Features**: 8 variables (median income, house age, average rooms, etc.)
- **Target**: Median house value (in thousands of dollars)

### Olivetti Faces Dataset
- **Content**: 400 grayscale images (64×64 pixels) of 40 individuals
- **Structure**: 10 images per individual with varying lighting and expressions
- **Usage**: Facial recognition and eigenface analysis

## Implementations

### 1. Image Compression with SVD
- **Process**:
  1. Convert RGB image to grayscale
  2. Apply SVD: `X = U @ S @ VT`
  3. Truncate singular values using two methods:
     - **Energy-based truncation**: Keeps 90% of total energy (`r_energy=72`)
     - **Manual truncation**: Keeps top 10 singular values (`r1=10`) for extreme compression
- **Outcome**: Low-rank approximations preserve image structure while reducing storage.

### 2. Housing Price Prediction
- Linear regression model solved using SVD: `x = VT.T @ np.linalg.inv(np.diag(S)) @ U.T @ b`
- **Train/Test Split**: 70/30
- **Performance**: 
  - Train RMSE: ~70–80k
  - Test RMSE: ~70–80k
  - Test R² indicates reasonable predictive ability
- **Feature Importance**: Standardized coefficients show median income as the most significant predictor.

### 3. Eigenfaces for Facial Recognition
- **Method**:
  1. Compute mean face
  2. Mean-center training faces
  3. Apply SVD to obtain eigenfaces
  4. Reconstruct test face using top `r` eigenfaces
- **Results**:
  - `r=1`: Very blurry, basic shape only
  - `r=10`: Recognizable features
  - `r=50`: High-quality reconstruction
- Shows that faces can be efficiently represented by a few components.

### 4. LASSO Regression
- **Synthetic example** with 100 samples and 10 predictors
- **Comparison**:
  - L2 regression (least squares)
  - LASSO (L1 regularization)
  - Debiased LASSO
- Cross-validation selects the optimal `alpha` for sparsity
- Demonstrates LASSO’s ability to drive small coefficients to zero for interpretability.

### 5. Robust PCA for Face Decomposition
- **Goal**: Decompose facial images into low-rank (structured) and sparse (noise/shadows) components
- **Algorithm**:
  - Singular value thresholding (SVT) updates low-rank
  - Shrinkage updates sparse component
  - Iterates until convergence
- **Results**: Clean separation of main facial structure and lighting/noise artifacts

### 6. Optimal Truncation for Noisy Data
- **Scenario**: SVD on noisy measurements
- **Optimal hard threshold** formula:
