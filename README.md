# Linear-Regression-and-Gradient-Descent-Methods

## Overview

This project involves a analysis and implementation of various linear regression models and optimization algorithms to predict the `energy` variable from a given dataset. The primary objectives are:

- Preprocessing data and encoding categorical variables.
- Training and evaluating different linear regression models.
- Performing feature selection to improve model performance.
- Implementing gradient descent variants for Ridge regression.
- Analyzing the convergence rates of different optimization methods.

---

## Table of Contents

- [Data Preparation](#data-preparation)
- [Task 1: Data Splitting](#task-1-data-splitting)
- [Task 2: Training Models without Categorical Features](#task-2-training-models-without-categorical-features)
- [Task 3: Training Models with Categorical Features](#task-3-training-models-with-categorical-features)
- [Task 4: Examination of Model Parameters](#task-4-examination-of-model-parameters)
- [Task 5: Feature Selection Using Backward Elimination](#task-5-feature-selection-using-backward-elimination)
- [Task 6: Hyperparameter Tuning for Lasso Regression](#task-6-hyperparameter-tuning-for-lasso-regression)
- [Task 7: Implementing Gradient Descent Variants for Ridge Regression](#task-7-implementing-gradient-descent-variants-for-ridge-regression)
- [Task 8: Training and Validating Manual Models](#task-8-training-and-validating-manual-models)
- [Task 9: Analyzing Convergence Rates of Gradient Descent Methods](#task-9-analyzing-convergence-rates-of-gradient-descent-methods)
- [Conclusion](#conclusion)

---

## Data Preparation

### Purpose

The goal of data preparation is to clean and preprocess the dataset to ensure it is suitable for modeling. This includes handling categorical variables, scaling numerical features, and splitting the data into training and test sets.

### Analysis

1. **Loading the Dataset**: The dataset is read into a Pandas DataFrame for easy manipulation and analysis.

2. **Separating Features and Target Variable**: The `energy` column is identified as the target variable, and the remaining columns are considered features.

3. **Identifying Categorical Variables**: Features such as `artists`, `album_name`, `track_name`, `explicit`, and `track_genre` are identified as categorical.

4. **Encoding Categorical Variables**:
   - **One-Hot Encoding** is applied to convert categorical variables into a format that can be provided to ML algorithms to do a better job in prediction.
   - **Drop First Category**: To prevent multicollinearity, one category from each feature is dropped.

5. **Combining Data**: The encoded categorical variables are concatenated with the numerical features to form the final feature set.

**Outcome**: A clean and preprocessed dataset ready for modeling.

---

## Task 1: Data Splitting

### Purpose

To evaluate the performance of the models on unseen data, the dataset is split into training and test sets.

### Analysis

- **Train-Test Split**: The data is split into 80% training and 20% test sets using a fixed random state for reproducibility.
- **Reasoning**: An 80:20 split is standard practice, providing sufficient data for training while reserving enough data to test the model's generalization.

**Outcome**: Two datasets (`X_train`, `y_train`) and (`X_test`, `y_test`) for training and evaluating models.

---

## Task 2: Training Models without Categorical Features

### Purpose

To assess the predictive power of numerical features alone by training various regression models.

### Analysis

1. **Feature Selection**: Only numerical features are selected to simplify the model and focus on quantitative data.

2. **Data Scaling**:
   - **Standardization**: Numerical features are scaled using StandardScaler to have a mean of zero and a standard deviation of one.
   - **Importance**: Scaling ensures that all features contribute equally to the result and improves the convergence of gradient-based methods.

3. **Model Training**:
   - **Ordinary Least Squares (OLS)**: Provides a benchmark for linear regression models.
   - **Linear Regression**: Uses the normal equation to find the best-fitting line.
   - **Ridge Regression**: Introduces L2 regularization to prevent overfitting by penalizing large coefficients.
   - **Lasso Regression**: Uses L1 regularization for feature selection by potentially shrinking some coefficients to zero.

4. **Model Evaluation**:
   - **RMSE (Root Mean Squared Error)**: Measures the model's prediction error magnitude.
   - **R² Score**: Indicates the proportion of variance in the dependent variable predictable from the independent variables.

**Findings**:

- OLS, Linear Regression, and Ridge Regression performed similarly with low RMSE and high R² values.
- Lasso Regression showed higher RMSE and lower R², suggesting over-regularization or that Lasso may not be suitable with the given `alpha` value.

---

## Task 3: Training Models with Categorical Features

### Purpose

To determine if including categorical features improves the model's predictive performance.

### Analysis

1. **Combining Features**: One-hot encoded categorical features are added to the numerical features.

2. **Data Scaling**: The combined features are scaled to maintain consistency.

3. **Model Training and Evaluation**: The same models as in Task 2 are retrained using the extended feature set.

**Findings**:

- The inclusion of categorical features did not significantly improve model performance.
- RMSE and R² values remained almost the same, indicating that categorical variables might not contribute substantially to predicting `energy` in this context.

---

## Task 4: Examination of Model Parameters

### Purpose

To understand the influence of each feature on the target variable by examining model coefficients.

### Analysis

1. **OLS Model**:
   - All coefficients are statistically significant, with p-values close to zero.
   - No coefficients are zero, indicating that all features contribute to the model.

2. **Ridge Regression**:
   - Coefficients are shrunk but not eliminated due to L2 regularization.
   - No coefficients are exactly zero.

3. **Lasso Regression**:
   - Some coefficients are reduced to zero, effectively performing feature selection.
   - Indicates which features are less important in predicting the target variable.

**Conclusion**:

- OLS and Ridge Regression models use all features, while Lasso simplifies the model by excluding less significant ones.
- Regularization impacts the model coefficients, and Lasso can be used for feature selection.

---

## Task 5: Feature Selection Using Backward Elimination

### Purpose

To improve model performance by removing insignificant features through backward elimination.

### Analysis

1. **Backward Elimination Process**:
   - Starts with all features and iteratively removes the least significant feature (highest p-value above a threshold).
   - Repeats until all remaining features have p-values below the significance level (e.g., 0.05).

2. **Application**:
   - Applied to the training data with categorical features included.
   - No features were removed, as all had p-values below the threshold.

**Conclusion**:

- All features are statistically significant predictors of the target variable.
- Backward elimination did not simplify the model further in this case.

---

## Task 6: Hyperparameter Tuning for Lasso Regression

### Purpose

To find the optimal `alpha` value for Lasso Regression to balance bias and variance.

### Analysis

1. **Cross-Validation**:
   - Used 4-fold cross-validation to evaluate model performance across different splits.
   - Provides a robust estimate of model performance.

2. **Alpha Range**:
   - Tested `alpha` values on a logarithmic scale from \(10^{-4}\) to \(10^{3}\).
   - Ensures a comprehensive search over possible regularization strengths.

3. **Model Evaluation**:
   - RMSE was used as the evaluation metric to assess prediction error.
   - The optimal `alpha` minimizes the RMSE.

**Findings**:

- The best `alpha` was found to be approximately `0.21`.
- This value provided a balance between model complexity and predictive performance.

---

## Task 7: Implementing Gradient Descent Variants for Ridge Regression

### Purpose

To implement and compare different optimization algorithms for training a Ridge Regression model.

### Analysis

1. **Gradient Descent Methods**:
   - **Full Gradient Descent**: Updates weights using the entire dataset.
   - **Stochastic Gradient Descent (SGD)**: Uses random subsets (mini-batches) for updates.
   - **Momentum**: Accelerates gradient descent by considering past gradients.
   - **Adagrad**: Adapts the learning rate based on past gradients.

2. **Implementation Details**:
   - Vectorized computations for efficiency.
   - Proper initialization of weights and hyperparameters.
   - Implementation of stopping criteria based on tolerance and maximum iterations.

3. **Regularization**:
   - Ridge regularization is incorporated into the loss function and gradient calculations to prevent overfitting.

**Outcome**:

- A flexible Ridge Regression model capable of using different gradient descent optimization methods.
- Prepared for experimentation and analysis in subsequent tasks.

---

## Task 8: Training and Validating Manual Models

### Purpose

To compare the manually implemented Ridge Regression model with established models and investigate the impact of hyperparameters.

### Analysis

1. **Model Comparison**:
   - The manual model was compared against `sklearn`'s Linear Regression and `statsmodels`' OLS.
   - Performance metrics (RMSE and R²) were used for evaluation.

2. **Hyperparameter Investigation**:
   - **max_iter**:
     - Tested different values to observe the effect on convergence and performance.
     - Found that beyond a certain point, increasing `max_iter` did not significantly improve the model.
   - **alpha (Regularization Strength)**:
     - Explored various `alpha` values to see the impact on overfitting and underfitting.
     - Identified optimal `alpha` that balances bias and variance.

**Conclusion**:

- The manual model's performance was comparable to established models.
- Proper tuning of hyperparameters is crucial for optimal model performance.

---

## Task 9: Analyzing Convergence Rates of Gradient Descent Methods

### Purpose

To visualize and compare the efficiency of different gradient descent algorithms in optimizing the Ridge Regression model.

### Analysis

1. **Loss Function Tracking**:
   - Recorded the loss function value at each iteration for each optimization method.
   - Provided data for plotting and analysis.

2. **Methods Compared**:
   - **Full Gradient Descent**
   - **Stochastic Gradient Descent**
   - **Momentum**
   - **Adagrad**

3. **Findings**:
   - **Adagrad**:
     - Converged the fastest due to adaptive learning rates.
     - Effective when dealing with sparse data or varying gradients.
   - **Momentum**:
     - Accelerated convergence by incorporating momentum from previous updates.
     - Reduced oscillations and improved stability.
   - **Stochastic Gradient Descent**:
     - Faster iterations but with more fluctuations in the loss due to randomness.
     - May require more iterations to converge.
   - **Full Gradient Descent**:
     - Slowest convergence as it processes the entire dataset each time.
     - Provided a smooth and stable decrease in the loss function.

**Conclusion**:

- Advanced optimization methods like Adagrad and Momentum significantly improve convergence rates.
- The choice of optimization algorithm can impact the training time and efficiency of the model.

---

## Conclusion

- **Data Preprocessing**: Proper handling of categorical variables and feature scaling is essential for model performance.
- **Model Evaluation**: Comparing different models and understanding their parameters helps in selecting the appropriate approach.
- **Feature Selection**: Techniques like Lasso regression and backward elimination can simplify models without sacrificing performance.
- **Optimization Methods**: Implementing and analyzing various gradient descent algorithms reveals their strengths and applicability.
- **Hyperparameter Tuning**: Adjusting parameters like `alpha` and `max_iter` is crucial for balancing bias and variance.



