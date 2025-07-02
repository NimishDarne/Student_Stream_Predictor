# Student_Stream_Predictor
This project uses machine learning to predict which academic stream (Science, Commerce, Arts, or Diploma) a student is likely to choose after 10th grade, based on historical data and engineered features.
The model developed uses **XGBoost Classifier**, a powerful gradient boosting algorithm, and has been fine-tuned using `RandomizedSearchCV` with early stopping for optimal performance.

Problem Statement
Choosing the right academic stream after the 10th standard is a crucial decision for students. This model helps guide that choice using patterns in past data, hobbies, marks consistency, and other factors.

Features
The dataset includes a variety of features that contribute to predicting student performance. Key features include:

* **Demographic Information:** Gender, age etc.
* **Academic Background:** Previous educational attainment, past grades etc.
* **Behavioral Factors:** Extracurricular activities, etc.

This project follows a standard machine learning workflow, all implemented within a single Python script:
1.  **Data Loading:** The raw student dataset is loaded.
2.  **Data Preprocessing:** This involves:
    * Encoding categorical features (e.g., One-Hot Encoding).
    * Scaling numerical features.
    * Feature engineering: Creating new, more informative features from existing ones.
3.  **Model Selection:** XGBoost Classifier was chosen due to its robust performance, ability to handle various data types, and strong predictive power in classification tasks.
4.  **Hyperparameter Tuning:** `RandomizedSearchCV` was employed to efficiently search for the optimal combination of XGBoost hyperparameters. Early stopping was integrated into the tuning process to prevent overfitting and speed up training.
    * `n_estimators`: Number of boosting rounds.
    * `learning_rate`: Step size shrinkage to prevent overfitting.
    * `max_depth`: Maximum depth of a tree.
    * `subsample`: Subsample ratio of the training instance.
    * `colsample_bytree`: Subsample ratio of columns when constructing each tree.
    * `gamma`: Minimum loss reduction required to make a further partition.
    * `reg_lambda`: L2 regularization term.
    * `min_child_weight`: Minimum sum of instance weight needed in a child.
5.  **Model Evaluation:** The model's performance was evaluated using metrics such as `Log Loss`, `Accuracy`, `Precision`, `Recall`, and `F1-Score` on an unseen test set.
6.  **Model Saving:** The best trained model and the fitted data preprocessor are saved for future use.



