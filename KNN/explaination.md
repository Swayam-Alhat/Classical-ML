# K-Nearest Neighbors (KNN)

**_KNN_** is a supervised ML algorithm. It is used in both **Classification** and **Regression**.

KNN works by finding the k nearest data points and predicts a value based on those data points.

**_It is different from other ML algorithms because it is a lazy learner — it has no training phase at all. All computation happens at prediction time._**

- It calculates the distance between a new data point and all other data points
- Finds the k nearest neighbors
- Uses those k neighbors to predict the class or continuous value

---

## Distance Metric

KNN uses a **distance metric** to find the nearest neighbors. The most common is **Euclidean distance**, but others can be used too:

| Metric    | Use case                                |
| --------- | --------------------------------------- |
| Euclidean | Most common, continuous features        |
| Manhattan | When outliers are a concern             |
| Minkowski | Generalization of Euclidean & Manhattan |
| Hamming   | Categorical features                    |

---

## Classification

Predicts the class which is **most frequent** among the k nearest neighbors.

---

## Regression

Predicts a continuous value by taking the **average** of the values of the k nearest neighbors.

This can also be a **weighted average** — closer neighbors get more influence than farther ones.

---

## Choosing k (Hyperparameter)

`k` is a hyperparameter — it is not learned, it is chosen by you before training.

- **Small k** → sensitive to noise, overfits
- **Large k** → smoother predictions, but may miss local patterns

#### Methods to select optimal k

**Cross-Validation** - A commonly used technique for choosing the value of k is cross-validation. By splitting the dataset into training and validation sets and evaluating model performance across different values of k, the optimal k value can be determined based on which k produces the lowest error rate.

**Common Values of k** - In practice, values of k such as 3, 5, or 7 are typically chosen. Smaller values of k allow the model to capture local patterns, while larger k values generalize better across the dataset.

## Implementation

For now, we are using **Pima Indians Diabetes dataset** to predict whether new patient is **diabetic** or **non-diabetic**.

Since we have feature values as continuous values, we are using **_Euclidean distance metric_**.

> [!NOTE]
> Read and understand how Euclidean distance is calculated for data points.

After understanding how Euclidean distance is calculated, we will know, if feature values are in different ranges, they affect the distance calculation.  
Example,  
Glucose ranges from ~0–200, while BMI ranges from ~18–50. This means Glucose differences dominate the distance calculation simply because its scale is larger — not because it's more important

Thats why we need Feature scaling,  
So before implementing, perform feature scaling on training and testing dataset

> [!NOTE]
> Use the mean and standard deviation of training data to scale testing data

## Choosing k — Manual Selection

Since this is a learning implementation, we manually tested k = 3, 5, and 7 on the Pima Indians dataset:

| k   | Correct | Accuracy |
| --- | ------- | -------- |
| 3   | 119/161 | 73%      |
| 5   | 124/161 | 77%      |
| 7   | 123/161 | 76%      |

**k = 5** gave the best accuracy and was selected as the final value.
