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
- Typical approach: try multiple values of k and pick the one with best validation performance (e.g. using cross-validation)
