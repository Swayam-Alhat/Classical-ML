# ML From Scratch

Implementing Machine Learning algorithms from scratch in Python
— without using ML libraries like Scikit-learn.

## Algorithms

- [ ] Linear Regression
- [ ] Logistic Regression
- [ ] Decision Trees
- [ ] KNN
- [ ] K-means clustering

## Foundation

`AI` ✅ — AI is technology that involves machines which simulate/mimics human intelligence.

`Machine Learning` ✅ — ML is a subset of AI that allows machines to learn from data. It involves using ML algorithms to learn patterns from data and make decisions/predictions on new, unseen data.

`ML Algorithms` ✅ — It's the step-by-step procedure/program that learns patterns from data.

`ML Models` ✅ — A trained algorithm becomes a model.

**So This algorithms are foundation of Machine learning.** All things are built on top of this.

## How this algorithms were built

This algorithms were built by mathematicians, statisticians & computer scientists. They just applied math concepts like statistics, calculus, etc in computers. So basically they founded way to make computer learn from data using math. Thats how ML was born.

> Linear Regression (Simplest ML algorithm) was invented by Carl Friedrich Gauss & Francis Galton in the 1800s — originally a statistics concept, later applied to computers.

## Types of ML algorithms

ML algorithms are step by step procedure or mathematical procedures that allows machine to learn patterns from data and make predictions on new data.

The outcome of applying a machine learning algorithm to a dataset is a trained model.

**This alogorithms are mainly divided into 3 types -**

### 1. Supervised ML algorithm

**This algorithms uses labeled data to learn patterns**. This Supervised ML algorithms are used for `Classification` (predict category) & `Regression` (predict numbers).

For Classification, following algorithms are used -

1. Logistic Regression
2. K-Nearest Neighbors - KNN
3. Naive Bayes
4. Support Vector Machine
5. Random Forest

For Regression, following algorithms are used -

1. Linear Regression
2. Polynomial Regression

### 2. Unsupervised ML algorithm

**This algorithms uses unlabeled data to train.** This Unsupervised ML algorithms are used in `Clustering` (grouping similar data) & `Dimensionality Reduction` (Simplifying data).

For Clustering, following algorithms are used -

1. K-Means Clustering
2. Hierarchical Clustering

For Dimensionality Reduction, following algorithms are used -

1. PCA - Principal Component Analysis
2. Autoencoders

### 3. Reinforcement Learning Algorithms

**This algorithms learn by trials and errors**. Agent learns by interacting with environment — gets reward for good actions, penalty for bad ones.  
Following are Reinforcement algorithms -

1. Q-Learning
2. Deep Q-Network - DQN

> For ML algorithms, check the individual folders — each has a README and implementation.

## EDA (Exploratory Data Analysis)

Before even implementing the ML algorithms, we need to explore and analyze data. We should check if there are null values in dataset. We should also understand data because understanding data directly influences the selection of ML algorithm.

Example, lets say we want to create ML system which predicts house price. If data is linear i.e features (like house size, no. of bedrooms, etc) have linear relationship with target (house price), then Linear Regression will be best choice for building such model.

The best way to understand and analyze data is data visualization. So for this, we should know about graphs.

---

### What is a graph actually?

A graph is just a **visual representation of data**. That's it.

Instead of looking at 1000 rows of numbers and trying to make sense of them, you convert them into a visual form that your brain can process instantly.

```
Numbers  →  hard for brain to see patterns
Graph    →  brain sees patterns instantly
```

That's the only purpose of a graph.

---

### Why are there different types of graphs?

Because data has different **types and relationships**. Each graph is designed to answer a specific question about your data.

Think of it like tools. You don't use a hammer to cut wood. Same way you don't use a scatter plot to show distribution.

---

### The main graph types and when to use them

**Scatter Plot**

- Question it answers: _"Is there a relationship between two numerical values?"_
- X axis → one numerical feature, Y axis → another numerical value
- Use it when both X and Y are continuous numbers
- Example: house size vs price

```
When to use: checking linearity, correlation between two numbers
```

**Line Chart**

- Question it answers: _"How does a value change over time?"_
- X axis is always time
- Example: stock price over months, model loss over epochs

```
When to use: anything involving time or sequence
```

**Bar Chart**

- Question it answers: _"How do values compare across categories?"_
- X axis → categories, Y axis → numerical value
- Example: average house price for each number of bedrooms

```
When to use: comparing a numerical value across categories
```

**Histogram**

- Question it answers: _"How is a single numerical feature distributed?"_
- Looks like a bar chart but X axis is numerical ranges (called bins)
- Example: how many houses fall in each price range

```
When to use: understanding spread, skew, outliers of one feature
```

**Box Plot**

- Question it answers: _"What is the spread and are there outliers?"_
- Shows min, max, median, and outliers in one plot

```
When to use: detecting outliers, comparing spread across categories
```

---

### Quick reference — which graph for which situation

| X axis    | Y axis    | Use          |
| --------- | --------- | ------------ |
| Numerical | Numerical | Scatter plot |
| Time      | Numerical | Line chart   |
| Category  | Numerical | Bar chart    |
| Numerical | Count     | Histogram    |
| Category  | Spread    | Box plot     |

---

### Specifically in ML, graphs are used for

- **EDA** — understanding your data before modeling
- **Checking assumptions** — like linearity before linear regression
- **Detecting outliers** — before scaling
- **Monitoring training** — loss curve over epochs (line chart)
- **Evaluating model** — predicted vs actual (scatter plot)

---
