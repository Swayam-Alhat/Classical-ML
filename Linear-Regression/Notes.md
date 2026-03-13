# Notes while Implementing Linear Regression

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

## Check if data is linear

Before getting into feature scaling, we should make sure that our training data is linear. i.e does each feature have a linear relationship with target.

Linear Regression should be used for learning only when training data is linear i.e features have linear relationship with target.

Example, if we have features _house size_ & _number of bedrooms_ for target _house price_, we should check if both features i.e _house size_ and _number of bedrooms_ have linear relationship with target i.e _house price_.

This is done by plotting graph (scatter plot) of each feature against target. If you see a roughly straight line trend, the relationship is linear. If it's curved or random, it's not.

```
   x-axis → feature (e.g. house size)
   y-axis → target (house price)
```

## Why we need Feature Scaling

> Explaination is based around linear regression algorithm. i.e Why **feature scaling** is important specifically for **Linear Regression**

In simple linear regression, we can find an appropriate learning rate based on our single feature's range. Too high and the gradient explodes, too low and learning is painfully slow — but at least one good value exists.

The problem arises when features have different scales. Since gradient magnitude depends directly on feature values, features with large ranges produce large gradients while features with small ranges produce small ones. Applying the same learning rate to these mismatched gradients means it will either be too large for some features (causing their weights to overshoot and diverge) or too small for others (causing them to crawl toward the minimum). There is no single learning rate that works well for all features simultaneously.

Feature scaling solves this by bringing all features into the same range, making their gradients comparable in magnitude. Now a single learning rate works properly across all weights, and gradient descent converges smoothly and efficiently.

#### Summary

Gradient magnitude depends on feature range → different features produce different magnitude gradients → one learning rate can't handle all of them properly → either overshoots for some or crawls for others → scaling fixes this by bringing all gradients into comparable range

> There are several ways to scale features. **Standardization** is used as it does not get affected by outliers.
