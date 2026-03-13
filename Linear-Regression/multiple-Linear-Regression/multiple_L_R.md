# Multiple Linear Regression

define multiple linear regression

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
