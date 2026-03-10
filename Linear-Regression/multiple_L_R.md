# Multiple Linear Regression

## Explaination about why we need feature scaling

> Explaination is based around linear regression algorithm. i.e Why **feature scaling** is important specifically for **Linear Regression**

In simple linear regression, we can find an appropriate learning rate based on our single feature's range. Too high and the gradient explodes, too low and learning is painfully slow — but at least one good value exists.

The problem arises when features have different scales. Since gradient magnitude depends directly on feature values, features with large ranges produce large gradients while features with small ranges produce small ones. Applying the same learning rate to these mismatched gradients means it will either be too large for some features (causing their weights to overshoot and diverge) or too small for others (causing them to crawl toward the minimum). There is no single learning rate that works well for all features simultaneously.

Feature scaling solves this by bringing all features into the same range, making their gradients comparable in magnitude. Now a single learning rate works properly across all weights, and gradient descent converges smoothly and efficiently.

#### Summary

Gradient magnitude depends on feature range → different features produce different magnitude gradients → one learning rate can't handle all of them properly → either overshoots for some or crawls for others → scaling fixes this by bringing all gradients into comparable range
