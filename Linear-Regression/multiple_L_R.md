# Multiple Linear Regression

## Explaination about why we need feature scaling

> Explaination is based on linear regression algorithm because that involves using gradient descent

In simple linear regression, we can find an appropriate learning rate based on our single feature's range. Too high and the gradient explodes, too low and learning is painfully slow — but at least one good value exists.

In multiple linear regression, each feature has its own range. Since all weights update using the same learning rate, a rate that works for one feature will be too large or too small for another — causing some gradients to explode while others crawl. There's no single learning rate that works well for all of them.

Feature scaling solves this by bringing all features into the same range, so one learning rate works properly across all of them, and gradient descent converges smoothly.
