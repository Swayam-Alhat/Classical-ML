# What is Linear Regression

Linear Regression is supervised ML algorithm. Meaning it learns from labeled data.

Its main goal is to find best fit straight line on graph of training data. That is, find the linear equation (represented in slope intercept form i.e y = mx + b) which calculates/predicts y for training data. And then compare actual y with predicted y so to calculate MSE and further Gradient which helps to update m and b.

> Note : Actually m and b values are updated at each iteration.

## Explaintion

Training data :

| House size (100 sqft) | House price (lakh) |
| --------------------- | ------------------ |
| 1                     | 2                  |
| 2                     | 4                  |
| 3                     | 6                  |

So, algorithm aims to find the linear equation which can predict house price for new houses (new unseen data). As we know linear equation can be represented using slope intercept form i.e `y = mx + b`

Now, if we plot the graph of above data, we will have house size on X-axis & house price on Y-axis.

So, in `y = mx + b`,  
y is any point on Y-axis  
x is any point on X-axis  
m is slope (how much y changes when x increases by 1 unit)  
b is y-intercept ( point on y-axis where straight line (formed using linear equation) cross/intercept Y-axis)

So now we have equation `y = mx + b` where y will be house price and x will be that house's size.

_So algorithm need to find optimal value of m and b so that it outputs correct value of y (house price) for its corresponding x (house size)_

Example, we take m = 0, b = 0  
let's select 1st data point
x = 1 (100 sqft)
y = 2 (lakh)  
So, data point is (x,y) = (1,2)

Put all values in equation,

y = mx + b  
2 = 1(0) + 0  
2 != 0

So when we take m and b as 0 for x = 1 (i.e house size 1 (100 sqft)), equation gives y = 0 (i.e 0 lakh), which is incorrect prediction. Because Actual value of y (house price) is 2 (in lakh) when x (house size) is 1 (100 sqft)

So, algorithm initially randomly sets value of m and b, and predict value of y for each x. That is

let's say, it sets m = 0, b = 0  
So, it puts this values in equation `y = mx + b` and predicts y (house price) for each x (house size).

| House size (X) (100 sqft) | House price (Y) (lakh) | predicted value of y | Error     |
| ------------------------- | ---------------------- | -------------------- | --------- |
| 1                         | 2                      | 0                    | 2 - 0 = 2 |
| 2                         | 4                      | 0                    | 4 - 0 = 4 |
| 3                         | 6                      | 0                    | 6 - 0 = 6 |

Error = Actual value - predicted value

Calculate MSE (Mean Square Error) = ( $(2)^{2}$ + $(4)^{2}$ + $(6)^{2}$ ) / 3

MSE = 18.67

So, 18.6 is error. Algorithms wants to update value of m and b in such way that MSE reduces. So, algorithm needs to know how m and b influences MSE. That is,

```
If I change m by tiny amount, how much does MSE change?
```

```
This rate of change is called the derivative.
```

```
Gradient tells us how much the MSE changes when we slightly change m or b.
```

_This is how Gradient Descent comes in picture_

That is called derivative from calculus

```
   Gradient of m = d(MSE)/dm
   Gradient of b = d(MSE)/db
```

When you solve this derivative mathematically, you get:

```
Gradient of m = (-2/n) × Σ(actual - predicted) × x
Gradient of b = (-2/n) × Σ(actual - predicted)
```

So, here we calculate gradient of m and b.  
This helps algorithm to know how m and b influences MSE and allows to update value of m and b using formula

```
m = m - (α × gradient of m)
b = b - (α × gradient of b)
```

where `α` is learning rate (Its a hyperparameter and heavily influences model training, so we should select appropriate learning rate).  
here α is 0.01

Now , algorithm again predicts value of y (house price) for each x (house size) using updated values of m and b.

Now we get MSE = 14.74

So, MSE decreased. Again algorithm -

1. calculate gradient of m and b
2. update value of m and b
3. predicts y using updated values
4. Calculate errors and MSE
5. Again MSE is decreased

This steps are repeated until MSE remain unchange or changes very slightly. This is where we get optimal value of m and b

_Now this optimal values of m and b are set. And using this values, algorithms predicts y (house price) for new x values (new house size)_

> **Linear Regression finds optimal m and b by minimizing MSE using Gradient Descent. Once trained, it predicts y for any new x using y = mx + b.**
