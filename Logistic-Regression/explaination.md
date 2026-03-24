# Explanation of Logistic Regression

Logistic Regression is ML algorithm used for classification. Unlike linear regression which predicts continuous values it predicts the probability that an input belongs to a specific class.

Example, to predict if email is spam or not spam.

_Since, probability lies between 0 and 1. we need to calculate value of y (target variable) which lies within 0 and 1._

Thus, we use **sigmoid function** which outputs value between 0 and 1. Sigmoid function is also known as Logistic function.

## Derivation of Sigmoid function

We cannot use linear function  
 y = $b_{0}$ + $b_{1}$ . $x_{1}$  
because this outputs value of y (target variable) within (-∞ , ∞). But Logistic regression predicts probability of outcome and we know probability lies between 0 and 1. So we work with _odds_.

- _Probability_ : Probability measures the chance of an event occurring out of all possible outcomes.

![probability-figure](./figures/probability-fig.png)

- _Odds_ : Odds compare the chance of an event occurring to the chance of it not occurring.

![odds-figure](./figures/odds-figure.png)

> In logistic Regression, we work with odds since we have to predict if outcome is favourable or not.

So, if we take odds of input x

$$
\text{odds(x)} =\frac{p(x)}{1 - p(x)}
$$

Odds ranges from 0 to ∞.

- An odds value greater than 1 indicates a favorable outcome. means p > 0.5, i.e. the event is more likely than not
- less than 1 indicates an unfavorable outcome. Means the event is less likely than not.
- equal to 1 means the event is just as likely to occur as not.

_To address this imbalance, we take the logarithm of the odds, which transforms range of odds from (0 ,∞) to the real number line (−∞, ∞). This is known as the **log-odds**, or logit, and is the foundation of the logistic regression model_

Since, Log-odds outputs value within (−∞, ∞), we can say Linear function y = $b_{0}$ + $b_{1}$ $x_{1}$ is equal to log-odds.

> Because $b_{0}$ + $b_{1}$ . $x_{1}$ and log-odds outputs value which lies within (−∞, ∞).

$ \text{log} \text{(}\frac{p(x)}{1 - p(x)} \text{)}$ = $b_{0}$ + $b_{1}$. $x_{1}$

We can then exponentiate both sides to get back to odds:

$ \frac{p(x)}{1 - p(x)}$ = $e^{b_0 + b_1 . x_1}$

Solving for `p(x)`we get the sigmoid function, which helps ensure the predicted value stays between 0 and 1:

p(x) = $ \frac{e^{b_0 + b_1 . x_1}}{1 + e^{b_0 + b_1 . x_1}} $

> [!NOTE]  
> The above form is equivalent to the more common compact form:
> $$p(x) = \frac{1}{1 + e^{-(b_0 + b_1 x_1)}}$$
> Multiply numerator and denominator by $ e^{-(b_0 + b_1 . x_1)} $ and solve it. We will get commonly used Sigmoid function form

This transformation allows logistic regression to output valid probabilities, even though we’re modeling them using a linear function underneath.
