import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("./Linear-Regression/House_dataset.csv")

# select important features
df = df[["area","bedrooms","price"]]

# shuffle data So we get different data in both training & testing dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# check if null value is present
is_null = df.isnull().any().any()
print(is_null) # False

# EDA

# check if house size have linear relationship with house price
# plt.scatter(df["area"],df["price"])
# plt.xlabel("house size")
# plt.ylabel("house price")
# plt.show()

# correlation_coefficient = df["area"].corr(df["price"], method="pearson")
# print(correlation_coefficient) # 0.53 this means Moderate linear relationship.

# check how house price influences for no. of bedrooms
# plt.bar(df["bedrooms"],df["price"])
# plt.xlabel("bedrooms")
# plt.ylabel("house price")
# plt.show()


# split data into training and testing
train_df = df.iloc[0:431,:]
test_df = df.iloc[431:,:]

# Note - First split data into training and testing. then feature scaling
# feature scaling for training data - standardization

x1_mean = train_df["area"].mean()
x1_std = train_df["area"].std()

x2_mean = train_df["bedrooms"].mean()
x2_std = train_df["bedrooms"].std()

train_df["area"] = (train_df["area"] - x1_mean) / x1_std
train_df["bedrooms"] = (train_df["bedrooms"] - x2_mean) / x2_std

# print(df["area"].mean())
# print(df["bedrooms"].mean()) 
# print(df["area"].std())
# print(df["bedrooms"].std())


# actual implementation of algorithm

learning_rate = 0.01
m1 = 0
m2 = 0
b = 0
N = len(train_df)
MSEs = []

epoch = 1000


for i in range(epoch):

    # predict y for each x
    # it returns new series object
    predicted_y = (m1 * train_df["area"]) + (m2 * train_df["bedrooms"]) + b

    # calculate error
    errors = train_df["price"] - predicted_y

    # calculate MSE
    MSE = (sum(errors ** 2)) / len(errors)
    MSEs.append(MSE)

    # calculate gradient
    grad_m1 = (-2 / N) * (sum(errors * train_df["area"]))
    grad_m2 = (-2 / N) * (sum(errors * train_df["bedrooms"]))
    grad_b = (-2 / N) * sum(errors)

    # update m1, m2 and b
    m1 = m1 - learning_rate * grad_m1
    m2 = m2 - learning_rate * grad_m2
    b = b - learning_rate * grad_b

print(MSEs[0]) # first MSE
print(MSEs[-1]) # Last MSE

# plot graph of MSEs
epochs = np.arange(1,epoch + 1)
plt.plot(epochs,MSEs)
plt.show()

# plot regression line
plt.scatter(train_df["area"],train_df["price"], color="red")
plt.plot(train_df["area"],predicted_y, color="black")
plt.show()


# Testing
# first scale testing data using same stats from training data i.e mean and std
# because as we know, we got optimal values of m1, m2 and b using training data which was scaled using its own  mean and std. So to get accurate prediction, we need scale testing data using training data stats (i.e mean and std)

# scaling testing data with mean and std of training data
test_df["area"] = (test_df["area"] - x1_mean) / x1_std
test_df["bedrooms"] = (test_df["bedrooms"] - x2_mean) / x2_std

# So now testing data is in same range where training data was. So it can predict accurate values since optimal values of m1,m2 and b was found using training data of same scale. SO we have to bring input data / testing data in same scale and then predict values.

# So in production we have store mean and std of training data so that we can use it to scale input data in same range in which training data was.
# Example, if we get input house size = 1500 sqft & 2 bedrooms, then we have scale both of these values in same range in which training data was. Thats why we use mean and std of training data so the input data also comes in same range and then prediction is done

# actual test
input_house_size = test_df.iloc[0,0]
print(f"input house size : {input_house_size}")

input_bedrooms = test_df.iloc[0,1]
print(f"input number of bedrooms : {input_bedrooms}")

# actual value from testing data
print(f"Actual price : {test_df.iloc[0,2]}")

prediction = (m1 * input_house_size) + (m2 * input_bedrooms) + b

print(f"Prediction : {prediction}")
print(f"Actual : {test_df.iloc[0,2]}")