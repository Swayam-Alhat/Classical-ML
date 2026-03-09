import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("./Linear-Regression/house_prediction_data.csv")

# select important features
df = df[["Area","Bedrooms","Garage","Price"]]

# shuffle data So we get different data in both training & testing dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# check if null value is present
is_null = df.isnull().any().any()
print(is_null) # False

# check if any string value is present and convert it
print(df.dtypes)

# split data into training and testing dataset
train_df = df.iloc[0:1600,:]
test_df = df.iloc[1600:, :]
# check start and end rows of both dfs using head() and tail()





