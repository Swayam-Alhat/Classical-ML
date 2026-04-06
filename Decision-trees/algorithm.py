import pandas as pd
import numpy as np

df = pd.read_csv("./Logistic-Regression/diabetes-dataset.csv")

# remove unwanted features
df = df.drop(["Pregnancies","DiabetesPedigreeFunction","BMI"], axis=1)

# removed rows of each columns whose values were 0
df = df[df["Glucose"] != 0]
df = df[df["BloodPressure"] != 0]
df = df[df["SkinThickness"] != 0]

# update index (reset index)
df.index = range(len(df))

# df[df['Glucose'] != 0] — the inner part df['Glucose'] != 0 returns True/False for each row. Then df[...] keeps only the rows where it's True. So you're basically saying "give me only rows where Glucose is not zero.

# training data
training_data = df.iloc[:373, :]

# implementation 
splits = []
for i in range(training_data.shape[1] - 1):
    unique_values = np.sort(training_data.iloc[:,i].unique())
    split = (unique_values[:-1] + unique_values[1:]) / 2
    splits.append(split)

# Now, splits contains array which contains split values for each feature. AT index 0, Glucose split values, At index 1, BloodPressure split values and so on


gini = []

index = 0
for split in splits:

    gini_of_column = []
    for i in split:

        # get left node values
        left_node = training_data.loc[training_data.iloc[:,index] <= i, training_data.columns[-1]].to_numpy()
        # get right node values
        right_node = training_data.loc[training_data.iloc[:,index] > i, training_data.columns[-1]].to_numpy()

        # defence if any of above node is empty
        # this will add gini as 1. i.e highest gini. SO it will be not picked as good gini
        if len(left_node) == 0 or len(right_node) == 0:
            gini_of_column.append(1.0)
            continue

        # left node gini
        left_node_gini = 1 - (((len(left_node[left_node == 0]) / len(left_node))**2) + ((len(left_node[left_node == 1]) / len(left_node))**2))

        # right node gini
        right_node_gini = 1 - (((len(right_node[right_node == 0]) / len(right_node))**2) + ((len(right_node[right_node == 1]) / len(right_node))**2))

        # weighted gini
        weighted_gini = ((len(left_node) / (len(left_node) + len(right_node))) * left_node_gini) + ((len(right_node) / (len(left_node) + len(right_node))) * right_node_gini)

        gini_of_column.append(weighted_gini)
    
    index = index + 1
    gini.append(gini_of_column)

# we get gini array
# Now find best gini, and then split

best_splits = []
best_gini = []

for idx, val in enumerate(gini):

    # find min gini and store it
    min_gini = np.min(val)
    best_gini.append(min_gini)
    
    # find index of min gini and then best split
    best_split = splits[idx][np.argmin(val)]
    best_splits.append(best_split)



