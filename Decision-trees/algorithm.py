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
max_depth = 3
depth = 0
nodes = []

# Calculate split values
splits = []

# iterate all columns. except last one (target column)
for feature in training_data.columns[:-1]:

    # find unique values and sort them to calculate split values
    sorted_feature = np.sort(training_data[feature].unique())

    # calculate split value
    split_arr = (sorted_feature[:-1] + sorted_feature[1:]) / 2
    
    # add in splits list
    splits.append(split_arr)


# calculate gini and find best split values
best_gini_list = []
best_split_list = []
for i in range(len(splits)):

    gini_arr = []
    # iterate each split value
    for val in splits[i]:

        # get left node values
        left_node_values = training_data.loc[training_data.iloc[:,i] <= val , training_data.columns[-1]].to_numpy()

        # get right node values
        right_node_values = training_data.loc[training_data.iloc[:,i] > val , training_data.columns[-1]].to_numpy()

        # defence check if node values are empty
        if len(left_node_values) == 0 or len(right_node_values) == 0:
            gini_arr.append(1.0)
            # skip further execution
            continue

        # calculate gini for left node
        p0_for_left = len(left_node_values[left_node_values == 0]) / len(left_node_values)

        p1_for_left = len(left_node_values[left_node_values == 1]) / len(left_node_values)

        left_node_gini = 1 - ((p0_for_left**2) + (p1_for_left**2))



        # calculate gini for right node
        p0_for_right = len(right_node_values[right_node_values == 0]) / len(right_node_values)

        p1_for_right = len(right_node_values[right_node_values == 1]) / len(right_node_values)

        right_node_gini = 1 - ((p0_for_right**2) + (p1_for_right**2))


        # calculate weighted gini
        weighted_gini = ((len(left_node_values) / len(training_data.iloc[:,i])) * left_node_gini) + ((len(right_node_values) / len(training_data.iloc[:,i])) * right_node_gini)

        gini_arr.append(weighted_gini)
    
    # get best gini, its index and split which produced it
    best_gini = min(gini_arr)
    best_split_index = gini_arr.index(best_gini)
    best_split = splits[i][best_split_index]

    # add best gini and split in their arrays
    best_gini_list.append(best_gini)
    best_split_list.append(best_split)

# get best gini
final_best_gini = min(best_gini_list)
# get best gini's index
final_best_gini_index = best_gini_list.index(final_best_gini)
# get best split value using best gini's index
final_split_value = best_split_list[final_best_gini_index]
# get index of best split value
final_split_value_index = best_split_list.index(final_split_value)
# get the feature of that split
best_feature = training_data.columns[final_split_value_index]

# store formed nodes
nodes.append({best_feature:final_split_value})

# increase depth
depth = depth + 1
