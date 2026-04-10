import pandas as pd
import numpy as np

import numbers

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

# testing data
testing_data = df.iloc[373:,:]

# check if data is pure.(95% of values in target columns are of same class)
def is_pure_enough(data):
    most_frequent_value = data.iloc[:,-1].value_counts().idxmax()
    ocurrence = data.iloc[:,-1].value_counts().max()
    is_pure = ((ocurrence / len(data)) * 100) >= 90
    return [is_pure, most_frequent_value]


# function to get extract best feature and its split value
def get_best_feature_split(node_data):
    # Code to find best feature + split
    splits = []

    # iterate all columns. except last one (target column)
    for feature in node_data.columns[:-1]:

        # find unique values and sort them to calculate split values
        sorted_feature = np.sort(node_data[feature].unique())

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
            left_node_values = node_data.loc[node_data.iloc[:,i] <= val , node_data.columns[-1]].to_numpy()

            # get right node values
            right_node_values = node_data.loc[node_data.iloc[:,i] > val , node_data.columns[-1]].to_numpy()

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
            weighted_gini = ((len(left_node_values) / len(node_data.iloc[:,i])) * left_node_gini) + ((len(right_node_values) / len(node_data.iloc[:,i])) * right_node_gini)

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
    # get the feature 
    best_feature = node_data.columns[final_best_gini_index]

    return [best_feature, final_split_value]


# Actual Implementation
def build_tree(node_data):

    # check if node_data is pure
    is_pure = is_pure_enough(node_data)

    # stoping condition
    if is_pure[0]:
        return is_pure[1]
    
    # get best feature and its best split value
    best_feature_split = get_best_feature_split(node_data)

    # get feature and split
    best_feature = best_feature_split[0]
    best_split_value = best_feature_split[1]

    # find left and right node data
    left_node_data = node_data[node_data[best_feature] <= best_split_value]
    right_node_data = node_data[node_data[best_feature] > best_split_value]

    # stopping condition
    #  stop if data after splitting is too small
    if(len(left_node_data) < 5) or (len(right_node_data) < 5):
        most_frequent_value = node_data.iloc[:,-1].value_counts().idxmax()
        return most_frequent_value

    node = {}
    node["feature"] = best_feature
    node["split_value"] = best_split_value
    node["left_node"] = build_tree(left_node_data)
    node["right_node"] = build_tree(right_node_data)

    # return formed tree
    return node

# get a trained CART decision tree
tree = build_tree(training_data)

# print tree
for i in tree:
    print(i)
    print(tree[i])
    print("\n")


# remove the nodes that have left leaf and right leaf as same value
def prune_tree(node):

    # if that node is integer value, return it
    if isinstance(node, numbers.Number):
        return node
    
    # if not integer, means it is dict. So further operation
    node["left_node"] = prune_tree(node["left_node"])
    node["right_node"] = prune_tree(node["right_node"])

    # check if both are numbers
    if isinstance(node["left_node"] , numbers.Number) and isinstance(node["right_node"] , numbers.Number):

        # check if both are equal
        if node["left_node"] == node["right_node"]:
            return node["left_node"]
        
    # if both or either is dict, then return that node
    return node

# get tree after pruning
pruned_tree = prune_tree(tree)

# print tree after pruning
for i in pruned_tree:
    print(i)
    print(pruned_tree[i])
    print("\n")


# Testing
def prediction(decision_node, test_data):
    # find the root feature value
    feature = decision_node["feature"]

    # use feature value to access the index of that feature value in test data
    features = testing_data.columns.to_list()
    index = features.index(feature)

    # find the value for that feature in test_data
    feature_value = test_data[index]

    # check if value is less than or equals to split value of that feature
    if feature_value <= decision_node["split_value"]:
        # check if left_node is dict
        if isinstance(decision_node["left_node"] , dict):
            return prediction(decision_node["left_node"], test_data)
        # if its not dict, Means decision_node is leaf node. So return left node value
        else:
            return decision_node["left_node"]
        
    # when value is greater than split value
    else:
        # check if right_node is dict
        if isinstance(decision_node["right_node"] , dict):
            return prediction(decision_node["right_node"], test_data)
        # if its not dict, means decision_node is leaf node. so return right node value
        else:
            return decision_node["right_node"]

# prediction array
prediction_arr = []
actual_outcome_arr = []

# prediction process
for i in range(len(testing_data)):
    # get single sample of data & convert it into array
    test_data = testing_data.iloc[i,:].to_numpy()

    # add the outcome value in actual_outcome_array
    actual_outcome_arr.append(test_data[-1])

    # get prediction value for that instance
    predicted_value = prediction(pruned_tree,test_data)

    # add it in prediction array
    prediction_arr.append(predicted_value)


# compare prediction and actual values

# it returns array of values True and false
comparison = np.array(actual_outcome_arr) == np.array(prediction_arr)
# collect total number of test samples
total_test_samples = len(actual_outcome_arr)
# get total number of correct predictions
# sum() assumes True as 1 and False as 0. And then calculates sum of all elements
total_correct_predictions = sum(comparison)
# get accuracy percent
accuracy = (total_correct_predictions / total_test_samples) * 100
print(f"Accuracy : {accuracy} %")
