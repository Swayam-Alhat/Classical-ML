sizes = [1,2,3] # house size in 100sqft
prices = [2,4,6] # house price in lakh

# initial values
m = 0
b = 0
MSE = 0
learning_rate = 0.05
MSE_list = []

for i in range(2000):
    errors = []
    for index, size in enumerate(sizes):
        # predict y for current house size
        predicted_y = m * size + b

        # calculate difference between actual and predicted value
        error = prices[index] - predicted_y
        # add in errors list
        errors.append(error)
    
    # calculate MSE
    MSE = sum(err ** 2 for err in errors) / len(errors)
    print(f"MSE at {i + 1} iteration is {MSE}")

    # add in MSE_list
    MSE_list.append(MSE)

    # check if Its 1s iteration
    if (i > 0):

        # check if difference between previous MSE and current MSE is < 0.5
        if (((MSE_list[-2] - MSE_list[-1]) / MSE_list[-2]) * 100 < 0.1):
            break

    # calculate gradient for m and b
    grad_m = (-2/len(sizes)) * sum( err * size for err, size in zip(errors,sizes))
    grad_b = (-2/len(sizes)) * sum(errors)

    # update m and b
    m = m - (learning_rate * grad_m)
    b = b - (learning_rate * grad_b)


print(f"{"=" * 45}")
print(f"Optimal value of m : {m}")
print(f"Optimal value of b : {b}")
print(f"Last MSE : {MSE}")

         

