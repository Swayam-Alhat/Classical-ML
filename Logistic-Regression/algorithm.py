import matplotlib.pyplot as plt

x = [1,2,3]
y = [2,4,6]

plt.scatter(x,y)
plt.xlabel("House size (100 sqft)")
plt.ylabel("House price (lakh)")
plt.title("House size vs house price")
plt.show()