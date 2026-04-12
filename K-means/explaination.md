# k-means Clustering

Its a unsupervised ML algorithm. Meaning it uses **unlabeled dataset** to learn patterns.

**Core idea**  
**This algorithm groups similar data points from dataset. This groups are called _clusters_.**

## Working of K-means clustering

**_Steps_**

1. Initialize the k centroid values. This are center points of k cluster.

2. Each data point is assigned to the nearest centroid based on a distance metric, usually **Euclidean distance**.

_This step ensures that each point is grouped with other similar points, forming the initial clusters_

3. Once all points are assigned to clusters, the algorithm recalculates the centroids by taking the mean of the data points in each cluster.

4. _Steps 2 and 3_ are repeated until the centroids stop moving or the change in their positions is negligible. This indicates that the algorithm has converged and the clusters are stable.

**_The final clusters are then outputted as the result of the K-Means algorithm_**

## Implementation

Before actual implementation, we need to -

1. **Preprocess the data by identifying and removing outliers, because k-means is sensitive to outliers**

2. **Choose the optimal value of k (number of clusters) using Elbow method or Silhouette Score**

3. **Select the appropriate initial centroid values by running algorithm with multiple initialization centroid values.**

### Detect Outliers using IQR (Interquartile Range)

IQR is simply, `Q3 - Q1`, where ,

_Q1 is value which represent 25% of data is below that value (Q1)_

_Q3 is value which represent 75% of data is below that value (Q3)_

> Q2 is value which represent 50% of data is below that value (Q2)

Example, data `[21,23,24,26,28,30,31,32,35,78]`,

so, first sort it (alredy sorted)

So, Q1 = 0.25 x no. of data points = 0.25 x 10 = 2.5, means data points at index 2 and 3. So,take average of them,

`24 + 26 / 2` = 25, meaning 25% of data points are below 25. SO **_Q1 = 25_**

Similarly, for Q3, we get `32 + 35 / 2` = 33.5, So, **_Q3 = 33.5_**  
Meaning, 75% of data points are below 33.5

So, we get Q1 and Q3. **_This means, we got to know that 50% of data points in dataset is between 25 (Q1) & 33.5 (Q3) _**

> **_Because 75% - 25% is 50%. And since, 25% of data is below 25 & 75% of data is below 33.5_**

Now, ***the middle 50% of data lives between 25 and 33.5. Everything between these two values is the "normal bulk" of our data***

**But**,

- **_we can't throw everything which are outside of this range (i.e Q3 - Q1), because values slightly outside this range are still perfectly normal — they're just on the fringe_**

- **_Also, this is range is just 50% of whole data_**

Since, `IQR = Q3 - Q1`, and this IQR is range of normal data points (i.e 33.5 - 25 = **8.5**)

So, we extend each side by 1.5 × IQR

```
Lower fence = Q1 - (1.5 × IQR) = 25 - (1.5 × 8.5) = 25 - 12.75 = 12.25
Upper fence = Q3 + (1.5 × IQR) = 33.5 + (1.5 × 8.5) = 33.5 + 12.75 = 46.25
```

**Acceptable zone = [12.25, 46.25]**

Now we compare each data point against this acceptable zone [12.25, 46.25].
Any data point that falls below 12.25 or above 46.25 is an **outlier** and gets removed.

In our example, `78` falls outside the upper fence (46.25), so it gets removed.
Clean dataset: `[21, 23, 24, 26, 28, 30, 31, 32, 35]`

> [!NOTE]  
> For dataset with multiple features, You apply IQR independently to each column, and remove a row if it's an outlier in any column.
> One bad feature poisons the whole row
