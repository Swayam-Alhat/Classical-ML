# Decision Trees

## What is a Decision Tree?

A decision tree is an ML algorithm that learns patterns from data and predicts outcomes for new unseen data. It forms a tree-like structure made of nodes and branches — essentially a flowchart of questions that leads to a prediction.

Each node in the tree represents a feature (like Glucose, Age, BMI). Each branch represents a split based on that feature's value. At the bottom are leaf nodes — these give the final prediction.

---

## How It Works

### 1. The core idea

The algorithm wants to ask the most important question first. "Most important" means — which feature, when split at the right value, best separates the target classes?

For example, if we want to predict whether a patient is diabetic, we want a feature that clearly divides diabetic vs non-diabetic patients into two groups. That feature becomes the root node (the first question).

### 2. How it finds the best feature

For every feature in the dataset, the algorithm:

- Tries every possible split threshold (e.g. Glucose <= 85, Glucose <= 105, Glucose <= 135...)
- Each threshold splits the data into two groups (left and right)
- It checks how **pure** each group is — meaning, does each group contain mostly one class?
- Picks the threshold that creates the purest groups for that feature

Then it compares the best split from every feature and picks the overall winner. That winner becomes the node.

Purity is measured using **Gini Impurity** — a number from 0 to 0.5. A value of 0 means the group is perfectly pure (all same class). A value of 0.5 means completely mixed.

### 3. It repeats recursively

After the first split creates two groups, the algorithm goes into each group separately and runs the exact same process again — find the best feature and best split for that subset of data. It keeps doing this until the groups are pure enough, or until a stopping condition is met (like max depth, or minimum samples in a node).

### 4. Prediction on new data

To predict, simply walk the new sample down the tree — at each node, check the condition, go left or right, and keep going until you hit a leaf. The leaf gives the prediction (majority class of training samples that ended up there).

---

## Full Example — Feature Selection Step by Step

### Dataset

5 patients, 2 features, target = Diabetic (1) or Not Diabetic (0).

| Patient | Glucose | Age | Diabetic? |
| ------- | ------- | --- | --------- |
| P1      | 80      | 25  | 0         |
| P2      | 90      | 35  | 0         |
| P3      | 120     | 45  | 1         |
| P4      | 150     | 55  | 1         |
| P5      | 160     | 60  | 1         |

---

### Step 1 — Try all splits for Glucose

Thresholds tried: midpoints between sorted unique values → 85, 105, 135, 155

**Glucose <= 85**

- Left: P1 → targets: [0] → pure ✅
- Right: P2, P3, P4, P5 → targets: [0, 1, 1, 1] → mixed ❌

**Glucose <= 105**

- Left: P1, P2 → targets: [0, 0] → pure ✅
- Right: P3, P4, P5 → targets: [1, 1, 1] → pure ✅
- **Perfect split! 🎯**

**Glucose <= 135**

- Left: P1, P2, P3 → targets: [0, 0, 1] → mixed ❌
- Right: P4, P5 → targets: [1, 1] → pure ✅

**Glucose <= 155**

- Left: P1, P2, P3, P4 → targets: [0, 0, 1, 1] → mixed ❌
- Right: P5 → targets: [1] → pure ✅

**Best split for Glucose → Glucose <= 105** (both sides pure)

> [!NOTE]
> **_Gini impurity_** of each feature split represent whether the current split best seprates the target values. That is, if we have **_Glucose <= 105_** as best split. Means its gini impurity is lower than other feature splits

---

### Step 2 — Try all splits for Age

Thresholds tried: 30, 40, 50, 57.5

**Age <= 30**

- Left: P1 → targets: [0] → pure ✅
- Right: P2, P3, P4, P5 → targets: [0, 1, 1, 1] → mixed ❌

**Age <= 40**

- Left: P1, P2 → targets: [0, 0] → pure ✅
- Right: P3, P4, P5 → targets: [1, 1, 1] → pure ✅
- **Perfect split! 🎯**

**Age <= 50**

- Left: P1, P2, P3 → targets: [0, 0, 1] → mixed ❌
- Right: P4, P5 → targets: [1, 1] → pure ✅

**Age <= 57.5**

- Left: P1, P2, P3, P4 → targets: [0, 0, 1, 1] → mixed ❌
- Right: P5 → targets: [1] → pure ✅

**Best split for Age → Age <= 40** (both sides pure)

---

### Step 3 — Compare winners across all features

| Feature | Best Split | Left targets | Right targets | Both pure? |
| ------- | ---------- | ------------ | ------------- | ---------- |
| Glucose | <= 105     | [0, 0]       | [1, 1, 1]     | ✅ Yes     |
| Age     | <= 40      | [0, 0]       | [1, 1, 1]     | ✅ Yes     |

Both splits are equally good here. In a tie, CART picks whichever comes first — so **Glucose <= 105** becomes the root node.

---

### Step 4 — The resulting tree

```
            [Glucose <= 105?]
               /            \
             YES              NO
          [0, 0]           [1, 1, 1]
        Predict: 0        Predict: 1
      (Not Diabetic)       (Diabetic)
```

Both leaves are pure → no further splitting needed. Tree is complete.

---

### Step 5 — Predict a new patient

New patient: Glucose = 130, Age = 48

- Start at root: **Glucose <= 105?** → 130 <= 105? → **NO**
- Go right → leaf says **Predict: 1 (Diabetic)**

---

## Key Terms

| Term          | Meaning                                                            |
| ------------- | ------------------------------------------------------------------ |
| Root node     | The first question — the most important feature                    |
| Internal node | Any node that still splits further                                 |
| Leaf node     | End of the tree — gives the final prediction                       |
| Threshold     | The value used to split a continuous feature (e.g. Glucose <= 105) |
| Gini Impurity | Measures how mixed a group is. 0 = pure, 0.5 = fully mixed         |
| Pure node     | A node where all samples belong to the same class                  |
| Overfitting   | When the tree memorizes training data by growing too deep          |
| Max depth     | Hyperparameter that limits how deep the tree can grow              |

> [!NOTE]
> Read above explaination **twice** so you will get it
