---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Imports

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes=True)
%matplotlib inline
```

```python
df = pd.read_csv('./iris.data')
```

```python
df.head(5)
```

```python
col_name = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
```

```python
df.columns = col_name
```

```python
df.head()
```

# Import Iris from SEABORN

```python
iris = sns.load_dataset('iris')
iris.head()
```

```python
df.describe()
```

```python
iris.describe()
```

```python
print(iris.info())
```

```python
print(iris.groupby('species').size())
```

```python
print(iris.all())
```

```python
iris.count()
```

# Visualization

```python
sns.pairplot(iris, hue='species', height=2, aspect=1);
plt.show()
```

```python
iris.hist(edgecolor='white', linewidth=2, figsize=(12, 8))
plt.show()
```

```python
plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
sns.violinplot(x='species', y='sepal_length', data=iris)
plt.subplot(2,2,2)
sns.violinplot(x='species', y='sepal_width', data=iris)
plt.subplot(2,2,3)
sns.violinplot(x='species', y='petal_length', data=iris)
plt.subplot(2,2,4)
sns.violinplot(x='species', y='petal_width', data=iris)
plt.show()
```

```python
iris.boxplot(by='species', figsize=(12,8))
plt.show()
```

```python
pd.plotting.scatter_matrix(iris, figsize=(12,8))
plt.show()
```

```python

```
