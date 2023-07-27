# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# %%
df = pd.read_csv("west_coast_data.csv")

# %%
df = df.dropna()
df['bed'] = df['bed'].astype(int)
df['bath'] =df['bath'].astype(int)

# %%
sns.histplot(df["price"])
plt.title("price histogram")
plt.xlabel("Price (in million dollars)")
plt.show()

# %%
sns.histplot(df["sqft"])
plt.title("Histogram of the area of the properties")
plt.xlabel("Area (in square feet)")
plt.show()

# %%
sns.countplot(data=df, x="bed")
plt.title("Histogram of bedrooms count")
plt.xlabel("No. of bedrooms in the property")
plt.show()

# %%
sns.countplot(data=df, x="bath")
plt.title("Histogram of bathrooms count")
plt.xlabel("No. of bathrooms in the property")
plt.show()

# %%
bed_bath_groups = df[["bed","bath","price"]].groupby(['bed','bath']).mean()

# %%
scatter = plt.scatter(bed_bath_groups.index.get_level_values('bed'),
                      bed_bath_groups.index.get_level_values('bath'),
                      c=bed_bath_groups['price'], cmap='viridis')

# Add colorbar
plt.colorbar(scatter, label='Average Price(in millions of dollars)')
_ = plt.xticks(range(int(df['bed'].min()), int(df['bed'].max()) + 1))
_ = plt.yticks(range(int(df['bath'].min()), int(df['bath'].max()) + 1))
_ = plt.title("avg price of (bedrooms ,bathrooms) combinations")
plt.xlabel("No. of bedrooms")
plt.ylabel("No. of bathrooms")
plt.show()

# %%
