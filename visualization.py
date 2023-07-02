# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import seaborn as sns
import pandas as pd
import numpy as np

# %%
df = pd.read_csv("california_data.csv")

# %%
df.dtypes

# %%
sns.histplot(df["price"])

# %%
sns.countplot(data=df,x="bed")

# %%
df.loc[df["bed"] == 12]

# %%
df

# %%
