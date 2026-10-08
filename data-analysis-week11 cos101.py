import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Load dataset
df = pd.read_csv("iris.csv")

# 1. Summary Statistics
print("--- Dataset Overview ---")
print(df.info())

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Descriptive Statistics ---")
print(df.describe())

print("\n--- Class Distribution ---")
print(df["species"].value_counts())

# Set plot style
sns.set_theme(style="whitegrid")

# 2. Exploratory Visualizations

# Histograms of features
df.hist(figsize=(10, 8), bins=15, edgecolor="black")
plt.suptitle("Feature Distributions", fontsize=14)
plt.tight_layout()
plt.show()

# Pairplot to visualize relationships between features colored by species
sns.pairplot(df, hue="species", markers=["o", "s", "D"])
plt.suptitle("Pairplot of Iris Features", y=1.02)
plt.show()

# Boxplots for each numerical feature across species
plt.figure(figsize=(12, 8))
features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

for i, feature in enumerate(features, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x="species", y=feature, data=df)
    plt.title(f"{feature.replace('_', ' ').title()} by Species")

plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8, 6))
numeric_df = df.drop(columns=["species"])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()
