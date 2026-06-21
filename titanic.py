import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (6, 4)

df = sns.load_dataset("titanic")

print("=== HEAD ===")
print(df.head(), "\n")

print("=== INFO ===")
print(df.info(), "\n")

print("=== DESCRIBE (numeric) ===")
print(df.describe(), "\n")

print("=== DESCRIBE (categorical) ===")
print(df.describe(include=["object", "str"]), "\n")   # ← fixed line

print("=== MISSING VALUES (before cleaning) ===")
print(df.isnull().sum(), "\n")

# 2. Data cleaning
df = df.dropna(how="all")
df["age"] = df["age"].fillna(df["age"].median())

for col in ["embarked", "embark_town"]:
    mode_val = df[col].mode(dropna=True)[0]
    df[col] = df[col].fillna(mode_val)

df["deck"] = df["deck"].cat.add_categories(["Unknown"]).fillna("Unknown")

if "alive" in df.columns:
    df = df.drop(columns=["alive"])

print("=== MISSING VALUES (after cleaning) ===")
print(df.isnull().sum(), "\n")

# Plots
plt.figure()
sns.countplot(data=df, x="survived")
plt.title("Survival count (0 = died, 1 = survived)")
plt.tight_layout()
plt.savefig("survival_count.png")
plt.close()

plt.figure()
sns.countplot(data=df, x="sex")
plt.title("Passenger count by sex")
plt.tight_layout()
plt.savefig("sex_count.png")
plt.close()

plt.figure()
sns.countplot(data=df, x="pclass")
plt.title("Passenger count by class")
plt.tight_layout()
plt.savefig("pclass_count.png")
plt.close()

plt.figure()
sns.histplot(df["age"], bins=30, kde=True)
plt.title("Age distribution")
plt.tight_layout()
plt.savefig("age_distribution.png")
plt.close()

plt.figure()
sns.histplot(df["fare"], bins=40, kde=True)
plt.title("Fare distribution")
plt.tight_layout()
plt.savefig("fare_distribution.png")
plt.close()

plt.figure()
sns.barplot(data=df, x="sex", y="survived")
plt.title("Survival rate by sex")
plt.tight_layout()
plt.savefig("survival_by_sex.png")
plt.close()

plt.figure()
sns.barplot(data=df, x="pclass", y="survived")
plt.title("Survival rate by passenger class")
plt.tight_layout()
plt.savefig("survival_by_pclass.png")
plt.close()

plt.figure()
sns.kdeplot(data=df, x="age", hue="survived", fill=True, common_norm=False, alpha=0.4)
plt.title("Age distribution by survival")
plt.tight_layout()
plt.savefig("age_by_survival.png")
plt.close()

numeric_df = df.select_dtypes(include=["float64", "int64", "bool"])
corr = numeric_df.corr(numeric_only=True)

print("=== CORRELATION WITH SURVIVED ===")
print(corr["survived"].sort_values(ascending=False), "\n")

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation heatmap (numeric features)")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

print("Script finished. Plots saved as PNG files in the current directory.")
