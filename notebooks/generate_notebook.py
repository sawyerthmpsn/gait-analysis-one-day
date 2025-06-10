import nbformat as nbf
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway

nb = nbf.v4.new_notebook()
cells = []

# Markdown and code cells based on EDA outline
cells.append(nbf.v4.new_markdown_cell("# 🧬 Exploratory Data Analysis of Human Gait Data"))

# Add all other cells (as shown in my previous response)
# [For brevity, reuse the detailed notebook content I shared above]

# For example, add this cell for loading data:
cells.append(nbf.v4.new_code_cell("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway

sns.set(style="whitegrid", palette="pastel")

file_path = r'C:\\Users\\M338548\\Documents\\git\\gait-analysis-one-day\\data\\synthetic_gait_analysis_dataset.csv'
df = pd.read_csv(file_path)

display(df.head())
print("Shape:", df.shape)
print("Data Types:\\n", df.dtypes)
print("Missing Values:\\n", df.isnull().sum())
display(df.describe())"""))

# Add additional markdown/code cells here as in previous message
cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(8, 5))
sns.histplot(df['Age'], kde=True, bins=20)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

print("Age Summary:\n", df['Age'].describe())"""))

cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(6, 4))
sns.countplot(x='Sex', data=df)
plt.title('Sex Distribution')
plt.ylabel('Count')
plt.show()"""))

cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(10, 5))
sns.countplot(y='Pathology', data=df, order=df['Pathology'].value_counts().index)
plt.title('Pathology Group Distribution')
plt.xlabel('Count')
plt.ylabel('Pathology')
plt.show()"""))

cells.append(nbf.v4.new_code_cell("""gait_features = ['StrideLength', 'StepTime', 'GaitSpeed', 'Cadence', 'SwingPhase', 'StancePhase']

for feature in gait_features:
    plt.figure(figsize=(7, 4))
    sns.histplot(df[feature], kde=True)
    plt.title(f'{feature} Distribution')
    plt.xlabel(feature)
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[feature])
    plt.title(f'{feature} Boxplot')
    plt.show()"""))

cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(10, 8))
corr_matrix = df[gait_features].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Gait Parameters')
plt.show()"""))

cells.append(nbf.v4.new_code_cell("""# Summary statistics by pathology
group_stats = df.groupby('Pathology')[gait_features].agg(['mean', 'std'])
display(group_stats)

# Visual comparison using boxplots
for feature in gait_features:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='Pathology', y=feature, data=df)
    plt.xticks(rotation=45)
    plt.title(f'{feature} by Pathology')
    plt.tight_layout()
    plt.show()"""))

cells.append(nbf.v4.new_code_cell("""for feature in gait_features:
    groups = [group[feature].dropna().values for name, group in df.groupby('Pathology')]
    stat, p = f_oneway(*groups)
    print(f"{feature}: ANOVA F={stat:.2f}, p={p:.4f}")"""))

nb['cells'] = cells

# Save the notebook
output_path = Path("gait_eda_notebook.ipynb")
with open(output_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
