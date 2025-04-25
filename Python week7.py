# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Task 1: Load and Explore the Dataset
def load_and_explore_iris():
    # Load the Iris dataset from seaborn
    df = sns.load_dataset('iris')
    print("Iris dataset loaded successfully!")
    print("\nFirst few rows:")
    print(df.head())
    print("\nDataset information:")
    print(df.info())
    print("\nMissing values:")
    print(df.isnull().sum())
    # The Iris dataset has no missing values, so no handling is needed for this example.
    return df

# Task 2: Basic Data Analysis
def basic_analysis_iris(df):
    if df is not None:
        print("\nDescriptive statistics for numerical columns:")
        print(df.describe())

        # Group by the categorical column 'species' and compute the mean of numerical columns
        grouped_mean = df.groupby('species').mean()
        print("\nMean of numerical features by species:")
        print(grouped_mean)

        print("\nInteresting findings:")
        print("- The different species of Iris show distinct average measurements for sepal length, sepal width, petal length, and petal width.")
        print("- For example, Iris-setosa generally has smaller petal length and width compared to the other two species.")
        print("- Sepal width seems to have less variation across the species compared to petal length and width.")

# Task 3: Data Visualization
def data_visualization_iris(df):
    if df is not None:
        # Visualization 1: Bar chart - Average petal length per species
        plt.figure(figsize=(8, 6))
        sns.barplot(x='species', y='petal_length', data=df)
        plt.xlabel('Species')
        plt.ylabel('Average Petal Length (cm)')
        plt.title('Average Petal Length per Iris Species')
        plt.show()

        # Visualization 2: Histogram - Distribution of sepal width
        plt.figure(figsize=(8, 6))
        sns.histplot(df['sepal_width'], bins=10, kde=True)
        plt.xlabel('Sepal Width (cm)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Sepal Width')
        plt.show()

        # Visualization 3: Scatter plot - Sepal length vs. petal length
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='sepal_length', y='petal_length', hue='species', data=df)
        plt.xlabel('Sepal Length (cm)')
        plt.ylabel('Petal Length (cm)')
        plt.title('Sepal Length vs. Petal Length')
        plt.legend(title='Species')
        plt.grid(True)
        plt.show()

        # Visualization 4: Box plot - Petal width distribution per species
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='species', y='petal_width', data=df)
        plt.xlabel('Species')
        plt.ylabel('Petal Width (cm)')
        plt.title('Petal Width Distribution per Iris Species')
        plt.show()

# Main execution
iris_df = load_and_explore_iris()
basic_analysis_iris(iris_df)
data_visualization_iris(iris_df)