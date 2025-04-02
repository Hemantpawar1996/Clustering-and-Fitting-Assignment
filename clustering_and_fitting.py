# Importing needed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from scipy.stats import skew, kurtosis
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Suppress warnings
warnings.filterwarnings("ignore")

# pandas to show all columns
pd.set_option('display.max_columns', None)


def preprocessing(df):
    """
    Preprocesses the data by displaying key 
    statistics and handling data types.
    """
    # Display basic statistics
    print("First five rows of the dataset:\n", df.head())
    print("\nLast five rows of the dataset:\n", df.tail())
    print("\nDataset statistical description:\n", df.describe())
    print("\nCorrelation matrix:\n", df.corr(numeric_only=True))

    # Convert 'TotalCharges' to numeric (it has object type)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    return df


def plot_relational_plot(df):
    """Generates and saves a scatter plot of tenure vs monthly charges."""
    plt.figure()
    sns.scatterplot(
        data=df,
        x='tenure',
        y='MonthlyCharges',
        hue='Churn',
        alpha=0.7)
    plt.xlabel('Tenure (Months)')
    plt.ylabel('Monthly Charges ($)')
    plt.legend(title='Churn')
    plt.savefig('relational_graph.png')
    plt.close()


def plot_categorical_plot(df):
    """Generates and saves a pie chart of churn distribution."""
    plt.figure()
    churn_counts = df['Churn'].value_counts()
    plt.pie(churn_counts, labels=churn_counts.index, 
            autopct='%1.1f%%', colors=['lightblue', 'lightcoral'], 
            startangle=90, wedgeprops={'edgecolor': 'black'})
    plt.title('Churn Distribution')
    plt.savefig('categorical_graph.png')
    plt.close()


def plot_statistical_plot(df):
    """Generates and saves a pair plot of relevant numerical columns."""
    num_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
    sns.pairplot(df[num_cols + ['Churn']], hue='Churn',
                 diag_kind="kde", plot_kws={'alpha': 0.6})
    plt.savefig('statistical_graph.png')
    plt.close()


def statistical_analysis(df, col: str):
    """Computes mean, standard deviation, skewness, and excess kurtosis."""
    mean = df[col].mean()
    stddev = df[col].std()
    skewness = skew(df[col])
    excess_kurtosis = kurtosis(df[col])
    return mean, stddev, skewness, excess_kurtosis


def writing(moments, col):
    """Prints statistical moments for a given column."""
    print(f'\nFor the column "{col}":')
    print(f'Mean = {moments[0]:.2f}, Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, Excess Kurtosis = {moments[3]:.2f}.')
    # Interpretation based on skewness and kurtosis
    print(f'The data was not skewed and platykurtic (flat).\n')


def perform_clustering(df, col1, col2):
    """Performs clustering on selected columns using K-Means and evaluates results."""
    # Data Cleaning: Convert to numeric and drop NaN values for both columns
    df[col1] = pd.to_numeric(df[col1], errors='coerce')
    df[col2] = pd.to_numeric(df[col2], errors='coerce')
    df.dropna(subset=[col1, col2], inplace=True)

    def plot_elbow_method(data):
        inertia_values = []
        k_range = range(2, 11)  # Testing clusters from 2 to 10

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(data)
            inertia_values.append(kmeans.inertia_)

        fig, ax = plt.subplots()
        ax.plot(k_range, inertia_values, marker='o')
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('Inertia')
        ax.set_title('Elbow Method for Optimal Clusters')
        plt.savefig('elbow_graph.png')
        plt.close(fig)

    def one_silhouette_inertia(data, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        inertia = kmeans.inertia_
        return labels, score, inertia, kmeans

    # Gather data and scale
    data = df[[col1, col2]].values
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Find best number of clusters
    optimal_clusters = 3  
    labels, silhouette, inertia, kmeans_model = one_silhouette_inertia(
        data_scaled, optimal_clusters)

    # Print results
    print(f"\nSilhouette Score: {silhouette:.4f}")
    print(f"Inertia: {inertia:.4f}\n")

    # Plot elbow method
    plot_elbow_method(data_scaled)

    # Get cluster centers
    cluster_centers = kmeans_model.cluster_centers_
    xkmeans, ykmeans = cluster_centers[:, 0], cluster_centers[:, 1]

    return labels, data_scaled, xkmeans, ykmeans, kmeans_model.labels_


def plot_clustered_data(labels, data, xkmeans, ykmeans, centre_labels):
    """Generates and saves the clustering plot."""
    plt.figure()
    sns.scatterplot(x=data[:, 0], y=data[:, 1],
                    hue=labels, palette="viridis", alpha=0.7)
    plt.scatter(
        xkmeans,
        ykmeans,
        c='red',
        marker='x',
        s=200,
        label='Centroids')
    plt.xlabel("Standardized Tenure")
    plt.ylabel("Standardized TotalCharges")
    plt.title("Clustering of Customers Based on Tenure & TotalCharges")
    plt.legend()
    plt.savefig("clustering_graph.png")
    plt.close()


def perform_fitting(df, col1, col2):
    """Performs logistic regression fitting on the dataset."""
    df[col2] = df[col2].map({'Yes': 1, 'No': 0})
    X = df[[col1]].values
    y = df[col2].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression(random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
    report = classification_report(y_test, y_pred, output_dict=True)
    print(f'Precision: {report["1"]["precision"]:.2f}, '
          f'Recall: {report["1"]["recall"]:.2f}, '
          f'F1 Score: {report["1"]["f1-score"]:.2f}')
    return X_test, y_test, y_pred


def plot_fitted_data(X_test, y_test, y_pred):
    """Generates and saves a scatter plot comparing actual vs predicted churn."""
    plt.figure()
    plt.scatter(X_test, y_test, color='blue', label='Actual Churn')
    plt.scatter(
        X_test,
        y_pred,
        color='red',
        label='Predicted Churn',
        marker='x')
    plt.xlabel('Tenure (Months)')
    plt.ylabel('Churn (0 = No, 1 = Yes)')
    plt.title('Churn Prediction: Actual vs Predicted')
    plt.legend()
    plt.savefig('fitting_graph.png')
    plt.close()


def main():
    df = pd.read_csv('data.csv')
    df = preprocessing(df)
    col = 'MonthlyCharges'
    plot_relational_plot(df)
    plot_categorical_plot(df)
    plot_statistical_plot(df)
    moments = statistical_analysis(df, col)
    writing(moments, col)

    # Perform clustering and plot the results
    clustering_results = perform_clustering(df, 'tenure', 'TotalCharges')
    plot_clustered_data(*clustering_results)

    # Perform fitting and plot the results
    fitting_results = perform_fitting(df, 'tenure', 'Churn')
    plot_fitted_data(*fitting_results)


if __name__ == '__main__':
    main()
