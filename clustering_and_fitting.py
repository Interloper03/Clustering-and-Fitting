"""
This is the template file for the clustering and fitting assignment.
You will be expected to complete all the sections and
make this a fully working, documented file.
You should NOT change any function, file or variable names,
 if they are given to you here.
Make use of the functions presented in the lectures
and ensure your code is PEP-8 compliant, including docstrings.
Fitting should be done with only 1 target variable and 1 feature variable,
likewise, clustering should be done with only 2 variables.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.optimize import curve_fit


def plot_relational_plot(df):
    """
    Creates a relational plot (scatter plot) comparing Energy and Loudness.
    Saves the plot as 'relational_plot.png'.
    """
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='energy', y='loudness', alpha=0.5, ax=ax)
    ax.set_title("Relational Plot: Energy vs Loudness")
    ax.set_xlabel("Energy")
    ax.set_ylabel("Loudness (dB)")
    plt.savefig('relational_plot.png')
    return


def plot_categorical_plot(df):
    """
    Creates a categorical plot (histogram) of the Danceability score.
    Saves the plot as 'categorical_plot.png'.
    """
    fig, ax = plt.subplots()
    sns.histplot(data=df, x='danceability', bins=20, kde=True, ax=ax)
    ax.set_title("Categorical Plot: Distribution of Danceability")
    ax.set_xlabel("Danceability Score")
    plt.savefig('categorical_plot.png')
    return


def plot_statistical_plot(df):
    """
    Creates a statistical plot (correlation heatmap) for numerical features.
    Saves the plot as 'statistical_plot.png'.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax)
    ax.set_title("Statistical Plot: Correlation Heatmap")
    plt.savefig('statistical_plot.png')
    return


def statistical_analysis(df, col: str):
    """
    Calculates the four main statistical moments for a given column.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        col (str): The name of the column to analyze.

    Returns:
        tuple: (mean, stddev, skew, excess_kurtosis)
    """
    data_series = df[col]
    mean = data_series.mean()
    stddev = data_series.std()
    skew = ss.skew(data_series, bias=False)  # bias=False for sample skewness
    # Fisher's definition (normal = 0.0) used for excess kurtosis
    excess_kurtosis = ss.kurtosis(data_series, fisher=True, bias=False)
    
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """
    Preprocesses the dataframe by dropping nulls and inspecting data.
    
    Args:
        df (pd.DataFrame): Raw dataframe.
        
    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    # Inspection (prints to console for user verification)
    print("--- Data Head ---")
    print(df.head())
    print("\n--- Data Description ---")
    print(df.describe())
    
    # Drop rows with missing values and explicitly COPY to avoid SettingWithCopyWarning
    df = df.dropna().copy()
    
    # Ensure numerical columns are floats (if read incorrectly)
    numeric_cols = ['danceability', 'energy', 'loudness', 'valence']
    for col in numeric_cols:
        if col in df.columns:
            # Use .loc to ensure we modify the dataframe safely
            df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')
            
    # Final cleanup (with another copy to be safe)
    df = df.dropna().copy()
    return df

def writing(moments, col):
    """
    Prints a textual description of the statistical moments.
    """
    print(f'For the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')
    
    # Skewness interpretation
    if moments[2] > 0.5:
        skew_str = "right-skewed"
    elif moments[2] < -0.5:
        skew_str = "left-skewed"
    else:
        skew_str = "not skewed"

    # Kurtosis interpretation (Fisher's definition, 0 is normal)
    if moments[3] > 0.5:
        kurt_str = "leptokurtic"
    elif moments[3] < -0.5:
        kurt_str = "platykurtic"
    else:
        kurt_str = "mesokurtic"

    print(f'The data was {skew_str} and {kurt_str}.')
    return


def perform_clustering(df, col1, col2):
    """
    Performs K-Means clustering on two columns.
    Generates an elbow plot and calculates silhouette score.
    """
    # Gather data and scale
    X = df[[col1, col2]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    def plot_elbow_method():
        """
        Plots the Elbow Method graph to determine optimal k.
        Saves as 'elbow_plot.png'.
        """
        inertias = []
        K_range = range(1, 11)
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_scaled)
            inertias.append(km.inertia_)
            
        fig, ax = plt.subplots()
        ax.plot(K_range, inertias, 'bx-')
        ax.set_xlabel('Values of K')
        ax.set_ylabel('Inertia')
        ax.set_title('The Elbow Method using Inertia')
        plt.savefig('elbow_plot.png')
        return

    def one_silhouette_inertia():
        """
        Calculates silhouette score and inertia for a chosen k (k=3).
        """
        # Using k=3 based on typical Spotify genre clusters (Low/Mid/High energy)
        k_chosen = 3
        km = KMeans(n_clusters=k_chosen, random_state=42, n_init=10)
        km.fit(X_scaled)
        _score = silhouette_score(X_scaled, km.labels_)
        _inertia = km.inertia_
        return _score, _inertia

    # Find best number of clusters
    # We run the plots and calculations as requested
    plot_elbow_method()
    score, inertia = one_silhouette_inertia()
    print(f"Silhouette Score for k=3: {score:.2f}")
    print(f"Inertia for k=3: {inertia:.2f}")

    # Final Clustering with k=3
    k_final = 3
    kmeans = KMeans(n_clusters=k_final, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    # Get cluster centers
    labels = kmeans.labels_
    
    # Inverse transform centers to original scale for plotting context
    centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
    xkmeans = centers_original[:, 0]
    ykmeans = centers_original[:, 1]
    
    cenlabels = [f'Cluster {i+1}' for i in range(k_final)]
    
    # Return the original data (not scaled) for plotting, along with labels
    return labels, df[[col1, col2]].values, xkmeans, ykmeans, cenlabels


def plot_clustered_data(labels, data, xkmeans, ykmeans, centre_labels):
    """
    Plots the results of K-Means clustering.
    Saves as 'clustering.png'.
    """
    fig, ax = plt.subplots()
    # Scatter plot of the data points colored by cluster label
    scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.6)
    
    # Plot the centers
    ax.scatter(xkmeans, ykmeans, c='red', s=200, marker='X', label='Centroids')
    
    ax.set_title("K-Means Clustering (Energy vs Valence)")
    ax.set_xlabel("Energy")
    ax.set_ylabel("Valence")
    plt.legend()
    plt.savefig('clustering.png')
    return


def perform_fitting(df, col1, col2):
    """
    Performs curve fitting (Linear Regression in this case) on two columns.
    """
    # Gather data and prepare for fitting
    x_data = df[col1].values
    y_data = df[col2].values
    
    # Fit model: Linear relationship (y = mx + c)
    def linear_model(x, m, c):
        return m * x + c
        
    popt, pcov = curve_fit(linear_model, x_data, y_data)
    
    # Predict across x (generate a line for plotting)
    x_range = np.linspace(min(x_data), max(x_data), 100)
    y_predicted = linear_model(x_range, *popt)
    
    # Print equation for user reference
    print(f"Fitting Equation: y = {popt[0]:.2f}x + {popt[1]:.2f}")

    return df[[col1, col2]].values, x_range, y_predicted


def plot_fitted_data(data, x, y):
    """
    Plots the original data and the fitted curve.
    Saves as 'fitting.png'.
    """
    fig, ax = plt.subplots()
    # Plot raw data
    ax.scatter(data[:, 0], data[:, 1], label='Data', color='gray', alpha=0.5)
    
    # Plot fitted line
    ax.plot(x, y, label='Fitted Linear Model', color='red', linewidth=2)
    
    ax.set_title("Curve Fitting: Loudness vs Energy")
    ax.set_xlabel("Loudness (dB)")
    ax.set_ylabel("Energy")
    plt.legend()
    plt.savefig('fitting.png')
    return


def main():
    # Ensure your Spotify csv is named 'data.csv'
    try:
        df = pd.read_csv('data.csv')
    except FileNotFoundError:
        print("Error: 'data.csv' not found. Please rename your dataset.")
        return

    df = preprocessing(df)
    
    # Column for statistical moments (Danceability is a good distribution)
    col = 'danceability'
    
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    
    moments = statistical_analysis(df, col)
    writing(moments, col)
    
    # Clustering: Energy (physical intensity) vs Valence (musical positiveness)
    # This groups songs into moods (e.g., Sad/Calm, Happy/Energetic, Aggressive)
    print("\n--- Performing Clustering ---")
    clustering_results = perform_clustering(df, 'energy', 'valence')
    plot_clustered_data(*clustering_results)
    
    # Fitting: Loudness vs Energy
    # These two are physically correlated (louder tracks usually have higher energy)
    print("\n--- Performing Fitting ---")
    fitting_results = perform_fitting(df, 'loudness', 'energy')
    plot_fitted_data(*fitting_results)
    return


if __name__ == '__main__':
    main()