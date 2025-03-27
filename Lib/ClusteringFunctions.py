import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class DataProcessing():

    def __init__(self):
        # Store standardisation parameters (mean, std) for each column
        self.standardisation_params = {}

    def process_well_data(self, file_paths, selected_columns, train_data=False, show_stats=False, show_rows=False):
        """
        Combines and processes data from multiple CSV files.
        
        Parameters:
            file_paths (list): List of file paths for the CSV files.
            selected_columns (list): List of columns to extract and process.
            show_stats (bool): Whether to display descriptive statistics of the processed data.
            show_rows (bool): Whether to display the first few rows of the processed data.
            train_data (bool): If True, combines all data into a single DataFrame + computes and stores standardisation parameters.
                               If False, keeps the data for each file separate.
    
        Returns:
            pd.DataFrame or list of pd.DataFrame: Processed DataFrame(s), either combined or separate for each file.
        """
        
        combined_data = []
        individual_dataframes = []
    
        for file in file_paths:
            # Load the data
            df = pd.read_csv(file)
            
            # Select relevant columns
            df = df[selected_columns]
            
            # Handle missing data: drop rows with missing values
            df.dropna(inplace=True)
            
            # Remove negative values
            numeric_columns = [col for col in selected_columns if col != 'LITHOLOGY']
            df = df[(df[numeric_columns] >= 0).all(axis=1)]
            
            # Compute logarithms for 'KLOGH' and 'RT' if necessary
            if 'KLOGH' in df.columns:
                df['log_KLOGH'] = np.log1p(df['KLOGH'])  # log(1 + x) for stability
                df.drop(columns=['KLOGH'], inplace=True)  # Remove the original 'KLOGH' column
            
            if 'RT' in df.columns:
                df['log_RT'] = np.log1p(df['RT'])
                df.drop(columns=['RT'], inplace=True)  # Remove the original 'RT' column
            
            # Append the processed DataFrame to the list of individual DataFrames
            individual_dataframes.append(df)
            
            # Add to the combined data list for further merging
            combined_data.append(df)
    
        # Combine all dataframes into one if combine_data is True
        if train_data:
            combined_df = pd.concat(combined_data, ignore_index=True)
            
            # Exclude 'LITHOLOGY' from standardisation 
            columns_to_standardise = [col for col in combined_df.columns if col != 'LITHOLOGY']
            
            # Handle standardisation 
            self.compute_standardisation_params(combined_df[columns_to_standardise])  # compute standardisation on selected columns only
    
            # Show statistics if enabled
            if show_stats:
                print("\nDescriptive Statistics of Data:")
                print(combined_df.describe())  # Shows descriptive statistics for the DataFrame
            
            # Show first few rows if enabled
            if show_rows:
                print("\nFirst Few Rows of Data:")
                print(combined_df.head())  # Shows the first few rows of the DataFrame
            
            return combined_df  # Return the combined DataFrame
        else:
            # If train_data is False, show stats and rows for each individual DataFrame
            for idx, df in enumerate(individual_dataframes):
                if show_stats:
                    print(f"\nDescriptive Statistics of DataFrame {idx + 1}:")
                    print(df.describe())  # Shows descriptive statistics for each individual DataFrame
                
                if show_rows:
                    print(f"\nFirst Few Rows of DataFrame {idx + 1}:")
                    print(df.head())  # Shows the first few rows of each individual DataFrame
            
            return individual_dataframes  # Return the list of individual DataFrames

    def compute_standardisation_params(self, df):
       
        """
        Computes and stores standardisation parameters (min, max) for each column.
        
        Parameters:
            df (pd.DataFrame): DataFrame containing the data for which to compute standardisation parameters.
        """
        
        # Compute and store mean and std values for each column
        self.standardisation_params = {
            column: {
                "mean": df[column].mean(),
                "std": df[column].std()
            }
            for column in df.columns #if column not in ['log_KLOGH', 'log_RT']  # To exclude transformed columns, un-comment this line
        }

    def standardise_dataframe(self, df, show_stats=False):
        """
        standardises the data using the stored standardisation parameters.
    
        Parameters:
            df (pd.DataFrame): DataFrame to standardise.
            show_stats (bool): Whether to display descriptive statistics of the standardised data.
        
        Returns:
            pd.DataFrame: standardised DataFrame.
        """
    
        standardised_df = df.copy()
        
        # Loop through columns to standardise, skipping 'LITHOLOGY'
        for column in standardised_df.columns:
            if column == 'LITHOLOGY':  # Skip the 'LITHOLOGY' column
                continue
            if column == 'DEPTH':  # Skip the 'DEPTH' column
                continue
            # Skip transformed columns like 'log_KLOGH' and 'log_RT' if needed
            #if column in ['log_KLOGH', 'log_RT']:
                #continue
    
            # Apply standardisation using stored parameters
            mean_value = self.standardisation_params[column]["mean"]
            std_value = self.standardisation_params[column]["std"]
            standardised_df[column] = (standardised_df[column] - mean_value) / (std_value)
    
        # Show statistics if enabled
        if show_stats:
            print("Standardisation Parameters:")
            for column, params in self.standardisation_params.items():
                print(f"{column}: mean = {params['mean']}, std = {params['std']}")
            print("\nDescriptive Statistics of Standardised Data:")
            print(standardised_df.describe())  # Shows descriptive statistics for the DataFrame
        
        return standardised_df

    def compare_distributions(self, pre_standardised_data, standardised_data, column, title="Feature Distribution Comparison"):
        
        """
        Plots side-by-side distributions for a specific column in pre-standardised and standardised data.
        
        Parameters:
            pre_standardised_data (pd.DataFrame): DataFrame containing pre-standardised data.
            standardised_data (pd.DataFrame): DataFrame containing standardised data.
            column (str): The column to plot.
            title (str): Title for the entire plot.
        """
        
        # Check if the specified column exists in both DataFrames
        if column not in pre_standardised_data.columns or column not in standardised_data.columns:
            raise ValueError(f"Column '{column}' not found in one or both dataframes.")
        
        # Create a figure with two subplots
        fig, axes = plt.subplots(ncols=2, figsize=(10, 4))
        
        # Plot pre-standardised data
        sns.histplot(pre_standardised_data[column], ax=axes[0], kde=True, bins=30, color="darkslategray")
        axes[0].set_title(f'Pre-standardised {column}', fontsize=10)
        axes[0].set_xlabel(column, fontsize=10)
        axes[0].set_ylabel('Frequency', fontsize=10)
        axes[0].grid(True)
        
        # Plot standardised data
        sns.histplot(standardised_data[column], ax=axes[1], kde=True, bins=30, color="firebrick")
        axes[1].set_title(f'Standardised {column}', fontsize=10)
        axes[1].set_xlabel(column, fontsize=10)
        axes[1].set_ylabel('Frequency', fontsize=10)
        axes[1].grid(True)
        
        # Adjust layout and add a title
        plt.suptitle(title, fontsize=12)
        plt.tight_layout(rect=[0, 0, 0.95, 0.95])  # Adjust for the suptitle
        plt.show()

class KMeansClustering():
    
    def __init__(self):
        pass
        
    def elbow_method(self, data, max_k, random_state=42):
    
        """
        Applies the elbow method to find the optimal number of clusters for k-means.
        
        Parameters:
            data (pd.DataFrame): standardised data to cluster (with 'LITHOLOGY' column included).
            max_k (int): Maximum number of clusters to consider.
            random_state (int): Random state for reproducibility.
        
        Returns:
            None: Displays the elbow plot.
        """
        
        # Make a copy of the data to avoid modifying the original
        data_copy = data.copy()
    
        # Exclude 'LITHOLOGY' column for clustering
        if 'LITHOLOGY' in data_copy.columns:
            data_copy = data_copy.drop(columns=['LITHOLOGY'])
        
        # Initialise lists to store results
        means = []
        inertias = []  # List to store within-cluster sum of squares (WCSS)
        
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            kmeans.fit(data_copy)  # Fit the model to the data without 'LITHOLOGY'
            means.append(k)
            inertias.append(kmeans.inertia_)
            print('k:', k)
            print('inertia: ', kmeans.inertia_)  # i.e., Within-Cluster Sum of Squares (WCSS)
        
        # Plot the inertia for each k
        plt.figure(figsize=(8, 6))
        plt.plot(means, inertias, marker='o', linestyle='--')
        plt.title('Elbow Method for Optimal k', fontsize=12)
        plt.tight_layout(rect=[0, 0, 0.7, 0.7])
        plt.xlabel('Number of Clusters, k', fontsize=10)
        plt.ylabel('Inertia', fontsize=10)
        plt.grid(True)
        plt.show()

    def run_kmeans_train(self, train_data, n_clusters=10, show_stats=False):
        
        """
        Runs K-Means clustering on training data and returns the trained model.
    
        Parameters:
            train_data (pd.DataFrame): Training data for clustering.
            n_clusters (int): Number of clusters for K-Means.
            show_stats (bool): Whether to display clustering statistics for training data.
    
        Returns:
            pd.DataFrame: Training data with 'Cluster' column.
            KMeans: Trained K-Means model.
        """
    
        train_data_copy = train_data.copy()
    
        # Exclude 'DEPTH' column for clustering
        if 'DEPTH' in train_data_copy.columns:
            train_data_copy = train_data_copy.drop(columns=['DEPTH'])
        
        # Exclude 'LITHOLOGY' column for clustering
        if 'LITHOLOGY' in train_data_copy.columns:
            train_data_copy = train_data_copy.drop(columns=['LITHOLOGY'])
            
        # Fit K-Means on training data
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        train_clusters = kmeans.fit_predict(train_data_copy)
        
        # Add cluster labels to training data
        train_clustered = train_data_copy.copy()
        train_clustered['Cluster'] = train_clusters
        
        # Show statistics for training data if enabled
        if show_stats:
            print(f"K-Means Clustering Results (Training Data) with {n_clusters} Clusters")
            print(f"Inertia (Sum of Squared Distances to Centroids): {kmeans.inertia_}")
            silhouette_avg = silhouette_score(train_data_copy, train_clusters)
            print(f"Silhouette Score (Training Data): {silhouette_avg:.4f}")
        
        return train_clustered, kmeans

    def run_kmeans_test(self, test_data, kmeans_model):
        
        """
        Applies a trained K-Means model to testing data.
    
        Parameters:
            test_data (pd.DataFrame): Testing data for clustering.
            kmeans_model (KMeans): Pre-trained K-Means model.
    
        Returns:
            pd.DataFrame: Testing data with 'Cluster' column.
        """
    
        test_data_copy = test_data.copy()
    
        # Exclude 'LITHOLOGY' column if it exists
        if 'LITHOLOGY' in test_data_copy.columns:
            test_data_copy = test_data_copy.drop(columns=['LITHOLOGY'])
        # Exclude 'DEPTH' column if it exists
        if 'DEPTH' in test_data_copy.columns:
            test_data_copy = test_data_copy.drop(columns=['DEPTH'])
        
        # Predict clusters using the trained K-Means model
        test_clusters = kmeans_model.predict(test_data_copy)
        
        # Add cluster labels to testing data
        test_clustered = test_data.copy() # copying the original dataframe, so we can use it for plotting
        test_clustered['Cluster'] = test_clusters
    
        return test_clustered
    
    def visualise_lithology_clusters(self, clustered_data):
        
        """
        Visualises lithology and cluster predictions separately on depth profiles.
        
        Parameters:
            clustered_data (pd.DataFrame): DataFrame with 'DEPTH', 'LITHOLOGY', and 'Cluster' columns.
        """
    
        # Sort data by depth
        clustered_data = clustered_data.sort_values(by='DEPTH')
    
        # Extract depth and lithology
        depth_for_lithology = clustered_data['DEPTH'].values
        lithology = clustered_data['LITHOLOGY'].values
        cluster_column = clustered_data['Cluster'].values
    
        # Define lithology labels with colors and hatch patterns
        lithology_labels = {
            'sandstone': {'color': '#ffff00', 'hatch': '..'},
            'marl': {'color': '#80ffff', 'hatch': ''}, 
            'limestone': {'color': '#4682B4', 'hatch': '++'},
            'coal': {'color': 'black', 'hatch': ''},
            'silt': {'color': '#7cfc00', 'hatch': '||'},
            'claystone': {'color': '#228B22', 'hatch': '--'}  
        }

        cluster_labels = {
            0: {'color': '#FF6347', 'label': 'Cluster 0'},  # Tomato red
            1: {'color': '#32CD32', 'label': 'Cluster 1'},  # Lime green
            2: {'color': '#1E90FF', 'label': 'Cluster 2'},  # Dodger blue
            3: {'color': '#FFC0CB', 'label': 'Cluster 3'},  # Pink
            4: {'color': '#FFD700', 'label': 'Cluster 4'},  # Gold
            5: {'color': '#8A2BE2', 'label': 'Cluster 5'},  # Blue violet
            6: {'color': '#00CED1', 'label': 'Cluster 6'},  # Dark turquoise
            7: {'color': '#DC143C', 'label': 'Cluster 7'},  # Crimson
            8: {'color': '#7FFF00', 'label': 'Cluster 8'},  # Chartreuse
            9: {'color': '#FF4500', 'label': 'Cluster 9'}   # Orange red
}
    
        # Create subplots: Lithology (left) and Clusters (right)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 8), sharey=True)
    
        # --- PLOT 1: Lithology (using your fill_betweenx method) ---
        ax1 = axes[0]
        for j in range(len(depth_for_lithology) - 1):
            lith = lithology[j].lower()  # Convert to lowercase for matching
            if lith in lithology_labels:
                hatch = lithology_labels[lith]['hatch']
                color = lithology_labels[lith]['color']
                ax1.fill_betweenx([depth_for_lithology[j], depth_for_lithology[j + 1]], 0, 1, 
                                  facecolor=color, hatch=hatch, alpha=0.6)
    
        ax1.set_xlabel('Lithology', fontsize=12)
        ax1.set_ylabel('Depth (m)', fontsize=12)
        ax1.set_title('Lithology Profile', fontsize=14)
        ax1.invert_yaxis()  # Depth increases downward

        handles = []
        for lith, attrs in lithology_labels.items():
            patch = mpatches.Patch(facecolor=attrs['color'], hatch=attrs['hatch'], edgecolor='k', label=f'{lith}')
            handles.append(patch)
        ax1.legend(handles=handles, loc='best')
    
        # --- PLOT 2: Clusters ---
        ax2 = axes[1]

        for j in range(len(depth_for_lithology) - 1):
            cluster = cluster_column[j]
            if cluster in cluster_labels:
                color = cluster_labels[cluster]['color']
                ax2.fill_betweenx([depth_for_lithology[j], depth_for_lithology[j + 1]], 0, 1, 
                                  facecolor=color, alpha=0.6)
                
        handles = []
        for cluster, attrs in cluster_labels.items():
            patch = mpatches.Patch(facecolor=attrs['color'], edgecolor='k', label=f'cluster {cluster}')
            handles.append(patch)
        ax2.legend(handles=handles, loc='best')

        ax2.set_xlabel('Clusters', fontsize=12)
        ax2.set_title('Cluster Profile', fontsize=14)
        
        plt.tight_layout()
        plt.show()





    
    
            
