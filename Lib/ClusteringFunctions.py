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

    def visualise_lithology_distribution(self, csv_file_paths, display='count'):
            """
        Visualises the combined lithology distribution for multiple wells across multiple CSV files in a single plot.

        Parameters:
            csv_file_paths (list): List of paths to CSV files containing well data.
            display (str): One of 'count', 'percentage', or 'both'.
        """

        # Create a dictionary to accumulate lithology counts across all wells
        combined_lithology_counts = {}

        # Iterate through each CSV file
        for csv_file_path in csv_file_paths:
            df = pd.read_csv(csv_file_path)

            # Check if 'LITHOLOGY' column exists
            if 'LITHOLOGY' not in df.columns:
                print(f"Skipping {csv_file_path}: Missing 'LITHOLOGY' column.")
                continue

            # Get lithology counts for the current well
            lithology_counts = df['LITHOLOGY'].value_counts()

            # Update the combined lithology counts
            for lithology, count in lithology_counts.items():
                combined_lithology_counts[lithology] = combined_lithology_counts.get(lithology, 0) + count

            # Total count across all lithologies
            total = sum(combined_lithology_counts.values())
    
            # Dictionary of lithology properties (color, hatch symbol)
            lithology_dict = self.lithology_labels
    
            # Create the plot
            fig, ax = plt.subplots(figsize=(10, 8))
    
            for lithology, count in combined_lithology_counts.items():
                color = lithology_dict.get(lithology, {}).get('color', '#D2B48C')  # Default tan color
                hatch = lithology_dict.get(lithology, {}).get('hatch', '')
    
                # Draw bar
                bar = ax.bar(lithology, count, color=color, hatch=hatch)
    
                # Correct percentage for this lithology
                percent = (count / total) * 100
    
                # Label above bar
                if display == 'count':
                    label = str(int(count))
                elif display == 'percentage':
                    label = f"{percent:.1f}%"
                elif display == 'both':
                    label = f"{int(count)}\n({percent:.1f}%)"
                else:
                    label = ''
    
                ax.text(bar[0].get_x() + bar[0].get_width() / 2, bar[0].get_height(), label,
                        ha='center', va='bottom', fontsize=10)
    
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title('Combined Lithology Distribution Across Wells', fontsize=14)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    def process_well_data(self, file_paths, selected_columns, method='standard', train_data=False, val_data=False, show_stats=False, show_rows=False):
        
        """
        Combines and processes data from multiple CSV files.
        
        Parameters:
            file_paths (list): List of file paths for the CSV files.
            selected_columns (list): List of columns to extract and process.
            train_data (bool): If True, combines all data into a single DataFrame + computes and stores scaling parameters.
            val_data (bool): If True, combines all data into a single DataFrame.
                             If neither train_data nor val_data are True >> False >> keeps the data for each file separate.
            show_stats (bool): Whether to display descriptive statistics of the processed data.
            show_rows (bool): Whether to display the first few rows of the processed data.
    
        Returns:
            pd.DataFrame or list of pd.DataFrame: Processed DataFrame(s), either combined or separate for each file.
        """
        
        combined_data = []
        individual_dataframes = []
    
        for file in file_paths:
            df = pd.read_csv(file) # Load the data
            df = df[selected_columns] # Select relevant columns
            df.dropna(inplace=True) # Handle missing data: drop rows with missing values 
            
            numeric_columns = [col for col in selected_columns if col != 'DEPTH'] # exclude 'LITHOLOGY' from selected columns
            numeric_columns = [col for col in numeric_columns if col != 'LITHOLOGY'] # exclude 'LITHOLOGY' from numerical columns
            df = df[(df[numeric_columns] >= 0).all(axis=1)] # Remove negative values
            
            # Compute logarithms for 'KLOGH' and 'RT' if necessary
            if 'KLOGH' in df.columns:
                df['log_KLOGH'] = np.log1p(df['KLOGH'])  # log(1 + x) for stability
                df.drop(columns=['KLOGH'], inplace=True)  # Remove the original 'KLOGH' column
            if 'RT' in df.columns:
                df['log_RT'] = np.log1p(df['RT'])
                df.drop(columns=['RT'], inplace=True)  # Remove the original 'RT' column
            if 'GR' in df.columns:
                df = df[df['GR'] <= 150]
        
            individual_dataframes.append(df) # Append the processed DataFrame to the list of individual DataFrames
            combined_data.append(df)  # Add to the combined data list for further merging
    
        if train_data: # if train_data is True
            combined_df = pd.concat(combined_data, ignore_index=True) 
            columns_to_scale = [col for col in combined_df.columns if col != 'LITHOLOGY'] # Exclude 'LITHOLOGY' from scaling 
            self.compute_scaling_params(combined_df[columns_to_scale], method=method)  # compute scaling parameters on selected columns only
            if show_stats: # Show statistics if enabled
                print("\nDescriptive Statistics of Data:")
                print(combined_df.describe())  # Shows descriptive statistics for the DataFrame
            if show_rows: # Show first few rows if enabled
                print("\nFirst Few Rows of Data:")
                print(combined_df.head())  # Shows the first few rows of the DataFrame
            combined_df = shuffle(combined_df, random_state=42)
            return combined_df  # Return the combined DataFrame
            
        elif val_data:
            combined_df = pd.concat(combined_data, ignore_index=True) 
            if show_stats: # Show statistics if enabled
                print("\nDescriptive Statistics of Data:")
                print(combined_df.describe())  # Shows descriptive statistics for the DataFrame
            if show_rows: # Show first few rows if enabled
                print("\nFirst Few Rows of Data:")
                print(combined_df.head())  # Shows the first few rows of the DataFrame
            combined_df = shuffle(combined_df, random_state=42)
            return combined_df  # Return the combined DataFrame
            
        else:
            for idx, df in enumerate(individual_dataframes):
                if show_stats:
                    print(f"\nDescriptive Statistics of DataFrame {idx + 1}:")
                    print(df.describe())  # Shows descriptive statistics for each individual DataFrame
                
                if show_rows:
                    print(f"\nFirst Few Rows of DataFrame {idx + 1}:")
                    print(df.head())  # Shows the first few rows of each individual DataFrame
            return individual_dataframes  # Return the list of individual DataFrames

    def compute_scaling_params(self, df, method='standard'):
        """
        Computes and stores scaling parameters for each column.
    
        Parameters:
            df (pd.DataFrame): DataFrame containing the data.
            method (str): 'standard', 'minmax_01', or 'minmax_11'.
        """
        self.scaling_method = method
        self.scaling_params = {}
    
        for column in df.columns:
            if column in ['DEPTH', 'LITHOLOGY']:
                continue
    
            if method == 'standard':
                self.scaling_params[column] = {
                    'mean': df[column].mean(),
                    'std': df[column].std()
                }
            elif method in ['minmax_01', 'minmax_11']:
                self.scaling_params[column] = {
                    'min': df[column].min(),
                    'max': df[column].max()
                }
            else:
                raise ValueError("Unsupported method. Choose from 'standard', 'minmax_01', 'minmax_11'.")

    def scale_dataframe(self, df, show_stats=False):
        """
        Scales the data using the stored scaling parameters.
    
        Parameters:
            df (pd.DataFrame): DataFrame to scale.
            show_stats (bool): Whether to display descriptive statistics of the scaled data.
    
        Returns:
            pd.DataFrame: Scaled DataFrame.
        """
        if not hasattr(self, 'scaling_params') or not hasattr(self, 'scaling_method'):
            raise ValueError("Scaling parameters not computed. Run compute_scaling_params first.")
    
        scaled_df = df.copy()
        method = self.scaling_method
    
        for column in scaled_df.columns:
            if column in ['DEPTH', 'LITHOLOGY']:
                continue
    
            if method == 'standard':
                mean = self.scaling_params[column]['mean']
                std = self.scaling_params[column]['std']
                scaled_df[column] = (scaled_df[column] - mean) / std
    
            elif method == 'minmax_01':
                min_val = self.scaling_params[column]['min']
                max_val = self.scaling_params[column]['max']
                scaled_df[column] = (scaled_df[column] - min_val) / (max_val - min_val)
    
            elif method == 'minmax_11':
                min_val = self.scaling_params[column]['min']
                max_val = self.scaling_params[column]['max']
                scaled_df[column] = 2 * ((scaled_df[column] - min_val) / (max_val - min_val)) - 1
    
        if show_stats:
            print(f"\nScaling Method: {method}")
            print("Scaling Parameters:")
            for column, params in self.scaling_params.items():
                # Convert all values to plain float before printing
                clean_params = {k: float(v) for k, v in params.items()}
                print(f"{column}: {clean_params}")
    
        return scaled_df

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
    
    def visualise_lithology_clusters(self, clustered_data, log_columns=None):
        
        """
        Visualises lithology and cluster predictions separately on depth profiles.
        
        Parameters:
            clustered_data (pd.DataFrame): DataFrame with 'DEPTH', 'LITHOLOGY', and 'Cluster' columns.
        """
    
        # Sort data by depth
        clustered_data = clustered_data.sort_values(by='DEPTH')
        depth = clustered_data['DEPTH'].values
        lithology = clustered_data['LITHOLOGY'].values
        cluster_column = clustered_data['Cluster'].values
    
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
        
        # Color cycle for logs
        log_colors = [
            "darkred", "royalblue", "forestgreen", "orange", "mediumpurple",
            "teal", "crimson", "slategray", "black"
        ]
        
        n_logs = len(log_columns) if log_columns else 0
        total_cols = n_logs + 2  # logs + lithology + cluster
    
        fig, axes = plt.subplots(nrows=1, ncols=total_cols, figsize=(3.5 * total_cols, 8), sharey=True)
    
        if total_cols == 1:
            axes = [axes]
    
        # --- Plot logs ---
        if log_columns:
            for i, log_name in enumerate(log_columns):
                ax = axes[i]
                if log_name not in clustered_data.columns:
                    print(f"Warning: Log column '{log_name}' not found in DataFrame.")
                    continue
                color = log_colors[i % len(log_colors)]
                ax.plot(clustered_data[log_name], depth, color=color, lw=2)
                ax.set_xlabel(log_name, fontsize=18)
                #ax.set_title(f'{log_name} Log', fontsize=12)
                ax.invert_yaxis()
                ax.xaxis.set_ticks_position('bottom')
                ax.xaxis.set_label_position('bottom')
                if i != 0:
                    ax.tick_params(labelleft=False)
                ax.grid(True)
        
        # --- Plot lithology ---
        ax_lith = axes[n_logs]
        for j in range(len(depth) - 1):
            lith = lithology[j].lower()
            if lith in lithology_labels:
                hatch = lithology_labels[lith]['hatch']
                color = lithology_labels[lith]['color']
                ax_lith.fill_betweenx([depth[j], depth[j + 1]], 0, 1, facecolor=color, hatch=hatch, alpha=0.6, edgecolor='k')
        
        ax_lith.set_xlabel('Lithology', fontsize=15)
        ax_lith.set_title('Lithology Profile', fontsize=18)
        ax_lith.invert_yaxis()
        #ax_lith.xaxis.set_ticks_position('bottom')
        #ax_lith.xaxis.set_label_position('bottom')
        #ax_lith.tick_params(labelleft=False, labelbottom=False)
    
        lith_handles = [
            mpatches.Patch(facecolor=attrs['color'], hatch=attrs['hatch'], edgecolor='k', label=lith)
            for lith, attrs in lithology_labels.items()
        ]
        ax_lith.legend(handles=lith_handles, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(lith_handles), fontsize=10, fancybox=True)
    
        # --- Plot clusters ---
        ax_cluster = axes[n_logs + 1]
        for j in range(len(depth) - 1):
            cluster = cluster_column[j]
            if cluster in cluster_labels:
                color = cluster_labels[cluster]['color']
                ax_cluster.fill_betweenx([depth[j], depth[j + 1]], 0, 1, facecolor=color, alpha=0.6, edgecolor='k')
        
        cluster_in_data = sorted(set(cluster_column) & set(cluster_labels.keys()))
        cluster_handles = [
            mpatches.Patch(facecolor=cluster_labels[c]['color'], edgecolor='k', label=cluster_labels[c]['label'])
            for c in cluster_in_data
        ]
        ax_cluster.legend(handles=cluster_handles, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(cluster_handles), fontsize=10, fancybox=True)
        #ax_cluster.set_xlabel('Clusters', fontsize=12)
        ax_cluster.set_title('Cluster Profile', fontsize=18)
        ax_cluster.invert_yaxis()
        #ax_cluster.xaxis.set_ticks_position('top')
        #ax_cluster.xaxis.set_label_position('top')
        #ax_cluster.tick_params(labelleft=False, labelbottom=False)
        
        # --- Format all axes ---
        for ax in axes:
            ax.set_ylim(max(depth), min(depth))
            ax.grid(True)
    
        fig.subplots_adjust(wspace=0.4)
        fig.suptitle('Lithology, Clusters and Logs by Depth', fontsize=16, y=1.05)
        plt.tight_layout()
        plt.show()





    
    
            
