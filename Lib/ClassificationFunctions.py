import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
from sklearn.utils import shuffle
from itertools import groupby

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from sklearn.svm import SVC

class DataProcessing():

    def __init__(self):
        # Store scaling parameters (mean, std) for each column
        self.scaling_params = {}
        self.lithology_labels = {
            'Sandstone': {'color': '#ffff00', 'hatch': '..'},
            'Marl': {'color': '#80ffff', 'hatch': ''}, 
            'Limestone': {'color': '#4682B4', 'hatch': '++'},
            'Coal': {'color': 'black', 'hatch': ''},
            'Silt': {'color': '#7cfc00', 'hatch': '||'},
            'Claystone': {'color': '#228B22', 'hatch': '--'}  
        }

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

class KNNClassification():
    
    def __init__(self):
        pass
        
    def tune_knn_k(self, df_train, df_val, feature_columns, target_column, k_max=None, weights='distance', metric='euclidean'):
        
        """
        Train a KNN classifier and tune the 'k' (number of neighbours) parameter using the validation set.
    
        Parameters:
        df_train (pd.DataFrame): The training set DataFrame.
        df_val (pd.DataFrame): The validation set DataFrame.
        feature_columns (list): List of feature columns in the DataFrame.
        target_column (str): The column name for the target variable.
        k_max (list): List of 'k' values to test (default is [1, 3, 5, 7, 9]).
        weights (str): Weight function for KNN (default is 'distance').
        metric (str): Distance metric to use (default is 'euclidean').
    
        Returns:
        best_k (int): The optimal number of neighbours (k).
        best_knn_classifier (KNeighboursClassifier): The trained KNN model with the best 'k'.
        best_accuracy (float): The accuracy score on the validation set with the best 'k'.
        """
        
        k_values = range(1, k_max + 1)
        accuracy_scores = []
    
        # Extract features and target variables
        X_train = df_train[feature_columns]
        y_train = df_train[target_column]
        X_val = df_val[feature_columns]
        y_val = df_val[target_column]
        
        best_k = None
        best_accuracy = 0
        best_knn_classifier = None
    
        # Loop over all k values and evaluate the model on the validation set
        for k in k_values:
            knn_classifier = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric)
            knn_classifier.fit(X_train, y_train)
    
            # Predict on the validation set
            y_val_pred = knn_classifier.predict(X_val)
    
            # Calculate accuracy on the validation set
            accuracy = accuracy_score(y_val, y_val_pred)
            accuracy_scores.append(accuracy)
            print(f"Accuracy for k={k}: {accuracy:.4f}")
    
            # Track the best k and the corresponding model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_k = k
                best_knn_classifier = knn_classifier
    
        print(f"\nBest k: {best_k} with accuracy: {best_accuracy:.4f}")

        plt.figure(figsize=(8, 5))
        plt.plot(k_values, accuracy_scores, marker='o', linestyle='-')
        plt.xlabel("Number of Neighbours (k)")
        plt.ylabel("Validation Accuracy")
        plt.title("KNN Hyperparameter Tuning: Accuracy vs k")
        plt.show()
    
        # Return the best k, the trained model, and the best accuracy
        return best_k, best_knn_classifier, best_accuracy

    def train_knn(self, df_train, feature_columns, target_column, k):
        
        """Train a KNN classifier using a DataFrame."""
        
        X_train = df_train[feature_columns].values
        y_train = df_train[target_column].values
        
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='euclidean')
        knn.fit(X_train, y_train)
        return knn

    def test_knn(self, knn, df_test, feature_columns, target_column):
        
        """Test the trained KNN model using a DataFrame and return accuracy."""
        
        X_test = df_test[feature_columns].values
        y_test = df_test[target_column].values
        
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy of KNN when tested is", accuracy)
        return accuracy, y_pred

    def plot_lithology_comparison(self, test_df, predicted_lithology, use_hatch=True, log_columns=None):
        """
        Visualises true and predicted lithology profiles for a given depth using test data.
    
        Parameters:
            test_df (pd.DataFrame): DataFrame with 'DEPTH' and 'LITHOLOGY' columns.
            predicted_lithology (np.array): Predicted lithology labels.
            use_hatch (bool): Whether to use hatch patterns.
            log_columns (list or None): Log column names to plot.
        """
    
        depth = test_df['DEPTH'].values
        true_lithology = test_df['LITHOLOGY'].values
    
        # Helper to group lithology intervals
        def group_intervals(depths, liths):
            grouped = []
            current = liths[0]
            start = depths[0]
            for i in range(1, len(liths)):
                if liths[i] != current:
                    grouped.append((start, depths[i - 1], current))
                    start = depths[i]
                    current = liths[i]
            grouped.append((start, depths[-1], current))
            return grouped
    
        true_blocks = group_intervals(depth, true_lithology)
        pred_blocks = group_intervals(depth, predicted_lithology)
    
        # Set up plot layout
        n_logs = len(log_columns) if log_columns else 0
        total_cols = n_logs + 2
        fig, axes = plt.subplots(nrows=1, ncols=total_cols, figsize=(3.5 * total_cols, 20), sharey=True)
    
        if total_cols == 1:
            axes = [axes]
    
        # --- Color palette for logs 
        log_colors = [
            "darkred", "royalblue", "forestgreen", "orange", "mediumpurple",
            "teal", "crimson", "slategray", "black"
        ]
        color_cycle = iter(log_colors)
    
        # --- Plot logs ---
        for i in range(n_logs):
            log_name = log_columns[i]
            ax = axes[i]
            if log_name not in test_df.columns:
                print(f"Warning: {log_name} not found in DataFrame.")
                continue
            color = next(color_cycle)
            ax.plot(test_df[log_name], test_df['DEPTH'], lw=2.0, color=color)
            ax.set_xlabel(log_name, fontsize=11)
            ax.set_title(f'{log_name} Log', fontsize=12)
            ax.invert_yaxis()
            ax.xaxis.set_ticks_position("top")
            ax.xaxis.set_label_position("top")
            if i != 0:
                ax.tick_params(labelleft=False)
            ax.tick_params(labelbottom=False)  # Remove bottom x-labels
    
        # --- True Lithology ---
        ax_true = axes[n_logs]
        for top, base, lith in true_blocks:
            lith = lith.lower()
            if lith in lithology_labels:
                color = lithology_labels[lith]['color']
                hatch = lithology_labels[lith]['hatch'] if use_hatch else ''
                ax_true.fill_betweenx([top, base], 0, 1, facecolor=color, hatch=hatch, edgecolor='k', alpha=0.6)
    
        ax_true.set_xlabel('True Lithology', fontsize=11)
        ax_true.set_title('True Lithology', fontsize=12)
        ax_true.invert_yaxis()
        ax_true.xaxis.set_ticks_position("top")
        ax_true.xaxis.set_label_position("top")
        ax_true.tick_params(labelleft=False, labelbottom=False)
    
        # --- Predicted Lithology ---
        ax_pred = axes[n_logs + 1]
        for top, base, lith in pred_blocks:
            lith = lith.lower()
            if lith in self.lithology_labels:
                color = self.lithology_labels[lith]['color']
                hatch = self.lithology_labels[lith]['hatch'] if use_hatch else ''
                ax_pred.fill_betweenx([top, base], 0, 1, facecolor=color, hatch=hatch, edgecolor='k', alpha=0.6)
    
        ax_pred.set_xlabel('Predicted Lithology', fontsize=11)
        ax_pred.set_title('Predicted Lithology', fontsize=12)
        ax_pred.invert_yaxis()
        ax_pred.xaxis.set_ticks_position("top")
        ax_pred.xaxis.set_label_position("top")
        ax_pred.tick_params(labelleft=False, labelbottom=False)
    
        # --- Legend ---
        handles = [
            mpatches.Patch(
                facecolor=props['color'],
                hatch=props['hatch'] if use_hatch else '',
                edgecolor='k',
                label=lith
            )
            for lith, props in self.lithology_labels.items()
        ]
        ax_true.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=len(handles), fontsize=10, fancybox=True)
    
        # Apply consistent y-limits
        for ax in axes:
            ax.set_ylim(max(depth), min(depth))  # Invert depth
            ax.grid(True)
    
        fig.subplots_adjust(wspace=0.5)
        fig.suptitle('Log Curves with True vs Predicted Lithology', fontsize=14, y=0.93)
        plt.tight_layout()
        plt.show()


    def plot_confusion_matrix(self, y_test, y_pred):
        
        """Plot confusion matrix for classification results."""
        
        # Define lithology labels directly within the function
        lithology_labels = ['Sandstone', 'Marl', 'Limestone', 'Coal', 'Silt', 'Claystone']
        
        # Generate the confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=lithology_labels)
        
        # Plot the heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=lithology_labels, yticklabels=lithology_labels)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

class RFClassification():
    
    def __init__(self):
        pass

    def train_random_forest(self, df_train, df_val, feature_columns, target_column, rf_params=None):
        """
        Trains a Random Forest classifier with user-specified hyperparameters.
    
        Parameters:
        - df_train: DataFrame containing training data
        - df_val: DataFrame containing validation data
        - feature_columns: List of feature column names
        - target_column: Name of the target column
        - rf_params: Dictionary of RandomForestClassifier hyperparameters (optional)
    
        Returns:
        - rf_model: Trained Random Forest model
        - rf_params: Used hyperparameters
        - accuracy: Accuracy on the validation set
        - y_pred: Predicted values on the validation set
        """
    
        # Extract features and target variables
        X_train = df_train[feature_columns]
        y_train = df_train[target_column]
        X_val = df_val[feature_columns]
        y_val = df_val[target_column]
    
        # Use default parameters if none provided
        if rf_params is None:
            rf_params = {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'auto',
                'bootstrap': True,
                'random_state': 42
            }
    
        # Initialise and train the model
        rf_model = RandomForestClassifier(**rf_params)
        rf_model.fit(X_train, y_train)
    
        # Predict on validation data
        y_pred = rf_model.predict(X_val)
    
        # Evaluate performance
        accuracy = accuracy_score(y_val, y_pred)
        print(f"Validation Accuracy: {accuracy:.4f}")
        print("\nClassification Report:\n", classification_report(y_val, y_pred, zero_division=0))
        print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred))
    
        return rf_model, rf_params, accuracy, y_pred

    def test_random_forest(self, trained_model, df_test, feature_columns, target_column):
       
        """
        Tests a trained Random Forest model on the test dataset.
    
        Parameters:
            trained_model (RandomForestClassifier): The trained Random Forest model.
            df_test (pd.DataFrame): The test dataset.
            feature_columns (list): List of feature column names.
            target_column (str): The target column name.
    
        Returns:
            float: Accuracy of the model on the test dataset.
            np.array: Predicted labels for the test set.
        """
        
        X_test = df_test[feature_columns]
        y_test = df_test[target_column]
    
        # Make predictions
        y_pred = trained_model.predict(X_test)
    
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Print classification report
        #print("Classification Report:\n", classification_report(y_test, y_pred))
        
        # Print confusion matrix
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
        return accuracy, y_pred

class SVMClassification():
    
    def __init__(self):
        pass

    def train_svm(self, df_train, df_val, feature_columns, target_column, svm_params=None):
        """
        Train an SVM classifier using user-specified hyperparameters.
    
        Parameters:
        - df_train: DataFrame containing training data
        - df_val: DataFrame containing validation data
        - feature_columns: List of feature column names
        - target_column: Name of the target column
        - svm_params: Dictionary of SVM hyperparameters (optional)
    
        Returns:
        - svm_model: Trained SVM model
        - used_params: Hyperparameters used for training
        - accuracy: Accuracy on the validation set
        - y_pred: Predicted values on the validation set
        """
    
        # Extract features and labels
        X_train = df_train[feature_columns]
        y_train = df_train[target_column]
        X_val = df_val[feature_columns]
        y_val = df_val[target_column]
    
        # Default parameters if none provided
        if svm_params is None:
            svm_params = {
                'C': 1.0,
                'kernel': 'rbf',
                'gamma': 'scale',
                'random_state': 42
            }
    
        # Initialize and train SVM model
        svm_model = SVC(**svm_params)
        svm_model.fit(X_train, y_train)
    
        # Predict on validation set
        y_pred = svm_model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
    
        # Print evaluation metrics
        print(f"Validation Accuracy: {accuracy:.4f}")
        print("\nUsed Parameters:", svm_params)
        print("\nClassification Report:\n", classification_report(y_val, y_pred, zero_division=0))
        print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred))
    
        return svm_model, svm_params, accuracy, y_pred

    def test_svm(self, trained_model, df_test, feature_columns, target_column):
        """
        Test a trained SVM model on test data.
        
        Returns:
        - accuracy: Accuracy score
        - y_pred: Predicted labels
        """
        X_test = df_test[feature_columns]
        y_test = df_test[target_column]

        y_pred = trained_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Test Accuracy: {accuracy:.4f}")
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

        return accuracy, y_pred
        

    
    
    
