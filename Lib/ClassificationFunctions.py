import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
from sklearn.utils import shuffle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

class DataProcessing():

    def __init__(self):
        # Store standardisation parameters (mean, std) for each column
        self.standardisation_params = {}
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
        fig, ax = plt.subplots(figsize=(10, 6))

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

    def process_well_data(self, file_paths, selected_columns, train_data=False, val_data=False, show_stats=False, show_rows=False):
        
        """
        Combines and processes data from multiple CSV files.
        
        Parameters:
            file_paths (list): List of file paths for the CSV files.
            selected_columns (list): List of columns to extract and process.
            train_data (bool): If True, combines all data into a single DataFrame + computes and stores standardisation parameters.
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
        
            individual_dataframes.append(df) # Append the processed DataFrame to the list of individual DataFrames
            combined_data.append(df)  # Add to the combined data list for further merging
    
        if train_data: # if train_data is True
            combined_df = pd.concat(combined_data, ignore_index=True) 
            columns_to_standardise = [col for col in combined_df.columns if col != 'LITHOLOGY'] # Exclude 'LITHOLOGY' from standardisation 
            self.compute_standardisation_params(combined_df[columns_to_standardise])  # compute standardisation parameters on selected columns only
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
            for column in df.columns 
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
        for column in standardised_df.columns: # Loop through columns to standardise, skipping 'DEPTH' and 'LITHOLOGY'
            if column == 'DEPTH':  # Skip the 'DEPTH' column
                continue
            if column == 'LITHOLOGY':  # Skip the 'LITHOLOGY' column
                continue
            # Apply standardisation using stored parameters
            mean_value = self.standardisation_params[column]["mean"]
            std_value = self.standardisation_params[column]["std"]
            standardised_df[column] = (standardised_df[column] - mean_value) / (std_value)

        if show_stats: # Show statistics if enabled
            print("Standardisation Parameters:")
            for column, params in self.standardisation_params.items():
                print(f"{column}: mean = {params['mean']}, std = {params['std']}")
            print("\nDescriptive Statistics of Standardised Data:")
            print(standardised_df.describe())  # Shows descriptive statistics for the DataFrame
        
        return standardised_df

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
        plt.grid(True)
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

    def plot_lithology_comparison(self, test_df, predicted_lithology):
        
        """
        Visualises true and predicted lithology profiles for a given depth using test data.
        
        Parameters:
            test_df (pd.DataFrame): The test DataFrame with 'DEPTH' and 'LITHOLOGY' columns.
            predicted_lithology (np.array): Predicted lithology labels from the model.
        """
        
        # Extract depth and true lithology from the test DataFrame
        depth_for_lithology = test_df['DEPTH'].values
        true_lithology = test_df['LITHOLOGY'].values
    
        # Define lithology labels with colors and hatch patterns
        lithology_labels = {
            'sandstone': {'color': '#ffff00', 'hatch': '..'},
            'marl': {'color': '#80ffff', 'hatch': ''}, 
            'limestone': {'color': '#4682B4', 'hatch': '++'},
            'coal': {'color': 'black', 'hatch': ''},
            'silt': {'color': '#7cfc00', 'hatch': '||'},
            'claystone': {'color': '#228B22', 'hatch': '--'}  
        }
        
        # Create subplots: True Lithology (left) and Predicted Lithology (right)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 8), sharey=True)
    
        # --- PLOT 1: True Lithology ---
        ax1 = axes[0]
        for j in range(len(depth_for_lithology) - 1):
            lith = true_lithology[j].lower()  # Convert to lowercase for matching
            if lith in lithology_labels:
                hatch = lithology_labels[lith]['hatch']
                color = lithology_labels[lith]['color']
                ax1.fill_betweenx([depth_for_lithology[j], depth_for_lithology[j + 1]], 0, 1, 
                                  facecolor=color, hatch=hatch, alpha=0.6)
    
        ax1.set_xlabel('True Lithology', fontsize=12)
        ax1.set_ylabel('Depth (m)', fontsize=12)
        ax1.set_title('True Lithology Profile', fontsize=14)
        ax1.invert_yaxis()  # Depth increases downward
    
        handles = []
        for lith, attrs in lithology_labels.items():
            patch = mpatches.Patch(facecolor=attrs['color'], hatch=attrs['hatch'], edgecolor='k', label=f'{lith}')
            handles.append(patch)
        ax1.legend(handles=handles, loc='best')
    
        # --- PLOT 2: Predicted Lithology ---
        ax2 = axes[1]
        for j in range(len(depth_for_lithology) - 1):
            lith = predicted_lithology[j].lower()  # Convert to lowercase for matching
            if lith in lithology_labels:
                hatch = lithology_labels[lith]['hatch']
                color = lithology_labels[lith]['color']
                ax2.fill_betweenx([depth_for_lithology[j], depth_for_lithology[j + 1]], 0, 1, 
                                  facecolor=color, hatch=hatch, alpha=0.6)
    
        ax2.set_xlabel('Predicted Lithology', fontsize=12)
        ax2.set_title('Predicted Lithology Profile', fontsize=14)
    
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

    def tune_random_forest(self, df_train, df_val, feature_columns, target_column, n_iter=20, cv=5):
        
        """
        Tune a Random Forest classifier using RandomizedSearchCV for lithology classification.
    
        Parameters:
        - df_train: DataFrame containing training data
        - df_val: DataFrame containing validation data
        - feature_columns: List of feature column names
        - target_column: Name of the target column
        - n_iter: Number of iterations for RandomizedSearchCV (default: 20)
        - cv: Number of cross-validation folds (default: 5)

        Returns:
        - best_rf: Trained Random Forest model with best parameters
        - best_params: Best hyperparameters found
        - accuracy: Accuracy on the test set
        - y_pred: Predicted values on the test set
        """

        # Extract features and target variables
        X_train = df_train[feature_columns]
        y_train = df_train[target_column]
        X_val = df_val[feature_columns]
        y_val = df_val[target_column]

        # Define hyperparameter grid
        param_dist = {
            'n_estimators': np.arange(50, 300, 50),
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
    
        # Initialise the classifier
        rf = RandomForestClassifier(random_state=42)

        # Randomised Search
        random_search = RandomizedSearchCV(rf, param_distributions=param_dist, 
                                           n_iter=n_iter, cv=cv, verbose=2, 
                                           n_jobs=-1, random_state=42)

        # Fit model
        random_search.fit(X_train, y_train)

        # Get best model
        best_rf = random_search.best_estimator_
        best_params = random_search.best_params_

        # Make predictions
        y_pred = best_rf.predict(X_val)

        # Evaluate performance
        accuracy = accuracy_score(y_val, y_pred)
        print(f"Validation Accuracy: {accuracy:.4f}")
        print("\nClassification Report:\n", classification_report(y_val, y_pred, zero_division=0))
        print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred))

        return best_rf, best_params, accuracy, y_pred

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

    def tune_svm(self, df_train, df_val, feature_columns, target_column, kernel='rbf', cv=5):
        """
        Tune and train an SVM classifier using GridSearchCV.
        
        Parameters:
        - df_train: DataFrame containing training data
        - df_val: DataFrame containing validation data
        - feature_columns: List of feature column names
        - target_column: Name of the target column
        - kernel: Kernel type for SVM ('linear', 'rbf', 'poly')
        - cv: Number of cross-validation folds
        
        Returns:
        - best_svm: Trained SVM model with best parameters
        - best_params: Best hyperparameters found
        - accuracy: Accuracy on the validation set
        - y_pred: Predicted values on validation set
        """
        X_train = df_train[feature_columns]
        y_train = df_train[target_column]
        X_val = df_val[feature_columns]
        y_val = df_val[target_column]

        # Parameter grid for tuning
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 0.1, 0.01, 0.001],
            'kernel': [kernel]
        }

        svm = SVC()

        grid_search = GridSearchCV(svm, param_grid, cv=cv, verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_svm = grid_search.best_estimator_
        best_params = grid_search.best_params_

        y_pred = best_svm.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)

        print(f"Validation Accuracy: {accuracy:.4f}")
        print("\nBest Parameters:", best_params)
        print("\nClassification Report:\n", classification_report(y_val, y_pred, zero_division=0))
        print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred))

        return best_svm, best_params, accuracy, y_pred

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
        

    
    
    
