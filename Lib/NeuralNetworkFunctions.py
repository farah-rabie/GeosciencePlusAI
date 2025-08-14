import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
from sklearn.utils import shuffle
from itertools import groupby

import tensorflow as tf
import pickle
import os
from tensorflow.keras.callbacks import EarlyStopping

os.environ['PYTHONHASHSEED'] = '42'
np.random.seed(42)
tf.random.set_seed(42)
tf.keras.backend.set_floatx('float32')

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

    def process_well_data(self, file_paths, selected_columns, method='standard', train_data=False, val_data=False, show_stats=False, show_rows=False, filter_lithology=None):
        
        combined_data = []
        individual_dataframes = []
    
        for file in file_paths:
            df = pd.read_csv(file)  # Load the data
            df = df[selected_columns]  # Select relevant columns
            df.dropna(inplace=True)  # Drop missing values
    
            # --- New: filter by lithology if requested ---
            if filter_lithology and 'LITHOLOGY' in df.columns:
                df = df[df['LITHOLOGY'] == filter_lithology]
    
            numeric_columns = [col for col in selected_columns if col not in ['DEPTH', 'LITHOLOGY']]
            df = df[(df[numeric_columns] >= 0).all(axis=1)]  # Remove negative values
    
            # Compute logarithms for 'KLOGH' and 'RT' if necessary
            if 'KLOGH' in df.columns:
                df['log_KLOGH'] = np.log1p(df['KLOGH'])
                df.drop(columns=['KLOGH'], inplace=True)
            if 'RT' in df.columns:
                df['log_RT'] = np.log1p(df['RT'])
                df.drop(columns=['RT'], inplace=True)
            if 'GR' in df.columns:
                df = df[df['GR'] <= 150]
    
            individual_dataframes.append(df)
            combined_data.append(df)
    
        if train_data:
            combined_df = pd.concat(combined_data, ignore_index=True)
            columns_to_scale = [col for col in combined_df.columns if col != 'LITHOLOGY']
            self.compute_scaling_params(combined_df[columns_to_scale], method=method)
            if show_stats:
                print("\nDescriptive Statistics of Data:")
                print(combined_df.describe())
            if show_rows:
                print("\nFirst Few Rows of Data:")
                print(combined_df.head())
            combined_df = shuffle(combined_df, random_state=42)
            return combined_df
    
        elif val_data:
            combined_df = pd.concat(combined_data, ignore_index=True)
            if show_stats:
                print("\nDescriptive Statistics of Data:")
                print(combined_df.describe())
            if show_rows:
                print("\nFirst Few Rows of Data:")
                print(combined_df.head())
            combined_df = shuffle(combined_df, random_state=42)
            return combined_df
    
        else:
            for idx, df in enumerate(individual_dataframes):
                if show_stats:
                    print(f"\nDescriptive Statistics of DataFrame {idx + 1}:")
                    print(df.describe())
                if show_rows:
                    print(f"\nFirst Few Rows of DataFrame {idx + 1}:")
                    print(df.head())
            return individual_dataframes

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

class FeedforwardNeuralNetwork(tf.keras.Model):
    '''
    FeedForward Artificial Neural Networks
    '''
    def __init__(self,
                 input_NN,
                 hidden_NN,
                 activation_NN,
                 output_NN,
                 output_names=None,      # optional list of output names
                 name="FeedForward_Artificial_Neural_Networks", **kwargs):
        
        super(FeedforwardNeuralNetwork, self).__init__(name=name, **kwargs)

        ''' size and dimensions '''        
        self.input_NN = input_NN
        self.hidden_NN = hidden_NN
        self.activation_NN = activation_NN
        self.output_NN = output_NN
        self.output_names = output_names

        ''' neural network initialisation '''
        self.NN_Feedforward = self.NN_Feedforward_Init(
            nodes_input=input_NN,
            nodes_hidden=hidden_NN,
            nodes_output=output_NN,
            output_names=output_names,
            activation_func=activation_NN,
            dtype=tf.float32,
            train=True
        )

    def NN_Feedforward_Init(self, nodes_input, nodes_hidden, nodes_output, output_names, activation_func, dtype=tf.float32, train=True):
        '''
        Initialise the feedforward neural network for multiple inputs and outputs.
        '''
        activation = self.get_activation(activation_func)

        # Build Input layers
        inputs = [tf.keras.layers.Input(shape=(n,), name=f"input_{i+1}") 
                  for i, n in enumerate(nodes_input)]

        # Concatenate if multiple inputs
        if len(inputs) > 1:
            x = tf.keras.layers.Concatenate()(inputs)
        else:
            x = inputs[0]

        # Hidden layers
        for nodes in nodes_hidden:
            x = tf.keras.layers.Dense(nodes, activation=activation, trainable=train, dtype=tf.float32)(x)

        # Output layers (one per output branch)
        outputs = []
        for i, (n_out, out_name) in enumerate(zip(nodes_output, output_names)):
            outputs.append(tf.keras.layers.Dense(n_out, trainable=train, dtype=tf.float32, name=out_name)(x))

        return tf.keras.Model(inputs=inputs, outputs=outputs, name="NN_Feedforward")

    # define activation functions
    def requ(self, x):
        x = tf.cast(x, dtype=tf.float32)  # assuming x is float32
        zero_float32 = tf.constant(0.0, dtype=tf.float32)  # define 0.0 as float32
        return tf.maximum(zero_float32, tf.square(x))
        
    def fast_gelu(self, x):
        x = tf.cast(x, dtype=tf.float32)
        return x * tf.nn.sigmoid(1.702 * x)
    
    def get_activation(self, activation_name):
        """
        Retrieves activation function based on its name

        Parameters:
            activation_name (str): name of activation function

        Returns:
            activation_function: activation function corresponding to the name
        """
        if activation_name == 'relu':
            return tf.nn.relu
        elif activation_name == 'sigmoid':
            return tf.nn.sigmoid
        elif activation_name == 'tanh':
            return tf.nn.tanh
        elif activation_name == 'softmax':
            return tf.nn.softmax
        elif activation_name == 'requ':
            return self.requ
        elif activation_name == 'fast_gelu':
            return self.fast_gelu
        elif activation_name == 'leaky_relu':
            return tf.keras.layers.LeakyReLU(alpha=0.2)
        # add more activation functions as needed
        else:
            raise ValueError(f"Unsupported activation function: {activation_name}")

    def call(self, inputs):
        return self.NN_Feedforward(inputs)

    class CustomCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            '''
            Print all training and validation losses/metrics at the end of each epoch.
            '''
            if logs is None:
                return
            print(f"\nEpoch {epoch + 1}:")
            for key, value in logs.items():
                print(f"{key}: {value:.4f}")

    # loss function options
    def mae(self, y_true, y_pred):
        '''
        MAE (Mean Absolute Error) Loss function
        '''
        return tf.reduce_mean(tf.abs(y_true - y_pred))
    
    def mse(self, y_true, y_pred):
        '''
        MSE (Mean Squared Error) Loss function
        '''
        return tf.reduce_mean(tf.square(y_true - y_pred))
    
    def L1L2_loss(self,y_true,y_pred):
        '''
        L1L2 Loss function - combines MAE and MSE
        '''
        delta = 1.e-3; x = y_pred-y_true
        loss = tf.where(tf.abs(x)>delta,tf.abs(x),tf.pow(x,2)/(2*delta)+delta/2)-0.5*delta
        return tf.reduce_mean(loss,axis=-1) 

    def compileNNFeedforward(self, l_rate, beta_1, beta_2, epsilon,
                             loss_type='l1l2', metrics=None):
        """
        Compile the Feedforward NN model with flexible multi-output support.
    
        Args:
            l_rate (float): Learning rate.
            beta_1 (float): Beta 1 parameter for optimizer.
            beta_2 (float): Beta 2 parameter for optimizer.
            epsilon (float): Epsilon for optimizer.
            loss_type (str): One of 'l1l2', 'mae', or 'mse'.
        """
        optimizer = tf.keras.optimizers.Nadam(
            learning_rate=l_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon
        )
    
        # Pick loss function
        if loss_type.lower() == 'l1l2':
            chosen_loss = self.L1L2_loss
        elif loss_type.lower() == 'mae':
            chosen_loss = self.mae
        elif loss_type.lower() == 'mse':
            chosen_loss = self.mse
        else:
            raise ValueError("loss_type must be 'l1l2', 'mae', or 'mse'")
    
        # Apply same loss to all outputs
        loss_dict = {name: chosen_loss for name in self.output_names}

        self.NN_Feedforward.compile(optimizer=optimizer, loss=loss_dict)

    def train(self, x_train, y_train, epochs, batch_size, verbose,
              validation_split=0.2, x_val=None, y_val=None, use_multiprocessing=True):
        """
        Train the Feedforward NN model.
    
        Args:
            x_train, y_train: Training data (arrays or lists for multi-input/output)
            epochs (int)
            batch_size (int)
            verbose (int)
            validation_split (float): Used only if x_val and y_val are not given.
            x_val, y_val: Optional explicit validation data.
            use_multiprocessing (bool)
        """
        callback = self.CustomCallback()
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=200,
            min_delta=1.e-6, verbose=0, mode='min',
            restore_best_weights=True
        )
    
        # Choose validation method
        if x_val is not None and y_val is not None:
            validation_arg = {'validation_data': (x_val, y_val)}
        else:
            validation_arg = {'validation_split': validation_split}
    
        hist_NNnet = self.NN_Feedforward.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            use_multiprocessing=use_multiprocessing,
            callbacks=[callback, early_stopping],
            **validation_arg
        )
    
        print("Number of epochs when early stopping triggered:", early_stopping.stopped_epoch)
        return hist_NNnet
