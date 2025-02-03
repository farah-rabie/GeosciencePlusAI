import pandas as pd
import os
import lasio
from dlisio import dlis

class ToCSVFileFormat():

    def __init__(self):
        pass
    
    def las_to_csv(self, las_file_path, csv_file_path):
        
        """
        Convert a LAS file to a CSV file, set depth as the index, and add a lithology column.
        
        Parameters:
        - las_file_path (str): Path to the input LAS file.
        - csv_file_path (str): Path to the output CSV file.
        """
        
        # Read LAS file
        las = lasio.read(las_file_path)
        
        # Convert LAS data to DataFrame
        las_df = las.df()
        
        # Reset index to move depth into a column
        las_df.reset_index(inplace=True)
        
        # OPTIONAL
        ## Define lithology classification based on flags
        def classify_lithology(row):
            CARB_FLAG = row.get('CARB_FLAG', 0)
            SAND_FLAG = row.get('SAND_FLAG', 0)
            COAL_FLAG = row.get('COAL_FLAG', 0)
            VSH = row.get('VSH', 0)
            
            if CARB_FLAG == 1:
                return "Limestone"
            elif SAND_FLAG == 1:
                return "Sandstone"
            elif COAL_FLAG == 1:
                return "Coal"
            else:
                return "Unknown"
        
        ## Apply lithology classification
        las_df['Lithology'] = las_df.apply(classify_lithology, axis=1)
        
        # Set depth as index
        las_df.set_index('DEPTH', inplace=True)
        
        # Save DataFrame to CSV
        las_df.to_csv(csv_file_path, index=True)
        
        print(f"CSV file created at: {csv_file_path}")
    
    def dlis_to_csv(self, dlis_file_path, csv_file_path):
        
        """
        Convert a DLIS file to a CSV file.
        
        Parameters:
        - dlis_file_path (str): Path to the input DLIS file.
        - csv_file_path (str): Path to the output CSV file.
        """
        
        # Load the DLIS file
        f, *tail = dlis.load(dlis_file_path)
    
        # Initialise a dictionary to hold data
        data_dict = {}
    
        # Iterate through each frame and extract curve data from channels
        for frame in f.frames:
            for channel in frame.channels:
                # Get channel name and associated curve data
                channel_name = channel.name
                curve_data = channel.curves()  # Extract structured numpy array
                
                # Convert curve data to a list and add it to the dictionary
                data_dict[channel_name] = curve_data.tolist()
    
        # Convert the dictionary to a pandas DataFrame
        df = pd.DataFrame(data_dict)
    
        # Save DataFrame to CSV
        df.to_csv(csv_file_path, index=False)
        print(f"CSV file created at: {csv_file_path}")