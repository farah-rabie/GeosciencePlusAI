import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from itertools import groupby
from sklearn.linear_model import LinearRegression

class VisualiseWellData():

    def __init__(self):
        
        self.lithology_labels = {
            'Sandstone': {'color': '#ffff00', 'hatch': '..'},
            'Marl': {'color': '#80ffff', 'hatch': ''}, 
            'Limestone': {'color': '#4682B4', 'hatch': '++'},
            'Coal': {'color': 'black', 'hatch': ''},
            'Silt': {'color': '#7cfc00', 'hatch': '||'},
            'Claystone': {'color': '#228B22', 'hatch': '--'}  
        }

    def visualise_lithology_distribution(self, csv_file_path, well_name, display='count'):
        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_file_path)

        # Check if 'LITHOLOGY' column exists
        if 'LITHOLOGY' not in df.columns:
            print("Column 'LITHOLOGY' not found in the CSV file.")
            return

        # Get lithology distribution
        lithology_counts = df['LITHOLOGY'].value_counts()
        total = lithology_counts.sum()

        # Dictionary of lithology properties (color, hatch symbol)
        lithology_dict = self.lithology_labels

        # Plot the distribution
        fig, ax = plt.subplots(figsize=(10, 8))
        bars = []

        for lithology, count in lithology_counts.items():
            color = lithology_dict.get(lithology, {}).get('color', '#D2B48C')  # Default color
            hatch = lithology_dict.get(lithology, {}).get('hatch', '')  # Default hatch

            bar = ax.bar(lithology, count, color=color, hatch=hatch)
            bars.append((bar, count))

        # Add annotations
        for bar, count in bars:
            for rect in bar:
                percent = (count / total) * 100
                label = ''
                if display == 'count':
                    label = str(int(count))
                elif display == 'percentage':
                    label = f"{percent:.1f}%"
                elif display == 'both':
                    label = f"{int(count)}\n({percent:.1f}%)"

                ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), label,
                        ha='center', va='bottom', fontsize=10)

        # Custom legend
        legend_handles = [
            Patch(facecolor=lithology_dict[lithology]["color"], hatch=lithology_dict[lithology]["hatch"], label=lithology)
            for lithology in lithology_dict
        ]
        ax.legend(handles=legend_handles, title='Lithology', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Labels and formatting
        ax.set_ylabel('Count', fontsize=10)
        ax.set_xlabel('Lithology', fontsize=10)
        ax.set_title(f'Lithology Distribution for Well {well_name}', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
    def show_available_logs(self, csv_file_path):
        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_file_path)

        # Check if the file has any data
        if df.empty:
            print("The CSV file is empty.")
            return

        # Show the column names (which represent the available logs)
        print("Available logs in the data file:")

        # Iterate through each column and print statistics
        for column in df.columns:
            if column == 'LITHOLOGY':  # Skip the 'LITHOLOGY' column
                continue

            column_data = df[column]

            if column_data.dropna().empty:
                print(f"\n ⚠️ Warning: Column '{column}' is completely empty or contains only NaNs.")
                continue

            print(f"\nStatistics for '{column}':")

            # Calculate statistics
            count = column_data.count()
            mean = column_data.mean()
            std_dev = column_data.std()
            min_val = column_data.min()
            max_val = column_data.max()

            # Print the statistics
            print(f"  Count: {count}")
            print(f"  Mean: {mean:.3f}")
            print(f"  Standard Deviation: {std_dev:.3f}")
            print(f"  Min: {min_val}")
            print(f"  Max: {max_val}")

    def crossplot_2D(self, csv_file_path, well_name, x_col, y_col, x_in_log=False, y_in_log=False, color_col=None, filter_lithology=None):
        
        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_file_path)
    
        # Filter by lithology if requested
        if filter_lithology and color_col:
            df = df[df[color_col] == filter_lithology]
    
        # Check if the selected columns exist
        for col in [x_col, y_col, color_col] if color_col else [x_col, y_col]:
            if col not in df.columns:
                print(f"Column '{col}' not found in the CSV file.")
                return
    
        # Handle negative values for the selected columns
        for col in [x_col, y_col]:
            if (df[col] < 0).any():
                negative_count = (df[col] < 0).sum()
                df[col] = df[col].clip(lower=0)
                print(f"{negative_count} negative values in '{col}' have been clipped to 0.")
    
        # Drop rows with NaN in required columns
        drop_cols = [x_col, y_col] + ([color_col] if color_col else [])
        df = df.dropna(subset=drop_cols)
    
        # Extract data for regression
        x_data = df[x_col].values.reshape(-1, 1)
        y_data = df[y_col].values
    
        # Fit linear regression model
        model = LinearRegression()
        model.fit(x_data, y_data)
        y_pred = model.predict(x_data)
    
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
    
        # Plot scatter with lithology-based coloring
        if color_col and not filter_lithology:
            unique_liths = df[color_col].unique()
            for lith in unique_liths:
                mask = df[color_col] == lith
                ax.scatter(df.loc[mask, x_col], df.loc[mask, y_col], 
                           label=str(lith), alpha=0.6)
        else:
            ax.scatter(x_data, y_data, label='Data points', color='blue', alpha=0.5)
    
        # Plot regression line
        ax.plot(x_data, y_pred, color='red', linewidth=2, label='Linear fit')
    
        # Set log scales if needed
        if x_in_log:
            ax.set_xscale("log")
        if y_in_log:
            ax.set_yscale("log")
    
        # Add labels and title
        ax.set_xlabel(x_col, fontsize=10)
        ax.set_ylabel(y_col, fontsize=10)
        title_suffix = f" ({filter_lithology})" if filter_lithology else ""
        ax.set_title(f'{x_col} vs {y_col} for Well {well_name}{title_suffix}', fontsize=12)
        ax.legend(title=color_col if color_col else "Legend")
    
        plt.tight_layout()
        plt.show()

        # Example usage:
        # visualiser.crossplot_2D(
        #    well_data_path,
        #    well_name,
        #    x_col='RHOB',
        #    y_col='NPHI',
        #    color_col='LITHOLOGY',
        #    filter_lithology='Sandstone'
        #)

    def plot_well_logs_and_lithology(self, csv_file_path, well_name):
        # Load the well data CSV
        well_data = pd.read_csv(csv_file_path)

        # Extracting necessary well log columns
        DEPTH = well_data['DEPTH'].values
        BVW = well_data['BVW'].values
        KLOGH = well_data['KLOGH'].values
        VSH = well_data['VSH'].values
        DT = well_data['DT'].values
        GR = well_data['GR'].values
        NPHI = well_data['NPHI'].values
        PEF = well_data['PEF'].values
        RHOB = well_data['RHOB'].values
        RT = well_data['RT'].values

        # Lithology data
        lithology_data = well_data[['DEPTH', 'LITHOLOGY']]  
        depth_for_lithology = lithology_data['DEPTH'].values
        lithology = lithology_data['LITHOLOGY'].values

        # Replace negative values with zero
        for column_name, column_data in zip(
            ['DEPTH', 'BVW', 'KLOGH', 'VSH', 'DT', 'GR', 'NPHI', 'PEF', 'RHOB', 'RT'],
            [DEPTH, BVW, KLOGH, VSH, DT, GR, NPHI, PEF, RHOB, RT]
        ):
            negative_indices = column_data < 0
            if any(negative_indices):
                print(f"Warning: {sum(negative_indices)} negative values found in {column_name}. Replacing with 0.")
                column_data[negative_indices] = 0

        # Create subplots
        fig, axes = plt.subplots(2, 5, figsize=(15, 30))
        ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10 = axes.flatten()

        # Plot each log
        ax1.plot(BVW, DEPTH, color="darkred", lw=2.5)
        ax1.set_xlabel('BVW (Bulk Volume Water)', labelpad=10, fontsize=12)
        ax1.set_ylabel('Depth (m)', labelpad=10, fontsize=12)

        ax2.plot(KLOGH, DEPTH, color="royalblue", lw=2.5)
        ax2.set_xlabel('KLOGH (Permeability)', labelpad=10, fontsize=12)

        ax3.plot(VSH, DEPTH, color="forestgreen", lw=2.5)
        ax3.set_xlabel('VSH (Shale Volume)', labelpad=10, fontsize=12)

        ax4.plot(DT, DEPTH, color="orange", lw=2.5)
        ax4.set_xlabel('DT (Travel Time)', labelpad=10, fontsize=12)
        ax4.set_xscale('log')

        ax5.plot(GR, DEPTH, color="mediumpurple", lw=2.5)
        ax5.set_xlabel('GR (Gamma Ray)', labelpad=10, fontsize=12)

        ax6.plot(NPHI, DEPTH, color="teal", lw=2.5)
        ax6.set_xlabel('NPHI (Neutron Porosity)', labelpad=10, fontsize=12)

        ax7.plot(PEF, DEPTH, color="crimson", lw=2.5)
        ax7.set_xlabel('PEF (Photoelectric Factor)', labelpad=10, fontsize=12)

        ax8.plot(RHOB, DEPTH, color="slategray", lw=2.5)
        ax8.set_xlabel('RHOB (Bulk Density)', labelpad=10, fontsize=12)

        ax9.plot(RT, DEPTH, color="black", lw=2.5)
        ax9.set_xlabel('RT (Resistivity)', labelpad=10, fontsize=12)
        ax9.set_xscale('log')

        # --- Optimized lithology plotting ---
        intervals = []
        for key, group in groupby(enumerate(lithology), key=lambda x: x[1]):
            group = list(group)
            start_idx = group[0][0]
            end_idx = group[-1][0] + 1  # inclusive
            lith = key
            if end_idx >= len(depth_for_lithology):  # avoid out-of-range
                end_idx = len(depth_for_lithology) - 1
            top = depth_for_lithology[start_idx]
            base = depth_for_lithology[end_idx]
            intervals.append((top, base, lith))

        for top, base, lith in intervals:
            hatch = self.lithology_labels.get(lith, {}).get('hatch', '')
            color = self.lithology_labels.get(lith, {}).get('color', '#D2B48C')
            ax10.fill_betweenx([top, base], 0, 1, facecolor=color, hatch=hatch, edgecolor='k')

        # Lithology legend
        handles = [
            mpatches.Patch(facecolor=attrs['color'], hatch=attrs['hatch'], edgecolor='k', label=lith)
            for lith, attrs in self.lithology_labels.items()
        ]
        ax10.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(handles), fancybox=True, fontsize=12)
        ax10.set_xlabel('Lithology', labelpad=20, fontsize=12)
        ax10.set_xticks([])

        # Set common depth scale and orientation
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10]:
            ax.set_ylim(max(depth_for_lithology), min(depth_for_lithology))  # Invert
            ax.xaxis.set_ticks_position("top")
            ax.xaxis.set_label_position("top")

            # Hide y-axis tick labels for all but the first column
            for ax in [ax2, ax3, ax4, ax5, ax7, ax8, ax9, ax10]:
                ax.tick_params(labelleft=False)

        fig.subplots_adjust(wspace=0.5)
        fig.suptitle(f"Well Logs and Lithology for {well_name}", fontsize=16, y=0.92)
        plt.show()

# References
#Andy McDonald. (2020). Petrophysics-Python-Series/14 - Displaying Lithology Data.ipynb at master · andymcdgeo/Petrophysics-Python-Series. GitHub. https://github.com/andymcdgeo/Petrophysics-Python-Series/blob/master/14%20-%20Displaying%20Lithology%20Data.ipynb



