import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
from itertools import groupby

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
        """
        Plot the distribution of lithology types for a selected well.

        Parameters:
            csv_file_path (str): Path to the well CSV file.
            well_name (str): Name of the well (used in the plot title).
            display (str): How to annotate bars — 'count', 'percentage', or 'both'.
        """
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

        # Custom legend — only show lithologies present in this well
        legend_handles = [
            Patch(facecolor=lithology_dict[lithology]["color"], hatch=lithology_dict[lithology]["hatch"], label=lithology)
            for lithology in lithology_dict
            if lithology in lithology_counts.index
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
        """
        Summarise available well logs for the selected well.

        Returns a DataFrame with count, mean, std, min, and max for each log.
        Logs that are completely empty are flagged with a warning.

        Parameters:
            csv_file_path (str): Path to the well CSV file.
        """
        df = pd.read_csv(csv_file_path)

        if df.empty:
            print("The CSV file is empty.")
            return

        log_cols = [col for col in df.columns if col not in ('LITHOLOGY', 'DEPTH')]

        empty_logs = [col for col in log_cols if df[col].dropna().empty]
        if empty_logs:
            print(f"Warning: the following logs are completely empty and excluded: {empty_logs}")

        log_cols = [col for col in log_cols if col not in empty_logs]

        stats = df[log_cols].agg(['count', 'mean', 'std', 'min', 'max']).T
        stats.columns = ['Count', 'Mean', 'Std', 'Min', 'Max']
        stats['Count'] = stats['Count'].astype(int)
        stats = stats.round(3)

        return stats

    def crossplot_2D(
        self,
        well_name,
        csv_file_path=None,
        df=None,
        x_col=None,
        y_col=None,
        x_data=None,
        y_data=None,
        color_col=None,
        filter_lithology=None,
        x_in_log=False,
        y_in_log=False
        ):
        """
        Crossplot of x vs y with optional coloring by lithology.
        
        Parameters:
            well_name (str): Name of the well.
            csv_file_path (str): Path to CSV file (optional if df or arrays are provided).
            df (pd.DataFrame): DataFrame with data (optional if CSV path is provided).
            x_col, y_col (str): Column names in df or CSV (ignored if using arrays).
            x_data, y_data (np.array): NumPy arrays with data (optional if CSV/df is used).
            color_col (str): Column name for coloring (lithology).
            filter_lithology (str): Optional, filter by a specific lithology.
            x_in_log, y_in_log (bool): Use log scale for axes.
        """
    
        # Load DataFrame if CSV provided
        if csv_file_path:
            df = pd.read_csv(csv_file_path)
    
        # Handle DataFrame input
        if df is not None:
            # Filter by lithology if requested
            if filter_lithology and color_col:
                df = df[df[color_col] == filter_lithology]
    
            # Drop NaNs
            drop_cols = [x_col, y_col] + ([color_col] if color_col else [])
            df = df.dropna(subset=drop_cols)
    
            x_data = df[x_col].values
            y_data = df[y_col].values
        else:
            # Arrays must be provided
            if x_data is None or y_data is None:
                raise ValueError("x_data and y_data arrays must be provided if no DataFrame or CSV is given.")
            x_data = x_data.reshape(-1)
            y_data = y_data.reshape(-1)
    
        if len(x_data) == 0 or len(y_data) == 0:
            raise ValueError("No data left after filtering; check your filter_lithology or input data.")
    
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
    
        if color_col is not None:
            if filter_lithology:  # Only one lithology after filtering
                color = self.lithology_labels.get(filter_lithology, {}).get('color', '#1f77b4')
                ax.scatter(x_data, y_data, label=filter_lithology, color=color, alpha=0.6)
            else:  # Multiple lithologies
                unique_lithologies = np.unique(df[color_col].values)
                for lith in unique_lithologies:
                    mask = df[color_col].values == lith
                    color = self.lithology_labels.get(lith, {}).get('color', '#1f77b4')
                    ax.scatter(x_data[mask], y_data[mask], label=str(lith), color=color, alpha=0.6)
        else:
            ax.scatter(x_data, y_data, color="blue", alpha=0.6)
    
        # Axis scales
        if x_in_log:
            ax.set_xscale("log")
        if y_in_log:
            ax.set_yscale("log")
    
        # Labels and title
        title_suffix = f" ({filter_lithology})" if filter_lithology else ""
        ax.set_xlabel(x_col if x_col else "X", fontsize=10)
        ax.set_ylabel(y_col if y_col else "Y", fontsize=10)
        ax.set_title(f"{x_col} vs {y_col} for Well {well_name}{title_suffix}", fontsize=12)
        if color_col or filter_lithology:
            ax.legend(title=color_col if color_col else "Lithology")
    
        plt.tight_layout()
        plt.show()


    def plot_well_logs_and_lithology(self, csv_file_path, well_name, logs=None):
        """
        Plot well logs alongside lithology for a selected well.

        Parameters:
            csv_file_path (str): Path to the well CSV file.
            well_name (str): Name of the well (used in the plot title).
            logs (list): List of log names to plot. Defaults to all available logs.
                         Available logs: 'BVW', 'KLOGH', 'VSH', 'DT', 'GR', 'NPHI', 'PEF', 'RHOB', 'RT'
        """

        # All supported logs with their display properties
        log_properties = {
            'BVW':   {'label': 'BVW (Bulk Volume Water)',   'color': 'darkred',      'log_scale': False},
            'KLOGH': {'label': 'KLOGH (Permeability)',      'color': 'royalblue',    'log_scale': False},
            'VSH':   {'label': 'VSH (Shale Volume)',        'color': 'forestgreen',  'log_scale': False},
            'DT':    {'label': 'DT (Travel Time)',          'color': 'orange',       'log_scale': True},
            'GR':    {'label': 'GR (Gamma Ray)',            'color': 'mediumpurple', 'log_scale': False},
            'NPHI':  {'label': 'NPHI (Neutron Porosity)',   'color': 'teal',         'log_scale': False},
            'PEF':   {'label': 'PEF (Photoelectric Factor)','color': 'crimson',      'log_scale': False},
            'RHOB':  {'label': 'RHOB (Bulk Density)',       'color': 'slategray',    'log_scale': False},
            'RT':    {'label': 'RT (Resistivity)',          'color': 'black',        'log_scale': True},
        }

        # Default to all logs if none specified
        if logs is None:
            logs = list(log_properties.keys())

        # Validate selected logs
        invalid = [l for l in logs if l not in log_properties]
        if invalid:
            raise ValueError(f"Invalid log(s): {invalid}. Available logs: {list(log_properties.keys())}")

        # Load data
        well_data = pd.read_csv(csv_file_path)
        DEPTH = well_data['DEPTH'].values

        # Replace negative values with zero
        for col in logs:
            if col in well_data.columns:
                mask = well_data[col] < 0
                if mask.any():
                    print(f"Warning: {mask.sum()} negative values found in {col}. Replacing with 0.")
                    well_data.loc[mask, col] = 0

        # Lithology data
        depth_for_lithology = well_data['DEPTH'].values
        lithology = well_data['LITHOLOGY'].values

        # Dynamic subplot grid: logs + 1 lithology column
        n_cols = len(logs) + 1
        fig_width = max(15, n_cols * 2)
        fig, axes = plt.subplots(1, n_cols, figsize=(fig_width, 15))
        if n_cols == 1:
            axes = [axes]

        # Plot each selected log
        for i, log in enumerate(logs):
            ax = axes[i]
            props = log_properties[log]
            ax.plot(well_data[log].values, DEPTH, color=props['color'], lw=2.5)
            ax.set_xlabel(props['label'], labelpad=10, fontsize=12)
            if props['log_scale']:
                ax.set_xscale('log')
            if i == 0:
                ax.set_ylabel('Depth (m)', labelpad=10, fontsize=12)
            else:
                ax.tick_params(labelleft=False)

        # Lithology panel (always last)
        ax_lith = axes[-1]
        intervals = []
        for key, group in groupby(enumerate(lithology), key=lambda x: x[1]):
            group = list(group)
            start_idx = group[0][0]
            end_idx = min(group[-1][0] + 1, len(depth_for_lithology) - 1)
            top = depth_for_lithology[start_idx]
            base = depth_for_lithology[end_idx]
            intervals.append((top, base, key))

        for top, base, lith in intervals:
            hatch = self.lithology_labels.get(lith, {}).get('hatch', '')
            color = self.lithology_labels.get(lith, {}).get('color', '#D2B48C')
            ax_lith.fill_betweenx([top, base], 0, 1, facecolor=color, hatch=hatch, edgecolor='k')

        handles = [
            mpatches.Patch(facecolor=attrs['color'], hatch=attrs['hatch'], edgecolor='k', label=lith)
            for lith, attrs in self.lithology_labels.items()
        ]
        ax_lith.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=len(handles), fancybox=True, fontsize=12)
        ax_lith.set_xlabel('Lithology', labelpad=20, fontsize=12)
        ax_lith.set_xticks([])
        ax_lith.tick_params(labelleft=False)

        # Common depth axis settings
        for ax in axes:
            ax.set_ylim(max(depth_for_lithology), min(depth_for_lithology))
            ax.xaxis.set_ticks_position("top")
            ax.xaxis.set_label_position("top")

        fig.subplots_adjust(wspace=0.5)
        fig.suptitle(f"Well Logs and Lithology for {well_name}", fontsize=16, y=1.02)
        plt.show()

# References
#Andy McDonald. (2020). Petrophysics-Python-Series/14 - Displaying Lithology Data.ipynb at master · andymcdgeo/Petrophysics-Python-Series. GitHub. https://github.com/andymcdgeo/Petrophysics-Python-Series/blob/master/14%20-%20Displaying%20Lithology%20Data.ipynb
