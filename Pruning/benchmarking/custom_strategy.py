# GA_BM/benchmarking/custom_strategy.py

import pandas as pd
import torch
from typing import Dict, List, Tuple, Any
import os


# from genetic_algorithm module as they are already set up for sensitivity scores
from genetic_algorithm.agents import PruningStrategyAgent, ModelPruningAgent, get_sensitivity_based_layer_ordering
from core.utils import get_layer_score_files_map


class CustomStrategyLoader:
    """
    Loads pruning strategies from a CSV file (e.g., Pareto front solutions)
    and prepares them for benchmarking.
    """
    def __init__(self, model_state_dict: Dict, score_dir_path: str):
        self.model_state_dict = model_state_dict
        self.score_dir_path = score_dir_path
        # Use sensitivity-based ordering to match how PruningStrategyAgent orders layers
        available_layers = [
            name for name in model_state_dict.keys()
            if ('features' in name or 'classifier' in name) and 'weight' in name
            and name in get_layer_score_files_map(score_dir_path, model_state_dict)
        ]
        self.prunable_layers = get_sensitivity_based_layer_ordering(available_layers, score_dir_path)
        if not self.prunable_layers:
            print("Warning: No prunable layers found with corresponding score files. Custom strategies may not apply correctly.")

    def load_strategies_from_csv(self, csv_filepath: str, best_only: bool = False) -> List[Dict[str, Any]]:
        """
        Loads pruning strategies (percentiles per layer) from a CSV file.
        The CSV is expected to contain columns like 'Layer_X_Gene' where X corresponds
        to the index of the layer in the `prunable_layers` list.

        Args:
            csv_filepath: Path to the CSV file containing pruning strategies.
            best_only: If True, only return the strategy with highest reliability.

        Returns:
            A list of dictionaries, where each dictionary represents one strategy
            and contains 'name' (optional) and 'percentiles' (list of floats).
        """
        if not os.path.exists(csv_filepath):
            raise FileNotFoundError(f"Strategy CSV file not found: {csv_filepath}")

        df = pd.read_csv(csv_filepath)
        
       
        if best_only and 'Estimated_Reliability' in df.columns:
            best_idx = df['Estimated_Reliability'].idxmax()
            df = df.iloc[[best_idx]]
            print(f"Selected best reliable strategy (index {best_idx}) with reliability: {df.iloc[0]['Estimated_Reliability']:.4f}")
        
        strategies = []

        # columns for strategy are named like 'Layer_features_0_0_weight_Gene'
        # We need to map them back to the self.prunable_layers order.
        strategy_gene_cols = [col for col in df.columns if '_Gene' in col]

        if not strategy_gene_cols:
            print("Warning: No '_Gene' columns found in the CSV. Assuming first N columns are percentiles.")
            # Fallback if no specific naming convention, assuming direct order
            for idx, row in df.iterrows():
                # Take first N columns as percentiles
                percentiles = row.iloc[:len(self.prunable_layers)].tolist()
                if len(percentiles) == len(self.prunable_layers):
                     # Also read actual sparsity if available
                     actual_sparsity = None
                     if 'Actual_Sparsity' in row:
                         actual_sparsity = float(row['Actual_Sparsity'])
                     
                     strategies.append({
                        'name': f"Custom_Strategy_{idx}",
                        'percentiles': percentiles,
                        'actual_sparsity': actual_sparsity
                    })
                else:
                    print(f"Skipping row {idx}: Number of percentiles ({len(percentiles)}) does not match number of prunable layers ({len(self.prunable_layers)}).")
        else:
            # Map CSV columns to prunable_layers based on name
           
            column_to_layer_map = {}
            for col_name in strategy_gene_cols:
                # Reconstruct the original layer name from the column name
               
                layer_part = col_name.replace('Layer_', '').replace('_Gene', '')
                # Replace '_weight' with '.weight' and other underscores with dots
                if '_weight' in layer_part:
                    layer_part = layer_part.replace('_weight', '.weight')
                    # Replace remaining underscores with dots
                    potential_layer_name = layer_part.replace('_', '.')
                else:
                    potential_layer_name = layer_part.replace('_', '.')
                
                if potential_layer_name in self.prunable_layers:
                    column_to_layer_map[col_name] = potential_layer_name
                else:
                    print(f"Warning: CSV column '{col_name}' does not exactly match any known prunable layer. Skipping.")


            if not column_to_layer_map or len(column_to_layer_map) != len(self.prunable_layers):
                 print(f"ERROR: Mismatch between CSV columns ({len(column_to_layer_map)} matched) and prunable layers ({len(self.prunable_layers)}).")
                 print(f"  Matched columns: {list(column_to_layer_map.keys())}")
                 print(f"  Expected layers: {self.prunable_layers}")
                 raise ValueError("Cannot load strategies: CSV column mapping failed")

            # CRITICAL FIX: Store percentiles as a DICTIONARY mapping layer_name → percentile
            # This way we don't rely on any ordering assumption!
            for idx, row in df.iterrows():
                # Create dictionary mapping layer name to percentile value
                percentile_dict = {}
                for csv_col, layer_name in column_to_layer_map.items():
                    percentile_dict[layer_name] = float(row[csv_col])

                # Verify we have values for all expected layers
                if len(percentile_dict) != len(self.prunable_layers):
                    print(f"WARNING: Strategy {idx} has {len(percentile_dict)} percentiles but expected {len(self.prunable_layers)}")

                # Also read actual sparsity if available
                actual_sparsity = None
                if 'Actual_Sparsity' in row:
                    actual_sparsity = float(row['Actual_Sparsity'])

                strategies.append({
                    'name': f"Custom_Strategy_{idx}",
                    'percentile_dict': percentile_dict,  # Dictionary, not list!
                    'actual_sparsity': actual_sparsity
                })
                print(f"  Loaded strategy {idx} with {len(percentile_dict)} layer-specific percentiles")

        if not strategies:
            raise ValueError("No valid pruning strategies could be loaded from the CSV file. Check format.")

        return strategies