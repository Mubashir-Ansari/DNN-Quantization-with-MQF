"""Enhanced fault analysis with comprehensive statistics."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from scipy import stats
import pickle
import os
from datetime import datetime


@dataclass
class FaultAnalysisResult:
    """Container for fault analysis results."""
    method_name: str
    fault_levels: List[int]
    mean_accuracies: List[float]
    std_accuracies: List[float]
    median_accuracies: List[float]
    min_accuracies: List[float]
    max_accuracies: List[float]
    raw_accuracies: List[List[float]]  # All repetitions for each fault level
    confidence_intervals: List[Tuple[float, float]]
    method_type: str = "Unknown"  # e.g., "Unstructured", "Structured", "Search"


class ComprehensiveFaultAnalyzer:
    """Enhanced fault analyzer with statistical analysis."""
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize fault analyzer.
        
        Args:
            confidence_level: Confidence level for confidence intervals
        """
        self.confidence_level = confidence_level
        self.results = {}
        
    def analyze_method_results(self, method_name: str, fault_results: Dict[int, List[float]], 
                             method_type: str = "Unknown") -> FaultAnalysisResult:
        """
        Analyze fault injection results for a single method.
        
        Args:
            method_name: Name of the pruning method
            fault_results: Dict mapping fault levels to list of accuracy values
            method_type: Type of method (Unstructured, Structured, Search, etc.)
            
        Returns:
            FaultAnalysisResult with comprehensive statistics
        """
        fault_levels = sorted(fault_results.keys())
        mean_accuracies = []
        std_accuracies = []
        median_accuracies = []
        min_accuracies = []
        max_accuracies = []
        confidence_intervals = []
        raw_accuracies = []
        
        for fault_level in fault_levels:
            accuracies = fault_results[fault_level]
            
            # Validate accuracy data - catch corruption early
            invalid_values = [acc for acc in accuracies if acc < 0 or acc > 100]
            if invalid_values:
                print(f"WARNING: Invalid accuracy values detected for {method_name} at {fault_level} faults:")
                print(f"  Invalid values: {invalid_values}")
                print(f"  This indicates data corruption in reliability testing")
                # Filter out invalid values
                accuracies = [acc for acc in accuracies if 0 <= acc <= 100]
                if not accuracies:
                    print(f"  All values invalid - using [0.0] as fallback")
                    accuracies = [0.0]
            
            raw_accuracies.append(accuracies)
            
            if len(accuracies) == 0:
                # Handle empty results
                mean_accuracies.append(0.0)
                std_accuracies.append(0.0)
                median_accuracies.append(0.0)
                min_accuracies.append(0.0)
                max_accuracies.append(0.0)
                confidence_intervals.append((0.0, 0.0))
                continue
            
            # Basic statistics
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies, ddof=1) if len(accuracies) > 1 else 0.0
            median_acc = np.median(accuracies)
            min_acc = np.min(accuracies)
            max_acc = np.max(accuracies)
            
            # Confidence interval
            if len(accuracies) > 1:
                ci = stats.t.interval(
                    self.confidence_level, 
                    len(accuracies) - 1,
                    loc=mean_acc,
                    scale=stats.sem(accuracies)
                )
            else:
                ci = (mean_acc, mean_acc)
            
            mean_accuracies.append(mean_acc)
            std_accuracies.append(std_acc)
            median_accuracies.append(median_acc)
            min_accuracies.append(min_acc)
            max_accuracies.append(max_acc)
            confidence_intervals.append(ci)
        
        result = FaultAnalysisResult(
            method_name=method_name,
            fault_levels=fault_levels,
            mean_accuracies=mean_accuracies,
            std_accuracies=std_accuracies,
            median_accuracies=median_accuracies,
            min_accuracies=min_accuracies,
            max_accuracies=max_accuracies,
            raw_accuracies=raw_accuracies,
            confidence_intervals=confidence_intervals,
            method_type=method_type
        )
        
        self.results[method_name] = result
        return result
    
    def compare_methods(self, results: List[FaultAnalysisResult]) -> pd.DataFrame:
        """
        Create a comprehensive comparison table of methods.
        
        Args:
            results: List of FaultAnalysisResult objects
            
        Returns:
            DataFrame with comparison statistics
        """
        comparison_data = []
        
        for result in results:
            for i, fault_level in enumerate(result.fault_levels):
                row = {
                    'Method': result.method_name,
                    'Method_Type': result.method_type,
                    'Fault_Level': fault_level,
                    'Mean_Accuracy': result.mean_accuracies[i],
                    'Std_Accuracy': result.std_accuracies[i],
                    'Median_Accuracy': result.median_accuracies[i],
                    'Min_Accuracy': result.min_accuracies[i],
                    'Max_Accuracy': result.max_accuracies[i],
                    'CI_Lower': result.confidence_intervals[i][0],
                    'CI_Upper': result.confidence_intervals[i][1],
                    'Num_Repetitions': len(result.raw_accuracies[i])
                }
                comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def generate_statistical_report(self, results: List[FaultAnalysisResult], 
                                  output_dir: str) -> str:
        """
        Generate a comprehensive statistical report.
        
        Args:
            results: List of analysis results
            output_dir: Directory to save the report
            
        Returns:
            Path to the generated report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f"fault_analysis_report_{timestamp}.txt")
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\\n")
            f.write("COMPREHENSIVE FAULT INJECTION ANALYSIS REPORT\\n")
            f.write("=" * 80 + "\\n\\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"Confidence Level: {self.confidence_level * 100:.1f}%\\n\\n")
            
            # Method summary
            f.write("METHOD SUMMARY:\\n")
            f.write("-" * 40 + "\\n")
            for result in results:
                f.write(f"• {result.method_name} ({result.method_type})\\n")
                f.write(f"  Fault levels tested: {len(result.fault_levels)}\\n")
                f.write(f"  Range: {min(result.fault_levels)} - {max(result.fault_levels)} faults\\n")
                f.write(f"  Repetitions per level: {len(result.raw_accuracies[0]) if result.raw_accuracies else 0}\\n\\n")
            
            # Detailed analysis for each method
            for result in results:
                f.write("=" * 60 + "\\n")
                f.write(f"DETAILED ANALYSIS: {result.method_name}\\n")
                f.write("=" * 60 + "\\n\\n")
                
                # Statistics table
                f.write("Fault Level | Mean±Std     | Median | Min-Max    | 95% CI\\n")
                f.write("-" * 60 + "\\n")
                
                for i, fault_level in enumerate(result.fault_levels):
                    mean_acc = result.mean_accuracies[i]
                    std_acc = result.std_accuracies[i]
                    median_acc = result.median_accuracies[i]
                    min_acc = result.min_accuracies[i]
                    max_acc = result.max_accuracies[i]
                    ci_lower, ci_upper = result.confidence_intervals[i]
                    
                    f.write(f"{fault_level:11d} | {mean_acc:5.2f}±{std_acc:5.2f} | "
                           f"{median_acc:6.2f} | {min_acc:4.1f}-{max_acc:4.1f} | "
                           f"[{ci_lower:5.2f}, {ci_upper:5.2f}]\\n")
                
                f.write("\\n")
                
                # Degradation analysis
                if len(result.mean_accuracies) > 1:
                    baseline_acc = result.mean_accuracies[0]  # 0 faults
                    final_acc = result.mean_accuracies[-1]    # Highest fault level
                    degradation = baseline_acc - final_acc
                    relative_degradation = (degradation / baseline_acc) * 100 if baseline_acc > 0 else 0
                    
                    f.write(f"DEGRADATION ANALYSIS:\\n")
                    f.write(f"Baseline Accuracy (0 faults): {baseline_acc:.2f}%\\n")
                    f.write(f"Final Accuracy ({result.fault_levels[-1]} faults): {final_acc:.2f}%\\n")
                    f.write(f"Absolute Degradation: {degradation:.2f} percentage points\\n")
                    f.write(f"Relative Degradation: {relative_degradation:.1f}%\\n\\n")
        
        return report_path
    
    def plot_comprehensive_comparison(self, results: List[FaultAnalysisResult], 
                                    output_dir: str, 
                                    plot_confidence_intervals: bool = True,
                                    plot_individual_runs: bool = False) -> str:
        """
        Create comprehensive comparison plots.
        
        Args:
            results: List of analysis results
            output_dir: Directory to save plots
            plot_confidence_intervals: Whether to show confidence intervals
            plot_individual_runs: Whether to show individual run results
            
        Returns:
            Path to the generated plot
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comprehensive Fault Injection Analysis', fontsize=16, fontweight='bold')
        
        # Color palette for different methods - distinct colors for each method
        method_colors = {
            'Custom_Strategy_83': '#1f77b4',  # Blue
            'wanda': '#ff7f0e',               # Orange  
            'snip': '#2ca02c',                # Green
            'grasp': '#d62728',               # Red
            'SPSD_inspired': '#9467bd',       # Purple
            'EVOP_inspired': '#8c564b',       # Brown
            'PDP_inspired': '#e377c2',        # Pink
            'HRank_inspired': '#7f7f7f',      # Gray
            'magnitude_global': '#bcbd22',    # Olive
            'magnitude_layerwise': '#17becf', # Cyan
            'random_global': '#ff9896',       # Light red
            'random_layerwise': '#98df8a',    # Light green
            'DEGRAPH': '#c5b0d5',             # Light purple
            'baseline': '#000000'             # Black
        }
        
        # Fallback colors for any additional methods
        fallback_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Plot 1: Mean accuracy with error bars
        ax1 = axes[0, 0]
        for i, result in enumerate(results):
            # Get method-specific color or fallback
            color = method_colors.get(result.method_name, fallback_colors[i % len(fallback_colors)])
            
            ax1.errorbar(result.fault_levels, result.mean_accuracies, 
                        yerr=result.std_accuracies, 
                        label=f"{result.method_name} ({result.method_type})",
                        marker='o', linewidth=2, markersize=6, color=color,
                        capsize=4, capthick=2)
        
        ax1.set_xlabel('Number of Injected Faults')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Mean Accuracy ± Standard Deviation')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Confidence intervals
        ax2 = axes[0, 1]
        for i, result in enumerate(results):
            # Get method-specific color or fallback
            color = method_colors.get(result.method_name, fallback_colors[i % len(fallback_colors)])
            
            ci_lower = [ci[0] for ci in result.confidence_intervals]
            ci_upper = [ci[1] for ci in result.confidence_intervals]
            
            ax2.fill_between(result.fault_levels, ci_lower, ci_upper, 
                           alpha=0.3, color=color)
            ax2.plot(result.fault_levels, result.mean_accuracies, 
                    label=f"{result.method_name}", marker='o', 
                    linewidth=2, color=color)
        
        ax2.set_xlabel('Number of Injected Faults')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title(f'{self.confidence_level*100:.0f}% Confidence Intervals')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Relative degradation
        ax3 = axes[1, 0]
        for i, result in enumerate(results):
            if len(result.mean_accuracies) > 1:
                baseline_acc = result.mean_accuracies[0]
                relative_degradation = []
                for acc in result.mean_accuracies:
                    rel_deg = ((baseline_acc - acc) / baseline_acc) * 100 if baseline_acc > 0 else 0
                    relative_degradation.append(rel_deg)
                
                # Get method-specific color or fallback
                color = method_colors.get(result.method_name, fallback_colors[i % len(fallback_colors)])
                
                ax3.plot(result.fault_levels, relative_degradation, 
                        label=f"{result.method_name}", marker='s', 
                        linewidth=2, color=color)
        
        ax3.set_xlabel('Number of Injected Faults')
        ax3.set_ylabel('Relative Degradation (%)')
        ax3.set_title('Relative Performance Degradation')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Box plots for selected fault levels
        ax4 = axes[1, 1]
        if results and len(results[0].fault_levels) > 1:
            # Select a few fault levels for box plot
            selected_levels = [0, len(results[0].fault_levels)//2, len(results[0].fault_levels)-1]
            box_data = []
            box_labels = []
            
            for level_idx in selected_levels:
                if level_idx < len(results[0].fault_levels):
                    fault_level = results[0].fault_levels[level_idx]
                    for result in results:
                        if level_idx < len(result.raw_accuracies):
                            box_data.append(result.raw_accuracies[level_idx])
                            box_labels.append(f"{result.method_name}\\n({fault_level} faults)")
            
            if box_data:
                bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True)
                patch_idx = 0
                for level_idx in selected_levels:
                    if level_idx < len(results[0].fault_levels):
                        for i, result in enumerate(results):
                            if level_idx < len(result.raw_accuracies) and patch_idx < len(bp['boxes']):
                                color = method_colors.get(result.method_name, fallback_colors[i % len(fallback_colors)])
                                bp['boxes'][patch_idx].set_facecolor(color)
                                patch_idx += 1
        
        ax4.set_ylabel('Accuracy (%)')
        ax4.set_title('Distribution Comparison (Selected Fault Levels)')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, f"comprehensive_fault_analysis_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.savefig(plot_path.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def export_results(self, results: List[FaultAnalysisResult], 
                      output_dir: str) -> Tuple[str, str]:
        """
        Export results to CSV and pickle files.
        
        Args:
            results: List of analysis results
            output_dir: Directory to save files
            
        Returns:
            Tuple of (csv_path, pickle_path)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export to CSV
        df = self.compare_methods(results)
        csv_path = os.path.join(output_dir, f"fault_analysis_data_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        
        # Export to pickle
        pickle_path = os.path.join(output_dir, f"fault_analysis_results_{timestamp}.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
        
        return csv_path, pickle_path


def load_fault_analysis_results(pickle_path: str) -> List[FaultAnalysisResult]:
    """Load fault analysis results from pickle file."""
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)