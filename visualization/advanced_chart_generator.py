#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Chart Generator - Generate CPU-Disk correlation charts according to documentation requirements
Implement visualization of statistical analysis methods, including correlation heatmaps
Fixed CSV field consistency issues, using unified field access interface
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.axes import Axes
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from typing import Dict, List, Tuple, Optional
import os
import sys
from pathlib import Path

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from visualization.chart_style_config import UnifiedChartStyle
from visualization.device_manager import DeviceManager
from utils.ena_field_accessor import ENAFieldAccessor
from utils.unified_logger import get_logger
from utils.csv_data_processor import CSVDataProcessor
from utils.unit_converter import UnitConverter

logger = get_logger(__name__)


class AdvancedChartGenerator(CSVDataProcessor):
    """Advanced Chart Generator - Based on unified CSV data processor"""
    
    @staticmethod
    def _calculate_ena_delta_series(df: pd.DataFrame, field: str) -> pd.Series:
        """
        Calculate ENA counter field delta series relative to baseline
        
        Args:
            df: DataFrame with ENA data
            field: ENA field name
        
        Returns:
            pd.Series: Delta series relative to baseline
        """
        if field not in df.columns or len(df) < 2:
            return pd.Series(0, index=df.index)
        
        baseline = int(df[field].iloc[0])
        delta_series = (df[field] - baseline).clip(lower=0)
        return delta_series
    
    def __init__(self, data_file: str, output_dir: str = None):
        """
        Initialize chart generator
        
        Args:
            data_file: Data file path
            output_dir: Output directory (will be adjusted to use REPORTS_DIR)
        """
        super().__init__()  # Initialize CSV data processor
        
        self.data_file = data_file
        if output_dir:
            self.output_dir = os.getenv('REPORTS_DIR', output_dir)
        else:
            # Derive reports directory from CSV file path: logs -> current -> reports
            base_dir = os.path.dirname(os.path.dirname(data_file))
            self.output_dir = os.getenv('REPORTS_DIR', os.path.join(base_dir, 'reports'))
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        try:
            self.unit_converter = UnitConverter()
        except:
            self.unit_converter = None
        
        # Set chart style - use unified style configuration
        UnifiedChartStyle.setup_matplotlib()
        
        # Using English label system directly
        self.font_manager = None
    
    def _log_error(self, operation: str, error: Exception) -> None:
        """Unified error log format"""
        logger.error(f"❌ {operation} failed: {str(error)}")
    
    def _log_success(self, operation: str) -> None:
        """Unified success log format"""
        logger.info(f"✅ {operation} completed successfully")
            
    def _get_localized_text(self, chinese_text: str, english_text: str) -> str:
        """Get localized text"""
        if self.font_manager:
            return self.font_manager.get_label(chinese_text, english_text)
        return english_text  # Fallback to English
        
    def _check_device_configured(self, logical_name: str) -> bool:
        """Check if device is configured and has data"""
        if self.df is None:
            return False
        
        # Check if device exists by column name prefix
        device_cols = [col for col in self.df.columns if col.startswith(f'{logical_name}_')]
        return len(device_cols) > 0
    
    def _get_device_columns_safe(self, logical_name: str, metric_suffix: str) -> List[str]:
        """Safely get device columns, only return existing columns"""
        if not self._check_device_configured(logical_name):
            return []
        
        return self.get_device_columns_safe(logical_name, metric_suffix)

    def load_data(self) -> bool:
        """Load data"""
        try:
            success = self.load_csv_data(self.data_file)
            if success:
                self.clean_data()  # Clean data
                logger.info(f"✅ Data loaded successfully: {len(self.df)} rows")
                self.print_field_info()  # Print field information for debugging
            return success
        except Exception as e:
            logger.error(f"❌ Data loading failed: {e}")
            return False
    
    def print_field_info(self):
        """Print field information for debugging"""
        if self.df is not None:
            logger.info(f"📊 Data field information: {list(self.df.columns)}")
            logger.info(f"📊 Data shape: {self.df.shape}")
        else:
            logger.warning("⚠️ Data not loaded")
    
    def get_field_name_safe(self, field_name: str) -> Optional[str]:
        """Safely get field name"""
        if self.df is None:
            return None
        
        # Direct match
        if field_name in self.df.columns:
            return field_name
        
        # Fuzzy match
        for col in self.df.columns:
            if field_name.lower() in col.lower():
                return col
        
        return None
    
    def generate_pearson_correlation_charts(self) -> List[str]:
        """Generate Pearson correlation charts"""
        if not self.load_data():
            return []
        
        print("📊 Generating Pearson correlation charts...")
        chart_files = []
        
        # Check Device configuration - use unified method
        accounts_configured = DeviceManager.is_accounts_configured(self.df)
        
        # Use device_manager for field access
        cpu_iowait_field = 'cpu_iowait'
        if cpu_iowait_field not in self.df.columns:
            print("⚠️ CPU I/O Wait field not found, skipping correlation analysis")
            return []
        
        # Get Device fields - use DeviceManager
        device_util_cols = []
        device_aqu_cols = []
        device_await_cols = []
        
        # DATA device (must exist)
        for col in self.df.columns:
            if col.startswith('data_') and col.endswith('_util'):
                device_util_cols.append(col)
            elif col.startswith('data_') and col.endswith('_aqu_sz'):
                device_aqu_cols.append(col)
            elif col.startswith('data_') and col.endswith('_avg_await'):
                device_await_cols.append(col)
        
        # ACCOUNTS device (optional)
        if accounts_configured:
            for col in self.df.columns:
                if col.startswith('accounts_') and col.endswith('_util'):
                    device_util_cols.append(col)
                elif col.startswith('accounts_') and col.endswith('_aqu_sz'):
                    device_aqu_cols.append(col)
                elif col.startswith('accounts_') and col.endswith('_avg_await'):
                    device_await_cols.append(col)
        
        # Build plot configuration
        plot_configs = []
        
        for util_col in device_util_cols:
            device_name = util_col.split('_')[0].upper()
            plot_configs.append((cpu_iowait_field, util_col, f'CPU I/O Wait vs {device_name} Device Utilization'))
        
        for aqu_col in device_aqu_cols:
            device_name = aqu_col.split('_')[0].upper()
            plot_configs.append((cpu_iowait_field, aqu_col, f'CPU I/O Wait vs {device_name} Device Queue Length'))
        
        for await_col in device_await_cols:
            device_name = await_col.split('_')[0].upper()
            plot_configs.append((cpu_iowait_field, await_col, f'CPU I/O Wait vs {device_name} Device Latency'))
        
        if not plot_configs:
            print("  ⚠️ No configured devices, skipping Pearson correlation chart generation")
            return []
        
        # Dynamically create subplot layout
        total_plots = len(plot_configs)
        if total_plots <= 3:
            rows, cols = 1, total_plots
        elif total_plots <= 6:
            rows, cols = 2, 3
        else:
            rows, cols = 2, 4
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        
        # Ensure axes is always 2D array for unified processing
        if total_plots == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Using English title directly
        fig.suptitle('CPU-Disk Pearson Correlation Analysis', fontsize=UnifiedChartStyle.FONT_CONFIG['title_size'], fontweight='bold')
        
        # Generate each subplot
        plot_idx = 0
        for i in range(rows):
            for j in range(cols):
                if plot_idx < len(plot_configs):
                    cpu_col, disk_col, title = plot_configs[plot_idx]
                    ax: Axes = axes[i, j]
                    
                    try:
                        # Safely get data
                        cpu_data = self.df[cpu_col] if cpu_col in self.df.columns else pd.Series(dtype=float)
                        disk_data = self.df[disk_col] if disk_col in self.df.columns else pd.Series(dtype=float)
                        
                        if len(cpu_data) > 0 and len(disk_data) > 0:
                            # Calculate correlation
                            corr, p_value = stats.pearsonr(cpu_data, disk_data)
                            
                            # Draw scatter plot
                            ax.scatter(cpu_data, disk_data, alpha=0.6, s=20, color=UnifiedChartStyle.COLORS['data_primary'])
                            
                            # Add trend line
                            z = np.polyfit(cpu_data, disk_data, 1)
                            p = np.poly1d(z)
                            ax.plot(cpu_data, p(cpu_data), color=UnifiedChartStyle.COLORS['critical'], linestyle='--', alpha=0.8, linewidth=2)
                            
                            ax.set_xlabel('CPU I/O Wait (%)', fontsize=UnifiedChartStyle.FONT_CONFIG['label_size'])
                            ax.set_ylabel(disk_col.replace('_', ' ').title(), fontsize=UnifiedChartStyle.FONT_CONFIG['label_size'])
                            ax.set_title(f'{title}\nr={corr:.3f}, p={p_value:.3f}', fontsize=UnifiedChartStyle.FONT_CONFIG['subtitle_size'])
                            ax.grid(True, alpha=0.3)
                        else:
                            ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', transform=ax.transAxes, 
                                   fontsize=UnifiedChartStyle.FONT_CONFIG['text_size'])
                            ax.set_title(title, fontsize=UnifiedChartStyle.FONT_CONFIG['subtitle_size'])
                    
                    except Exception as e:
                        print(f"⚠️ Subplot generation failed: {e}")
                        ax.text(0.5, 0.5, f'Generation Failed\n{str(e)}', ha='center', va='center', transform=ax.transAxes,
                               fontsize=UnifiedChartStyle.FONT_CONFIG['text_size'])
                        ax.set_title(title, fontsize=UnifiedChartStyle.FONT_CONFIG['subtitle_size'])
                    
                    plot_idx += 1
                else:
                    # Hide extra subplots
                    axes[i, j].set_visible(False)
        
        UnifiedChartStyle.apply_layout(fig, 'auto')
        
        # Save chart
        output_file = os.path.join(self.output_dir, 'pearson_correlation_analysis.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        chart_files.append(output_file)
        print(f"✅ Pearson correlation chart saved: {output_file}")
        
        return chart_files
        
    def generate_regression_analysis_charts(self) -> List[str]:
        """Generate regression analysis charts"""
        if not self.load_data():
            return []
        
        print("📈 Generating regression analysis charts...")
        chart_files = []
        
        # Check Device configuration - use unified method
        accounts_configured = DeviceManager.is_accounts_configured(self.df)
        
        # Dynamically get device columns - directly iterate DataFrame columns
        data_r_cols = [col for col in self.df.columns if col.startswith('data_') and col.endswith('_r_s')]
        data_w_cols = [col for col in self.df.columns if col.startswith('data_') and col.endswith('_w_s')]
        accounts_r_cols = [col for col in self.df.columns if col.startswith('accounts_') and col.endswith('_r_s')] if accounts_configured else []
        accounts_w_cols = [col for col in self.df.columns if col.startswith('accounts_') and col.endswith('_w_s')] if accounts_configured else []
        
        # Build regression configuration
        regression_configs = []
        if data_r_cols:
            regression_configs.append(('cpu_usr', data_r_cols[0], 'User CPU vs DATA Read Requests'))
        if data_w_cols:
            regression_configs.append(('cpu_sys', data_w_cols[0], 'System CPU vs DATA Write Requests'))
        if accounts_configured and accounts_r_cols:
            regression_configs.append(('cpu_usr', accounts_r_cols[0], 'User CPU vs ACCOUNTS Read Requests'))
        if accounts_configured and accounts_w_cols:
            regression_configs.append(('cpu_sys', accounts_w_cols[0], 'System CPU vs ACCOUNTS Write Requests'))
        
        if not regression_configs:
            print("  ⚠️ No configured devices, skipping regression analysis chart generation")
            return []
        
        # Dynamically create subplot layout
        total_plots = len(regression_configs)
        if total_plots <= 2:
            rows, cols = 1, total_plots
        else:
            rows, cols = 2, 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 6*rows))
        
        # Ensure axes is always 2D array for unified processing
        if total_plots == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Using English title directly
        fig.suptitle('Linear Regression Analysis', fontsize=UnifiedChartStyle.FONT_CONFIG['title_size'], fontweight='bold')
        
        for idx, (x_col, y_col, title) in enumerate(regression_configs):
            row, col = divmod(idx, cols)
            ax: Axes = axes[row, col]
            
            if x_col in self.df.columns and y_col and y_col in self.df.columns:
                # Prepare data
                x = self.df[[x_col]].values
                y = self.df[y_col].values
                
                # Linear regression
                model = LinearRegression()
                model.fit(x, y)
                y_pred = model.predict(x)
                
                # Calculate R²
                r2 = model.score(x, y)
                
                # Draw scatter plot and regression line
                ax.scatter(self.df[x_col], self.df[y_col], alpha=0.6, s=20, color=UnifiedChartStyle.COLORS['data_primary'])
                ax.plot(self.df[x_col], y_pred, color=UnifiedChartStyle.COLORS['critical'], linestyle='-', linewidth=2)
                
                # Set title and labels
                ax.set_title(f'{title}\nR²={r2:.3f}, Coefficient={model.coef_[0]:.3f}', fontsize=UnifiedChartStyle.FONT_CONFIG['subtitle_size'])
                ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=UnifiedChartStyle.FONT_CONFIG['label_size'])
                ax.set_ylabel(y_col.replace('_', ' ').title(), fontsize=UnifiedChartStyle.FONT_CONFIG['label_size'])
                ax.grid(True, alpha=0.3)
                
                # Add regression equation
                equation = f'y = {model.coef_[0]:.3f}x + {model.intercept_:.3f}'
                ax.text(0.05, 0.95, equation, transform=ax.transAxes, fontsize=UnifiedChartStyle.FONT_CONFIG['text_size'],
                       verticalalignment='top')
            else:
                ax.text(0.5, 0.5, 'Data Not Available', ha='center', va='center', transform=ax.transAxes,
                       fontsize=UnifiedChartStyle.FONT_CONFIG['text_size'])
                ax.set_title(title, fontsize=UnifiedChartStyle.FONT_CONFIG["subtitle_size"])
        
        UnifiedChartStyle.apply_layout(fig, 'auto')
        chart_file = os.path.join(self.output_dir, 'linear_regression_analysis.png')
        plt.savefig(chart_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        chart_files.append(chart_file)
        print(f"  ✅ Linear regression chart: {os.path.basename(chart_file)}")
        
        return chart_files
    
    def generate_negative_correlation_charts(self) -> List[str]:
        """Generate negative correlation analysis charts"""
        if not self.load_data():
            return []
        
        print("📉 Generating negative correlation analysis charts...")
        chart_files = []
        
        # Check Device configuration - use unified method
        accounts_configured = DeviceManager.is_accounts_configured(self.df)
        
        # Dynamically get device columns
        data_aqu_cols = [col for col in self.df.columns if col.startswith('data_') and 'aqu_sz' in col]
        accounts_aqu_cols = [col for col in self.df.columns if col.startswith('accounts_') and 'aqu_sz' in col] if accounts_configured else []
        
        # Build negative correlation configuration
        negative_configs = []
        if data_aqu_cols:
            negative_configs.append(('cpu_idle', data_aqu_cols[0], 'CPU Idle vs DATA Queue Length'))
        if accounts_configured and accounts_aqu_cols:
            negative_configs.append(('cpu_idle', accounts_aqu_cols[0], 'CPU Idle vs ACCOUNTS Queue Length'))
        
        if not negative_configs:
            print("  ⚠️ No configured devices, skipping negative correlation analysis chart generation")
            return []
        
        # Dynamically create subplot layout
        total_plots = len(negative_configs)
        fig, axes = plt.subplots(1, total_plots, figsize=(8*total_plots, 6))
        
        # Ensure axes is always array
        if total_plots == 1:
            axes = [axes]
        
        # Using English title directly
        fig.suptitle('Negative Correlation Analysis', fontsize=UnifiedChartStyle.FONT_CONFIG["title_size"], fontweight='bold')
        
        for idx, (x_col, y_col, title) in enumerate(negative_configs):
            ax: Axes = axes[idx]
            
            if x_col in self.df.columns and y_col and y_col in self.df.columns:
                # Calculate correlation
                corr, p_value = stats.pearsonr(self.df[x_col], self.df[y_col])
                
                # Draw scatter plot
                ax.scatter(self.df[x_col], self.df[y_col], alpha=0.6, s=20, color=UnifiedChartStyle.COLORS['data_primary'])
                
                # Add regression line
                z = np.polyfit(self.df[x_col], self.df[y_col], 1)
                p = np.poly1d(z)
                ax.plot(self.df[x_col], p(self.df[x_col]), color=UnifiedChartStyle.COLORS['critical'], linestyle='--', alpha=0.8, linewidth=2)
                
                # Set title and labels
                correlation_type = "Negative" if corr < 0 else "Positive"
                ax.set_title(f'{title}\nr={corr:.3f} ({correlation_type})', fontsize=UnifiedChartStyle.FONT_CONFIG["subtitle_size"])
                ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=UnifiedChartStyle.FONT_CONFIG['label_size'])
                ax.set_ylabel(y_col.replace('_', ' ').title(), fontsize=UnifiedChartStyle.FONT_CONFIG['label_size'])
                ax.grid(True, alpha=0.3)
                
                # Highlight negative correlation
                if corr < 0:
                    ax.text(0.05, 0.95, '✓ Negative Correlation', transform=ax.transAxes,
                           fontsize=UnifiedChartStyle.FONT_CONFIG['text_size'], verticalalignment='top')
                else:
                    ax.text(0.05, 0.95, '⚠ Non-negative Correlation', transform=ax.transAxes,
                           fontsize=UnifiedChartStyle.FONT_CONFIG['text_size'], verticalalignment='top')
            else:
                ax.text(0.5, 0.5, 'Data Not Available', ha='center', va='center', transform=ax.transAxes,
                       fontsize=UnifiedChartStyle.FONT_CONFIG['text_size'])
                ax.set_title(title, fontsize=UnifiedChartStyle.FONT_CONFIG["subtitle_size"])
        
        UnifiedChartStyle.apply_layout(fig, 'auto')
        chart_file = os.path.join(self.output_dir, 'negative_correlation_analysis.png')
        plt.savefig(chart_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        chart_files.append(chart_file)
        print(f"  ✅ Negative correlation analysis chart: {os.path.basename(chart_file)}")
        
        return chart_files
    
    def generate_comprehensive_correlation_matrix(self) -> List[str]:
        """Generate comprehensive correlation matrix heatmap"""
        if not self.load_data():
            return []
        
        print("🔥 Generating comprehensive correlation matrix...")
        chart_files = []
        
        # Select key columns for correlation analysis
        key_columns = []
        
        # CPU related columns
        cpu_cols = ['cpu_usr', 'cpu_sys', 'cpu_iowait', 'cpu_idle', 'cpu_soft']
        for col in cpu_cols:
            if col in self.df.columns:
                key_columns.append(col)
        
        # Disk related columns
        disk_patterns = ['util', 'aqu_sz', 'avg_await', 'r_s', 'w_s', 'total_iops', 'throughput_mibs']
        for pattern in disk_patterns:
            matching_cols = [col for col in self.df.columns if pattern in col]
            # Filter ACCOUNTS device columns (if not configured)
            if not DeviceManager.is_accounts_configured(self.df):
                matching_cols = [col for col in matching_cols if not col.startswith('accounts_')]
            key_columns.extend(matching_cols[:2])  # Take at most 2 related columns
        
        # Remove duplicates and ensure columns exist
        key_columns = list(set(key_columns))
        key_columns = [col for col in key_columns if col in self.df.columns]
        
        if len(key_columns) < 4:
            print("  ⚠️ Insufficient available columns, skipping correlation matrix generation")
            return []
        
        # Calculate correlation matrix
        correlation_matrix = self.df[key_columns].corr()
        
        # Create heatmap
        plt.figure(figsize=(14, 12))
        
        # Use unified style configuration
        heatmap_config = UnifiedChartStyle.CHART_CONFIGS['heatmap']
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=heatmap_config['annot'], 
                   cmap=heatmap_config['cmap'], 
                   vmin=-1, 
                   vmax=1,
                   center=0,
                   square=True,
                   fmt=heatmap_config['fmt'],
                   cbar_kws={"shrink": .8})
        
        # Using English title directly
        plt.title('CPU-Disk Performance Metrics Correlation Matrix', 
                 fontsize=UnifiedChartStyle.FONT_CONFIG["title_size"], 
                 fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Use unified layout
        UnifiedChartStyle.apply_layout('auto')
        
        chart_file = os.path.join(self.output_dir, 'comprehensive_correlation_matrix.png')
        plt.savefig(chart_file, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        chart_files.append(chart_file)
        print(f"  ✅ Comprehensive correlation matrix: {os.path.basename(chart_file)}")
        
        return chart_files
    
    def generate_performance_trend_analysis(self) -> List[str]:
        """Generate performance trend analysis charts"""
        if not self.load_data():
            return []
        
        print("📈 Generating performance trend analysis...")
        chart_files = []
        
        # Ensure timestamp column exists
        if 'timestamp' not in self.df.columns:
            print("  ⚠️ Missing timestamp column, skipping trend analysis")
            return []
        
        # Convert timestamp
        if not pd.api.types.is_datetime64_any_dtype(self.df['timestamp']):
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        # Check ACCOUNTS device configuration
        accounts_configured = DeviceManager.is_accounts_configured(self.df)

        def numeric_series(col):
            return pd.to_numeric(self.df[col], errors='coerce').replace([np.inf, -np.inf], np.nan)

        def has_valid_data(cols):
            return any(numeric_series(col).notna().any() for col in cols)

        def no_data(ax, title, reason='No valid samples in this run'):
            ax.text(0.5, 0.5, reason, ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='#627d98')
            ax.set_title(f'{title} (No Data)', fontsize=UnifiedChartStyle.FONT_CONFIG['subtitle_size'])
            ax.grid(True, alpha=0.2)
        
        fig, axes = plt.subplots(3, 2, figsize=(18, 15))
        fig.suptitle('CPU-Disk Performance Trend Analysis', fontsize=UnifiedChartStyle.FONT_CONFIG["title_size"], fontweight='bold')
        
        # CPU Usage trends
        if 'cpu_iowait' in self.df.columns:
            axes[0, 0].plot(self.df['timestamp'], self.df['cpu_iowait'], color=UnifiedChartStyle.COLORS['data_primary'], linewidth=2, alpha=0.7)
            axes[0, 0].set_title('CPU I/O Wait Time Trends', fontsize=UnifiedChartStyle.FONT_CONFIG['subtitle_size'])
            axes[0, 0].set_ylabel('I/O Wait (%)', fontsize=UnifiedChartStyle.FONT_CONFIG['label_size'])
            axes[0, 0].grid(True, alpha=0.3)
            UnifiedChartStyle.format_time_axis(axes[0, 0], self.df['timestamp'])
        
        # Disk utilization trends - show DATA and ACCOUNTS
        data_util_cols = [col for col in self.df.columns if col.startswith('data_') and col.endswith('_util')]
        accounts_util_cols = [col for col in self.df.columns if col.startswith('accounts_') and col.endswith('_util')] if accounts_configured else []
        
        util_cols = data_util_cols + accounts_util_cols
        if has_valid_data(util_cols):
            if data_util_cols and numeric_series(data_util_cols[0]).notna().any():
                axes[0, 1].plot(self.df['timestamp'], numeric_series(data_util_cols[0]), color=UnifiedChartStyle.COLORS['data_primary'], linewidth=2, alpha=0.7, label='DATA')
            if accounts_util_cols and numeric_series(accounts_util_cols[0]).notna().any():
                axes[0, 1].plot(self.df['timestamp'], numeric_series(accounts_util_cols[0]), color=UnifiedChartStyle.COLORS['accounts_primary'], linewidth=2, alpha=0.7, label='ACCOUNTS')
            axes[0, 1].set_title('Disk Device Utilization Trends', fontsize=UnifiedChartStyle.FONT_CONFIG['subtitle_size'])
            axes[0, 1].set_ylabel('Utilization (%)', fontsize=UnifiedChartStyle.FONT_CONFIG['label_size'])
            axes[0, 1].legend(fontsize=UnifiedChartStyle.FONT_CONFIG['legend_size'])
            axes[0, 1].grid(True, alpha=0.3)
            UnifiedChartStyle.format_time_axis(axes[0, 1], self.df['timestamp'])
        else:
            no_data(axes[0, 1], 'Disk Device Utilization Trends')
        
        # IOPS trends - show DATA and ACCOUNTS
        data_iops_cols = [col for col in self.df.columns if col.startswith('data_') and 'total_iops' in col]
        accounts_iops_cols = [col for col in self.df.columns if col.startswith('accounts_') and 'total_iops' in col] if accounts_configured else []
        
        iops_cols = data_iops_cols + accounts_iops_cols
        if has_valid_data(iops_cols):
            if data_iops_cols and numeric_series(data_iops_cols[0]).notna().any():
                axes[1, 0].plot(self.df['timestamp'], numeric_series(data_iops_cols[0]), color=UnifiedChartStyle.COLORS['data_primary'], linewidth=2, alpha=0.7, label='DATA')
            if accounts_iops_cols and numeric_series(accounts_iops_cols[0]).notna().any():
                axes[1, 0].plot(self.df['timestamp'], numeric_series(accounts_iops_cols[0]), color=UnifiedChartStyle.COLORS['accounts_primary'], linewidth=2, alpha=0.7, label='ACCOUNTS')
            axes[1, 0].set_title('IOPS Trends', fontsize=UnifiedChartStyle.FONT_CONFIG['subtitle_size'])
            axes[1, 0].set_ylabel('IOPS', fontsize=UnifiedChartStyle.FONT_CONFIG['label_size'])
            axes[1, 0].legend(fontsize=UnifiedChartStyle.FONT_CONFIG['legend_size'])
            axes[1, 0].grid(True, alpha=0.3)
            UnifiedChartStyle.format_time_axis(axes[1, 0], self.df['timestamp'])
        else:
            no_data(axes[1, 0], 'IOPS Trends')
        
        # Throughput trends - show DATA and ACCOUNTS
        data_throughput_cols = [col for col in self.df.columns if col.startswith('data_') and 'throughput' in col and 'mibs' in col]
        accounts_throughput_cols = [col for col in self.df.columns if col.startswith('accounts_') and 'throughput' in col and 'mibs' in col] if accounts_configured else []
        
        throughput_cols = data_throughput_cols + accounts_throughput_cols
        if has_valid_data(throughput_cols):
            if data_throughput_cols and numeric_series(data_throughput_cols[0]).notna().any():
                axes[1, 1].plot(self.df['timestamp'], numeric_series(data_throughput_cols[0]), color=UnifiedChartStyle.COLORS['data_primary'], linewidth=2, alpha=0.7, label='DATA')
            if accounts_throughput_cols and numeric_series(accounts_throughput_cols[0]).notna().any():
                axes[1, 1].plot(self.df['timestamp'], numeric_series(accounts_throughput_cols[0]), color=UnifiedChartStyle.COLORS['accounts_primary'], linewidth=2, alpha=0.7, label='ACCOUNTS')
            axes[1, 1].set_title('Throughput Trends', fontsize=UnifiedChartStyle.FONT_CONFIG['subtitle_size'])
            axes[1, 1].set_ylabel('Throughput (MiB/s)', fontsize=UnifiedChartStyle.FONT_CONFIG['label_size'])
            axes[1, 1].legend(fontsize=UnifiedChartStyle.FONT_CONFIG['legend_size'])
            axes[1, 1].grid(True, alpha=0.3)
            UnifiedChartStyle.format_time_axis(axes[1, 1], self.df['timestamp'])
        else:
            no_data(axes[1, 1], 'Throughput Trends')
        
        # Latency trends - show DATA and ACCOUNTS
        data_await_cols = [col for col in self.df.columns if col.startswith('data_') and 'avg_await' in col]
        accounts_await_cols = [col for col in self.df.columns if col.startswith('accounts_') and 'avg_await' in col] if accounts_configured else []
        
        await_cols = data_await_cols + accounts_await_cols
        if has_valid_data(await_cols):
            if data_await_cols and numeric_series(data_await_cols[0]).notna().any():
                axes[2, 0].plot(self.df['timestamp'], numeric_series(data_await_cols[0]), color=UnifiedChartStyle.COLORS['data_primary'], linewidth=2, alpha=0.7, label='DATA')
            if accounts_await_cols and numeric_series(accounts_await_cols[0]).notna().any():
                axes[2, 0].plot(self.df['timestamp'], numeric_series(accounts_await_cols[0]), color=UnifiedChartStyle.COLORS['accounts_primary'], linewidth=2, alpha=0.7, label='ACCOUNTS')
            axes[2, 0].set_title('I/O Latency Trends', fontsize=UnifiedChartStyle.FONT_CONFIG['subtitle_size'])
            axes[2, 0].set_ylabel('Latency (ms)', fontsize=UnifiedChartStyle.FONT_CONFIG['label_size'])
            axes[2, 0].legend(fontsize=UnifiedChartStyle.FONT_CONFIG['legend_size'])
            axes[2, 0].grid(True, alpha=0.3)
            UnifiedChartStyle.format_time_axis(axes[2, 0], self.df['timestamp'])
        else:
            no_data(axes[2, 0], 'I/O Latency Trends')
        
        # Queue depth trends - show DATA and ACCOUNTS
        data_queue_cols = [col for col in self.df.columns if col.startswith('data_') and 'aqu_sz' in col]
        accounts_queue_cols = [col for col in self.df.columns if col.startswith('accounts_') and 'aqu_sz' in col] if accounts_configured else []
        
        queue_cols = data_queue_cols + accounts_queue_cols
        if has_valid_data(queue_cols):
            if data_queue_cols and numeric_series(data_queue_cols[0]).notna().any():
                axes[2, 1].plot(self.df['timestamp'], numeric_series(data_queue_cols[0]), color=UnifiedChartStyle.COLORS['data_primary'], linewidth=2, alpha=0.7, label='DATA')
            if accounts_queue_cols and numeric_series(accounts_queue_cols[0]).notna().any():
                axes[2, 1].plot(self.df['timestamp'], numeric_series(accounts_queue_cols[0]), color=UnifiedChartStyle.COLORS['accounts_primary'], linewidth=2, alpha=0.7, label='ACCOUNTS')
            axes[2, 1].set_title('I/O Queue Depth Trends', fontsize=UnifiedChartStyle.FONT_CONFIG['subtitle_size'])
            axes[2, 1].set_ylabel('Queue Depth', fontsize=UnifiedChartStyle.FONT_CONFIG['label_size'])
            axes[2, 1].legend(fontsize=UnifiedChartStyle.FONT_CONFIG['legend_size'])
            axes[2, 1].grid(True, alpha=0.3)
            UnifiedChartStyle.format_time_axis(axes[2, 1], self.df['timestamp'])
        else:
            no_data(axes[2, 1], 'I/O Queue Depth Trends')
        
        UnifiedChartStyle.apply_layout(fig, 'auto')
        chart_file = os.path.join(self.output_dir, 'performance_trend_analysis.png')
        plt.savefig(chart_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        chart_files.append(chart_file)
        print(f"  ✅ Performance trend analysis: {os.path.basename(chart_file)}")
        
        return chart_files
    
    def _get_correlation_strength(self, corr: float) -> str:
        """Get correlation strength description"""
        abs_corr = abs(corr)
        if abs_corr >= 0.8:
            return "Very Strong"
        elif abs_corr >= 0.6:
            return "Strong"
        elif abs_corr >= 0.4:
            return "Moderate"
        elif abs_corr >= 0.2:
            return "Weak"
        else:
            return "Very Weak"
    
    def generate_ena_network_analysis_charts(self) -> List[str]:
        """Generate ENA network limitation analysis charts - use ENAFieldAccessor unified interface"""
        if not self.load_data():
            return []
        
        print("🌐 Generating ENA network limitation analysis charts...")
        chart_files = []
        
        # Use ENAFieldAccessor to check ENA data - configuration driven
        ena_columns = ENAFieldAccessor.get_available_ena_fields(self.df)
        if not ena_columns:
            print("  ⚠️ No ENA network data available, skipping ENA analysis charts")
            print("  💡 Tip: Ensure ENA_MONITOR_ENABLED=true and the AWS provider exports ENA_ALLOWANCE_FIELDS_STR")
            return []
        
        # Check timestamp column
        if 'timestamp' not in self.df.columns:
            print("  ⚠️ Missing timestamp column, skipping ENA trend analysis")
            return []
        
        # Convert timestamp
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        # Generate ENA limitation trend charts
        trend_chart = self._generate_ena_limitation_trends_chart()
        if trend_chart:
            chart_files.append(trend_chart)
        
        # Generate ENA connection capacity chart
        capacity_chart = self._generate_ena_connection_capacity_chart()
        if capacity_chart:
            chart_files.append(capacity_chart)
        
        # Generate ENA comprehensive status chart
        comprehensive_chart = self._generate_ena_comprehensive_status_chart()
        if comprehensive_chart:
            chart_files.append(comprehensive_chart)
        
        return chart_files

    def _generate_ena_limitation_trends_chart(self):
        """Generate ENA limitation trend charts - use real field names"""
        try:
            # Use ENAFieldAccessor to get available ENA fields
            available_ena_fields = ENAFieldAccessor.get_available_ena_fields(self.df)
            
            # Dynamically build limitation field configuration - based on actually available fields
            limitation_fields = {}
            field_colors = UnifiedChartStyle.COLOR_PALETTE[:6]
            color_index = 0
            
            for field in available_ena_fields:
                if 'exceeded' in field:  # Only process exceeded type fields
                    field_analysis = ENAFieldAccessor.analyze_ena_field(self.df, field)
                    if field_analysis:
                        limitation_fields[field] = {
                            'label': field_analysis['display_name'],
                            'color': field_colors[color_index % len(field_colors)]
                        }
                        color_index += 1
            
            # Check if there is any limitation data
            has_limitation_data = False
            for field in limitation_fields.keys():
                if field in self.df.columns:
                    delta_series = self._calculate_ena_delta_series(self.df, field)
                    if delta_series.iloc[-1] > 0:
                        has_limitation_data = True
                        break
            
            # Create chart
            fig, ax = plt.subplots(1, 1, figsize=(16, 8))
            
            if not has_limitation_data:
                # When no limitation events, display prompt message
                ax.text(0.5, 0.5, 'No ENA Network Limitations Detected\nAll limitation counters remain at 0 during test period', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=UnifiedChartStyle.FONT_CONFIG["title_size"],
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
                ax.set_title('ENA Network Limitation Trend Analysis', fontsize=UnifiedChartStyle.FONT_CONFIG["title_size"], fontweight='bold')
                ax.set_xlabel('Time', fontsize=UnifiedChartStyle.FONT_CONFIG["subtitle_size"])
                ax.set_ylabel('Limitation Triggers (Delta)', fontsize=UnifiedChartStyle.FONT_CONFIG["subtitle_size"])
                ax.grid(True, alpha=0.3)
            else:
                # Plot trend lines for each ENA limitation metric
                lines_plotted = 0
                for field, config in limitation_fields.items():
                    if field in self.df.columns:
                        delta_series = self._calculate_ena_delta_series(self.df, field)
                        # Only display fields with delta data
                        if delta_series.iloc[-1] > 0:
                            ax.plot(self.df['timestamp'], delta_series, 
                                   label=config['label'], 
                                   color=config['color'],
                                   linewidth=2,
                                   marker='o',
                                   markersize=3,
                                   alpha=0.8)
                            lines_plotted += 1
                
                if lines_plotted == 0:
                    plt.close()
                    return None
                
                # Chart styling with English labels
                ax.set_title('ENA Network Limitation Trend Analysis', fontsize=UnifiedChartStyle.FONT_CONFIG["title_size"], fontweight='bold')
                ax.set_xlabel('Time', fontsize=UnifiedChartStyle.FONT_CONFIG["subtitle_size"])
                ax.set_ylabel('Limitation Triggers (Delta)', fontsize=UnifiedChartStyle.FONT_CONFIG["subtitle_size"])
                ax.legend(loc='upper left')
                ax.grid(True, alpha=0.3)
                
                # Add statistics info box (upper right corner)
                stats_lines = ["Statistics:"]
                for field, config in limitation_fields.items():
                    if field in self.df.columns:
                        delta_series = self._calculate_ena_delta_series(self.df, field)
                        if delta_series.iloc[-1] > 0:
                            total = int(delta_series.iloc[-1])
                            events = (delta_series > 0).sum()
                            stats_lines.append(f"{config['label']}:")
                            stats_lines.append(f"  Total: {total:,}")
                            stats_lines.append(f"  Events: {events}")
                
                # Add test duration
                time_span = (self.df['timestamp'].iloc[-1] - self.df['timestamp'].iloc[0]).total_seconds()
                stats_lines.append(f"\nDuration: {time_span/60:.1f} min")
                
                stats_text = "\n".join(stats_lines)
                ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                       verticalalignment='top', horizontalalignment='right',
                       fontsize=UnifiedChartStyle.FONT_CONFIG['text_size'],
                       fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
            
            # Time axis formatting
            plt.xticks(rotation=45)
            UnifiedChartStyle.apply_layout('auto')
            
            # Save chart
            chart_file = os.path.join(self.output_dir, 'ena_limitation_trends.png')
            plt.savefig(chart_file, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            print(f"  ✅ ENA limitation trend chart generated: {os.path.basename(chart_file)}")
            return chart_file
            
        except Exception as e:
            self._log_error("ENA limitation trend chart generation", e)
            return None

    def _generate_ena_connection_capacity_chart(self):
        """Generate ENA connection capacity charts - use real field names"""
        try:
            # Find conntrack_allowance_available field
            available_field = None
            for field in ENAFieldAccessor.get_available_ena_fields(self.df):
                if 'available' in field and 'conntrack' in field:
                    available_field = field
                    break
            
            if not available_field:
                return None
            
            # Check if there is connection capacity data
            if self.df[available_field].max() == 0:
                print("  ℹ️ No ENA connection capacity data, skipping connection capacity chart")
                return None
            
            # Create chart
            fig, ax = plt.subplots(1, 1, figsize=(16, 6))
            
            # Draw connection capacity trend
            ax.plot(self.df['timestamp'], self.df[available_field], 
                   color=UnifiedChartStyle.COLORS["success"], linewidth=2, marker='o', markersize=2, alpha=0.8,
                   label='Available Connections')
            
            # Chart styling with English labels
            ax.set_title('ENA Connection Capacity Monitoring', fontsize=UnifiedChartStyle.FONT_CONFIG["title_size"], fontweight='bold')
            ax.set_xlabel('Time', fontsize=UnifiedChartStyle.FONT_CONFIG["subtitle_size"])
            ax.set_ylabel('Available Connections', fontsize=UnifiedChartStyle.FONT_CONFIG["subtitle_size"])
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add statistics info box (upper right corner)
            capacity_data = self.df[available_field]
            current = int(capacity_data.iloc[-1])
            average = int(capacity_data.mean())
            minimum = int(capacity_data.min())
            maximum = int(capacity_data.max())
            
            stats_lines = [
                "Statistics:",
                "",
                f"Current:  {current:,}",
                f"Average:  {average:,}",
                f"Minimum:  {minimum:,}",
                f"Maximum:  {maximum:,}"
            ]
            
            stats_text = "\n".join(stats_lines)
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   fontsize=UnifiedChartStyle.FONT_CONFIG['text_size'],
                   fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
            
            # Format Y-axis values
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
            
            # Time axis formatting
            plt.xticks(rotation=45)
            UnifiedChartStyle.apply_layout('auto')
            
            # Save chart
            chart_file = os.path.join(self.output_dir, 'ena_connection_capacity.png')
            plt.savefig(chart_file, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            print(f"  ✅ ENA connection capacity chart generated: {os.path.basename(chart_file)}")
            return chart_file
            
        except Exception as e:
            self._log_error("ENA connection capacity chart generation", e)
            return None

    def _generate_ena_comprehensive_status_chart(self):
        """Generate ENA comprehensive status charts - use ENAFieldAccessor"""
        try:
            # Use ENAFieldAccessor to get available ENA fields
            available_fields = ENAFieldAccessor.get_available_ena_fields(self.df)
            if len(available_fields) < 3:
                print("  ℹ️ Insufficient ENA fields for comprehensive analysis")
                return None
            
            # Create 2x2 subplot layout
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            # Using English title directly
            fig.suptitle('ENA Network Comprehensive Analysis', fontsize=UnifiedChartStyle.FONT_CONFIG["title_size"], fontweight='bold')
            
            # 1. Limitation type distribution (upper left)
            ax1 = axes[0, 0]
            limitation_counts = {}
            label_mapping = {
                'Inbound Bandwidth Allowance Exceeded': 'Inbound Bandwidth\nAllowance Exceeded',
                'Outbound Bandwidth Allowance Exceeded': 'Outbound Bandwidth\nAllowance Exceeded',
                'PPS Allowance Exceeded': 'PPS Allowance\nExceeded',
                'Connection Tracking Allowance Exceeded': 'Connection Tracking\nAllowance Exceeded',
                'Link Local Allowance Exceeded': 'Link Local\nAllowance Exceeded'
            }
            
            # Use ENAFieldAccessor to dynamically get field labels
            for field in available_fields:
                if 'exceeded' in field:  # Only process exceeded type fields
                    field_analysis = ENAFieldAccessor.analyze_ena_field(self.df, field)
                    if field_analysis and field in self.df.columns:
                        delta_series = self._calculate_ena_delta_series(self.df, field)
                        delta_value = int(delta_series.iloc[-1])
                        if delta_value > 0:
                            full_label = field_analysis['display_name']
                            wrapped_label = label_mapping.get(full_label, full_label)
                            limitation_counts[wrapped_label] = delta_value
            
            if limitation_counts:
                pie_config = UnifiedChartStyle.CHART_CONFIGS['pie']
                ax1.pie(limitation_counts.values(), labels=limitation_counts.keys(), 
                       autopct=pie_config['autopct'], startangle=pie_config['startangle'],
                       pctdistance=pie_config['pctdistance'], 
                       labeldistance=1.05)
                ax1.set_title('Limitation Type Distribution')
            else:
                ax1.text(0.5, 0.5, 'No Network Limitations Detected', ha='center', va='center', 
                        transform=ax1.transAxes, fontsize=UnifiedChartStyle.FONT_CONFIG["subtitle_size"])
                ax1.set_title('Limitation Type Distribution')
            
            # 2. Connection capacity status (upper right)
            ax2 = axes[0, 1]
            # Find available type field
            available_field = None
            for field in available_fields:
                if 'available' in field:
                    available_field = field
                    break
            
            if available_field and available_field in self.df.columns:
                capacity_data = self.df[available_field]
                ax2.hist(capacity_data, bins=20, alpha=0.7, color=UnifiedChartStyle.COLORS["success"], edgecolor='black')
                ax2.axvline(capacity_data.mean(), color=UnifiedChartStyle.COLORS["critical"], linestyle='--', 
                           label=f'Average: {capacity_data.mean():,.0f}')
                ax2.set_title('Connection Capacity Distribution')
                ax2.set_xlabel('Available Connections')
                ax2.set_ylabel('Frequency')
                ax2.legend()
            else:
                ax2.text(0.5, 0.5, 'No Connection Capacity Data', ha='center', va='center', 
                        transform=ax2.transAxes, fontsize=UnifiedChartStyle.FONT_CONFIG["subtitle_size"])
                ax2.set_title('Connection Capacity Distribution')
            
            # 3. Limitation severity timeline (lower left)
            ax3 = axes[1, 0]
            # Calculate total limitation severity at each time point - use exceeded type fields
            severity_fields = [field for field in available_fields if 'exceeded' in field]
            
            severity_score = pd.Series(0, index=self.df.index)
            for field in severity_fields:
                if field in self.df.columns:
                    delta_series = self._calculate_ena_delta_series(self.df, field)
                    severity_score += (delta_series > 0).astype(int)
            
            if severity_score.max() > 0:
                ax3.plot(self.df['timestamp'], severity_score, color=UnifiedChartStyle.COLORS["critical"], linewidth=2)
                ax3.fill_between(self.df['timestamp'], severity_score, alpha=0.3, color=UnifiedChartStyle.COLORS["critical"])
                ax3.set_title('Network Limitation Severity')
                ax3.set_xlabel('Time')
                ax3.set_ylabel('Concurrent Limitation Types')
                plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
            else:
                ax3.text(0.5, 0.5, 'No Network Limitation Records', ha='center', va='center', 
                        transform=ax3.transAxes, fontsize=UnifiedChartStyle.FONT_CONFIG["subtitle_size"])
                ax3.set_title('Network Limitation Severity')
            
            # 4. ENA status summary (lower right)
            ax4 = axes[1, 1]
            # Create status summary table - use ENAFieldAccessor
            summary_data = []
            
            for field in available_fields:
                if field in self.df.columns:
                    field_analysis = ENAFieldAccessor.analyze_ena_field(self.df, field)
                    if field_analysis:
                        if field_analysis['type'] == 'gauge':  # available type field
                            field_mean = self.df[field].mean()
                            field_min = self.df[field].min()
                            unit = field_analysis.get('unit', 'connections')
                            summary_data.append([
                                field_analysis['display_name'], 
                                f'{field_mean:,.0f} {unit}', 
                                f'{field_min:,.0f} {unit}'
                            ])
                        else:  # counter type field (exceeded)
                            delta_series = self._calculate_ena_delta_series(self.df, field)
                            delta_value = int(delta_series.iloc[-1])
                            event_count = (delta_series > 0).sum()
                            unit = field_analysis.get('unit', 'packets')
                            summary_data.append([
                                field_analysis['display_name'], 
                                f'{delta_value} {unit}', 
                                f'{event_count} events'
                            ])
            
            if summary_data:
                table = ax4.table(cellText=summary_data,
                                colLabels=['Metric', 'Max/Avg Value', 'Event Count/Min Value'],
                                cellLoc='left',
                                loc='center',
                                colWidths=[0.5, 0.25, 0.25])
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 2.2)
                ax4.axis('off')
                ax4.set_title('ENA Status Summary')
            else:
                ax4.text(0.5, 0.5, 'No ENA Data', ha='center', va='center', 
                        transform=ax4.transAxes, fontsize=UnifiedChartStyle.FONT_CONFIG["subtitle_size"])
                ax4.set_title('ENA Status Summary')
            
            # Use unified style to apply layout
            UnifiedChartStyle.apply_layout('auto')
            
            # Save chart
            chart_file = os.path.join(self.output_dir, 'ena_comprehensive_status.png')
            plt.savefig(chart_file, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            print(f"  ✅ ENA comprehensive status chart generated: {os.path.basename(chart_file)}")
            return chart_file
            
        except Exception as e:
            self._log_error("ENA comprehensive status chart generation", e)
            return None

    def generate_all_charts(self) -> List[str]:
        """Generate all charts"""
        print("🎨 Starting complete CPU-Disk correlation analysis chart generation...")
        
        # 🎨 Refactor: apply unified style configuration
        try:
            unified_style = UnifiedChartStyle()
            unified_style.setup_matplotlib()
            print("✅ Unified style applied to advanced charts")
        except ImportError:
            print("⚠️ Unified style configuration unavailable, using default style")
        
        all_charts = []
        
        # 1. Pearson correlation charts (6-8 types, dynamically adjusted based on configured Device)
        all_charts.extend(self.generate_pearson_correlation_charts())
        
        # 2. Regression analysis charts (4 types, dynamically adjusted based on configured Device)
        all_charts.extend(self.generate_regression_analysis_charts())
        
        # 3. Negative correlation analysis charts (2 types, dynamically adjusted based on configured Device)
        all_charts.extend(self.generate_negative_correlation_charts())
        
        # 4. ENA network limitation analysis charts
        all_charts.extend(self.generate_ena_network_analysis_charts())
        
        # 5. Comprehensive correlation matrix
        all_charts.extend(self.generate_comprehensive_correlation_matrix())
        
        # 6. Performance trend analysis
        all_charts.extend(self.generate_performance_trend_analysis())
        
        # New: correlation heatmap
        all_charts.extend(self.generate_correlation_heatmap())
        
        print(f"\n🎉 Chart generation completed! Generated {len(all_charts)} chart files:")
        for chart in all_charts:
            print(f"  📊 {os.path.basename(chart)}")
        
        return all_charts

    def generate_correlation_heatmap(self) -> List[str]:
        """
        Generate performance metrics correlation heatmap
        Generate comprehensive correlation analysis based on existing 71 CSV field mappings
        """
        if not self.load_data():
            return []
        
        print("\n📊 Generating correlation heatmap...")
        
        try:
            # Select numeric fields for correlation analysis
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Exclude timestamp and some fields not suitable for correlation analysis
            exclude_cols = ['timestamp', 'current_qps', 'test_duration']
            numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            if len(numeric_cols) < 2:
                print("⚠️  Insufficient numeric fields available for correlation analysis")
                return []
            
            # Calculate correlation matrix
            correlation_data = self.df[numeric_cols].dropna()
            correlation_matrix = correlation_data.corr(method='pearson')
            
            # Create heatmap
            plt.figure(figsize=(16, 14))
            
            # Create mask, only show lower triangle
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            
            # Generate heatmap
            sns.heatmap(
                correlation_matrix, 
                mask=mask,
                annot=True, 
                fmt='.2f',
                cmap='RdYlBu_r', 
                center=0,
                square=True,
                cbar_kws={"shrink": .8},
                annot_kws={'size': 8}
            )
            
            # Using English labels directly
            plt.title('Performance Metrics Correlation Heatmap', 
                     fontsize=UnifiedChartStyle.FONT_CONFIG["title_size"], fontweight='bold', pad=20)
            plt.xlabel('Performance Metrics', fontsize=UnifiedChartStyle.FONT_CONFIG["subtitle_size"])
            plt.ylabel('Performance Metrics', fontsize=UnifiedChartStyle.FONT_CONFIG["subtitle_size"])
            
            # Rotate labels to improve readability
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            plt.tight_layout()
            
            # Save chart
            chart_file = os.path.join(self.output_dir, 'performance_correlation_heatmap.png')
            plt.savefig(chart_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()
            
            print(f"  ✅ Correlation heatmap: {os.path.basename(chart_file)}")
            
            # Generate strong correlation analysis report
            self._generate_correlation_insights(correlation_matrix, chart_file)
            
            return [chart_file]
            
        except Exception as e:
            self._log_error("Correlation heatmap generation", e)
            return []

    def _generate_correlation_insights(self, correlation_matrix: pd.DataFrame, chart_file: str):
        """
        Generate correlation insights analysis
        Identify strong positive and negative correlation metric pairs
        """
        try:
            # Find strong correlations (|r| > 0.7)
            strong_correlations = []
            
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:
                        strong_correlations.append({
                            'metric1': correlation_matrix.columns[i],
                            'metric2': correlation_matrix.columns[j],
                            'correlation': corr_value,
                            'strength': 'Strong Positive' if corr_value > 0 else 'Strong Negative'
                        })
            
            # Sort by correlation strength
            strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            # Generate insights report
            insights_file = chart_file.replace('.png', '_insights.txt')
            with open(insights_file, 'w', encoding='utf-8') as f:
                f.write("Performance Metrics Correlation Analysis Insights Report\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Analysis Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Number of Metrics Analyzed: {len(correlation_matrix.columns)}\n")
                f.write(f"Strong Correlation Pairs: {len(strong_correlations)}\n\n")
                
                if strong_correlations:
                    f.write("🔍 Strong Correlation Pairs (|r| > 0.7):\n")
                    f.write("-" * 40 + "\n")
                    for i, corr in enumerate(strong_correlations[:10], 1):  # Show top 10 only
                        f.write(f"{i:2d}. {corr['metric1']} ↔ {corr['metric2']}\n")
                        f.write(f"    Correlation Coefficient: {corr['correlation']:.3f} ({corr['strength']})\n\n")
                else:
                    f.write("No strong correlation pairs found (|r| > 0.7)\n")
            
            print(f"  📋 Correlation insights: {os.path.basename(insights_file)}")
            
        except Exception as e:
            print(f"⚠️  Correlation insights generation failed: {e}")


# Usage example
if __name__ == "__main__":
    print("🎨 Advanced chart generator usage example:")
    print("generator = AdvancedChartGenerator('performance_data.csv')")
    print("charts = generator.generate_all_charts()")
    print("# Generate statistical analysis visualization charts including correlation heatmaps")
