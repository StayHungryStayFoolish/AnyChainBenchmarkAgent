#!/usr/bin/env python3
"""
Report Generator - Enhanced Version with Bottleneck Mode Support
Integrates monitoring overhead analysis, configuration status check, and specific analysis features
Supports bottleneck detection results display and specialized analysis reports
"""

import pandas as pd
import json
import subprocess
import os
import sys
import argparse
import glob
import traceback
import numpy as np
import html
import re
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import pearsonr
from typing import Dict, Union

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from visualization.chart_style_config import UnifiedChartStyle
from visualization.device_manager import DeviceManager
from visualization.performance_visualizer import format_time_axis
from utils.ena_field_accessor import ENAFieldAccessor
from utils.csv_schema_registry import CSVSchemaRegistry

# Report text is stored outside Python code so report layout and copy can evolve independently.
def _load_report_translations() -> Dict[str, Dict[str, str]]:
    i18n_dir = os.path.join(script_dir, 'i18n')
    translations: Dict[str, Dict[str, str]] = {}
    for language in ('en', 'zh'):
        path = os.path.join(i18n_dir, f'report.{language}.json')
        with open(path, 'r', encoding='utf-8') as f:
            translations[language] = json.load(f)

    en_keys = set(translations.get('en', {}))
    for language, entries in translations.items():
        language_keys = set(entries)
        missing = en_keys - language_keys
        extra = language_keys - en_keys
        if missing or extra:
            raise ValueError(
                f"Report i18n key mismatch for {language}: "
                f"missing={', '.join(sorted(missing))}; extra={', '.join(sorted(extra))}"
            )
    return translations


TRANSLATIONS = _load_report_translations()

def safe_get_env_int(env_name, default_value=0):
    """Safely get environment variable and convert to integer"""
    try:
        value = os.getenv(env_name)
        if value and value != 'N/A' and value.strip():
            return int(value)
        return default_value
    except (ValueError, TypeError):
        print(f"⚠️ Environment variable {env_name} format error")
        return default_value

def get_visualization_thresholds():
    """Get visualization threshold configuration - using safe environment variable access"""
    return {
        'warning': safe_get_env_int('BOTTLENECK_CPU_THRESHOLD', 85),
        'critical': safe_get_env_int('SUCCESS_RATE_THRESHOLD', 95),
        'io_warning': safe_get_env_int('BOTTLENECK_NETWORK_THRESHOLD', 80),
        'memory': safe_get_env_int('BOTTLENECK_MEMORY_THRESHOLD', 90)
    }

class ReportGenerator:
    def __init__(self, performance_csv, config_file='config_loader.sh', overhead_csv=None, bottleneck_info=None, language='en'):
        self.performance_csv = performance_csv
        self.config_file = config_file
        self.overhead_csv = overhead_csv
        self.bottleneck_info = bottleneck_info
        self.language = language
        self.t = TRANSLATIONS.get(language, TRANSLATIONS['en'])
        self.output_dir = os.getenv('REPORTS_DIR', os.path.dirname(performance_csv))
        self.logs_dir = self._runtime_logs_dir()
        self.disk_log_path = os.path.join(self.logs_dir, 'disk_bottleneck_detector.log')
        self.config = self._load_config()
        self.overhead_data = self._load_overhead_data()
        self.bottleneck_data = self._load_bottleneck_data()

        # Execute data integrity validation
        self.validation_results = self.validate_data_integrity()

    def _runtime_logs_dir(self):
        return os.getenv('LOGS_DIR') or os.path.dirname(os.getenv('PERFORMANCE_LATEST_CSV', self.performance_csv))

    def _runtime_file_candidates(self, env_var, *fallback_paths):
        candidates = []
        env_path = os.getenv(env_var)
        if env_path:
            candidates.append(env_path)
        for path in fallback_paths:
            if path and path not in candidates:
                candidates.append(path)
        return candidates

    def _load_config(self):
        config = {}
        # Read configuration from environment variables
        config_keys = [
            'BLOCKCHAIN_NODE',
            'CLOUD_PROVIDER', 'REPORT_CLOUD_PROVIDER', 'CLOUD_REGION', 'CLOUD_ZONE', 'MACHINE_TYPE',
            'DEPLOYMENT_PLATFORM', 'DEPLOYMENT_MODE',
            'NETWORK_INTERFACE',
            'DATA_VOL_TYPE', 'DATA_VOL_SIZE', 'ACCOUNTS_VOL_TYPE', 'ACCOUNTS_VOL_SIZE',
            'NETWORK_MAX_BANDWIDTH_GBPS', 'ENA_MONITOR_ENABLED',
            'LEDGER_DEVICE', 'ACCOUNTS_DEVICE',
            'DATA_VOL_MAX_IOPS', 'DATA_VOL_MAX_THROUGHPUT',
            'ACCOUNTS_VOL_MAX_IOPS', 'ACCOUNTS_VOL_MAX_THROUGHPUT'
        ]
        for key in config_keys:
            value = os.getenv(key)
            if value:
                config[key] = value
        return config

    def _config_value(self, key, default='N/A'):
        value = self.config.get(key) if hasattr(self, 'config') else None
        if value is None or str(value).strip() == '':
            return default
        return str(value)

    def _first_csv_value(self, df, column, default='N/A'):
        if df is None or column not in df.columns or df[column].dropna().empty:
            return default
        value = df[column].dropna().iloc[0]
        if value is None or str(value).strip() == '':
            return default
        return str(value)

    def _env_item(self, label, value):
        return f"""
        <div class="env-item">
            <span class="env-label">{html.escape(str(label))}</span>
            <span class="env-value">{html.escape(str(value))}</span>
        </div>
        """

    def _generate_environment_summary_section(self, df):
        """Generate a concise first-screen environment summary."""
        cloud_provider = self._config_value(
            'REPORT_CLOUD_PROVIDER',
            self._config_value('CLOUD_PROVIDER', self._first_csv_value(df, 'cloud_provider'))
        )
        cloud_region = self._config_value('CLOUD_REGION')
        cloud_zone = self._config_value('CLOUD_ZONE')
        machine_type = self._config_value('MACHINE_TYPE')
        deployment_mode = self._config_value('DEPLOYMENT_MODE', self._config_value('DEPLOYMENT_PLATFORM'))
        chain = self._config_value('BLOCKCHAIN_NODE', self._first_csv_value(df, 'blockchain_node', 'General'))
        network_interface = self._config_value('NETWORK_INTERFACE')
        network_bandwidth = self._config_value('NETWORK_MAX_BANDWIDTH_GBPS')
        if network_bandwidth != 'N/A':
            network_bandwidth = f"{network_bandwidth} Gbps"

        data_storage = [
            self._env_item(self.t['data_device'], self._config_value('LEDGER_DEVICE')),
            self._env_item(self.t['data_volume_type'], self._config_value('DATA_VOL_TYPE')),
            self._env_item(self.t['volume_size_gib'], self._config_value('DATA_VOL_SIZE')),
            self._env_item(self.t['max_iops'], self._config_value('DATA_VOL_MAX_IOPS')),
            self._env_item(self.t['max_throughput_mibs'], self._config_value('DATA_VOL_MAX_THROUGHPUT')),
        ]
        accounts_storage = [
            self._env_item(self.t['accounts_device'], self._config_value('ACCOUNTS_DEVICE')),
            self._env_item(self.t['accounts_volume_type'], self._config_value('ACCOUNTS_VOL_TYPE')),
            self._env_item(self.t['volume_size_gib'], self._config_value('ACCOUNTS_VOL_SIZE')),
            self._env_item(self.t['max_iops'], self._config_value('ACCOUNTS_VOL_MAX_IOPS')),
            self._env_item(self.t['max_throughput_mibs'], self._config_value('ACCOUNTS_VOL_MAX_THROUGHPUT')),
        ]

        return f"""
        <div class="section environment-section">
            <h2>{self.t['run_environment']}</h2>
            <div class="environment-grid">
                <div class="environment-card">
                    <h3>{self.t['cloud_and_machine']}</h3>
                    {self._env_item(self.t['cloud_provider'], cloud_provider)}
                    {self._env_item(self.t['cloud_region'], cloud_region)}
                    {self._env_item(self.t['cloud_zone'], cloud_zone)}
                    {self._env_item(self.t['machine_type'], machine_type)}
                    {self._env_item(self.t['deployment_mode'], deployment_mode)}
                    {self._env_item(self.t['blockchain_node_type'], chain)}
                </div>
                <div class="environment-card">
                    <h3>{self.t['storage_profile']} - DATA</h3>
                    {''.join(data_storage)}
                </div>
                <div class="environment-card">
                    <h3>{self.t['storage_profile']} - ACCOUNTS</h3>
                    {''.join(accounts_storage)}
                </div>
                <div class="environment-card">
                    <h3>{self.t['network_profile']}</h3>
                    {self._env_item(self.t['network_interface'], network_interface)}
                    {self._env_item(self.t['network_bandwidth'], network_bandwidth)}
                    {self._env_item('ENA', self._config_value('ENA_MONITOR_ENABLED'))}
                </div>
            </div>
        </div>
        """

    def _quality_item(self, label, value, tone='neutral'):
        return f"""
        <div class="quality-item quality-{html.escape(str(tone))}">
            <span class="quality-label">{html.escape(str(label))}</span>
            <span class="quality-value">{html.escape(str(value))}</span>
        </div>
        """

    def _count_valid_disk_samples(self, df):
        if df is None or df.empty:
            return 0
        disk_cols = [
            col for col in df.columns
            if (col.startswith('data_') or col.startswith('accounts_'))
            and (
                col.endswith('_total_iops')
                or col.endswith('_total_throughput_mibs')
                or col.endswith('_util')
                or col.endswith('_avg_await')
            )
        ]
        if not disk_cols:
            return 0
        numeric = df[disk_cols].apply(pd.to_numeric, errors='coerce')
        return int(numeric.notna().any(axis=1).sum())

    def _read_proxy_record_count(self):
        proxy_csv = next(
            (path for path in self._runtime_file_candidates(
                'PROXY_METHOD_CSV',
                os.path.join(self.logs_dir, 'proxy_method.csv'),
                os.path.join(self.output_dir, 'proxy_method.csv'),
            ) if os.path.exists(path)),
            None,
        )
        if not proxy_csv:
            return 0, 0, 0
        try:
            from analysis.per_method_attribution import (
                filter_proxy_records_by_methods,
                read_proxy_csv,
            )
            records = list(read_proxy_csv(proxy_csv))
            allowed_methods = self._load_configured_workload_methods()
            workload_records = filter_proxy_records_by_methods(records, allowed_methods)
            excluded = max(len(records) - len(workload_records), 0) if allowed_methods else 0
            return len(records), len(workload_records), excluded
        except Exception:
            return 0, 0, 0

    def _generate_data_quality_section(self, df):
        """Summarize whether missing charts are caused by missing source data."""
        monitor_samples = len(df) if df is not None else 0
        active_qps_samples = 0
        if df is not None and 'current_qps' in df.columns:
            active_qps_samples = int((pd.to_numeric(df['current_qps'], errors='coerce').fillna(0) > 0).sum())
        valid_disk_samples = self._count_valid_disk_samples(df)
        proxy_records, workload_records, excluded_probe_records = self._read_proxy_record_count()

        disk_tone = 'ok' if valid_disk_samples > 0 else 'warn'
        proxy_tone = 'ok' if workload_records > 0 else 'warn'
        qps_tone = 'ok' if active_qps_samples > 0 else 'warn'

        return f"""
        <div class="section data-quality-section">
            <h2>{self.t['data_quality_summary']}</h2>
            <div class="quality-grid">
                {self._quality_item(self.t['monitor_samples'], monitor_samples, 'ok' if monitor_samples > 0 else 'warn')}
                {self._quality_item(self.t['active_qps_samples'], active_qps_samples, qps_tone)}
                {self._quality_item(self.t['valid_disk_samples'], valid_disk_samples, disk_tone)}
                {self._quality_item(self.t['proxy_records'], proxy_records, 'ok' if proxy_records > 0 else 'warn')}
                {self._quality_item(self.t['workload_rpc_records'], workload_records, proxy_tone)}
                {self._quality_item(self.t['excluded_probe_records'], excluded_probe_records, 'neutral')}
            </div>
            <div class="quality-notes">
                <p>{self.t['disk_data_note']}</p>
                <p>{self.t['proxy_filter_note']}</p>
            </div>
        </div>
        """

    def _add_section_id(self, section_html, section_id):
        """Attach a stable id to the first top-level section div."""
        if not section_html or not str(section_html).strip():
            return section_html
        first_div = re.search(r'<div\b[^>]*\bclass="[^"]*\bsection\b[^"]*"[^>]*>', section_html)
        if not first_div:
            return section_html

        opening = first_div.group(0)
        if re.search(r'\bid=', opening):
            updated = re.sub(r'\bid="[^"]*"', f'id="{section_id}"', opening, count=1)
        else:
            updated = opening.replace('<div', f'<div id="{section_id}"', 1)

        return section_html[:first_div.start()] + updated + section_html[first_div.end():]

    def _generate_report_nav(self, items):
        links = []
        for section_id, label, is_available in items:
            if not is_available:
                continue
            links.append(
                f'<a href="#{html.escape(section_id)}">{html.escape(str(label))}</a>'
            )
        if not links:
            return ""
        return f'<nav class="report-nav" aria-label="Report navigation">{"".join(links)}</nav>'

    def _load_bottleneck_data(self):
        """Load bottleneck detection data - enhanced fault tolerance"""
        # Default bottleneck data structure
        default_data = {
            "timestamp": datetime.now().isoformat(),
            "status": "no_bottleneck_detected",
            "bottleneck_detected": False,
            "bottlenecks": [],
            "bottleneck_types": [],
            "bottleneck_values": [],
            "bottleneck_summary": "No bottleneck detected",
            "detection_time": "",
            "current_qps": 0,
            "last_check": datetime.now().isoformat(),
            "version": "1.0"
        }

        # Try loading bottleneck data from multiple possible locations
        bottleneck_files = []
        if self.bottleneck_info:
            bottleneck_files.append(self.bottleneck_info)

        memory_share_dir = os.getenv('MEMORY_SHARE_DIR', '/tmp/blockchain_monitoring')
        bottleneck_files.extend(self._runtime_file_candidates(
            'BOTTLENECK_STATUS_FILE',
            os.path.join(memory_share_dir, "bottleneck_status.json"),
            os.path.join(self.logs_dir, "bottleneck_status.json"),
            os.path.join(self.output_dir, "bottleneck_status.json"),
        ))

        for bottleneck_file in bottleneck_files:
            try:
                if os.path.exists(bottleneck_file):
                    with open(bottleneck_file, 'r') as f:
                        data = json.load(f)
                        # Validate data structure
                        if isinstance(data, dict) and 'bottleneck_detected' in data:
                            print(f"✅ Successfully loaded bottleneck data: {bottleneck_file}")
                            return data
                        else:
                            print(f"⚠️ Invalid bottleneck data format: {bottleneck_file}")

            except (json.JSONDecodeError, IOError) as e:
                print(f"⚠️ Failed to load bottleneck data {bottleneck_file}: {e}")
                continue

        print(f"ℹ️ No valid bottleneck data file found, using default data")
        return default_data

    def _load_overhead_data(self):
        """Load monitoring overhead data - supports auto-discovery"""
        try:
            # Method 1: Auto-discover monitoring overhead file
            auto_discovered_file = self._find_latest_monitoring_overhead_file()
            if auto_discovered_file:
                self.overhead_csv = auto_discovered_file
                print(f"✅ Auto-discovered monitoring overhead file: {os.path.basename(auto_discovered_file)}")
                return self._load_from_overhead_csv()

            # Method 2: Fallback, extract IOPS data from performance_csv
            if hasattr(self, 'performance_csv') and os.path.exists(self.performance_csv):
                return self._extract_iops_from_performance_csv()

            # Method 3: Last resort, return empty data
            return None
        except Exception as e:
            print(f"Error loading overhead data: {e}")
            return None

    def _load_from_overhead_csv(self):
        """Load data from dedicated overhead CSV"""
        try:
            df = pd.read_csv(self.overhead_csv)
            if df.empty:
                return None

            # Record sample count
            sample_count = len(df)

            # Define required fields and their possible variants
            field_mappings = {
                # Monitoring process resources
                'monitoring_cpu_percent': ['monitoring_cpu_percent', 'monitoring_cpu', 'monitor_cpu', 'overhead_cpu'],
                'monitoring_memory_percent': ['monitoring_memory_percent', 'monitor_memory_percent'],
                'monitoring_memory_mb': ['monitoring_memory_mb', 'monitor_memory', 'overhead_memory'],
                'monitoring_process_count': ['monitoring_process_count', 'process_count', 'monitor_processes'],

                # Blockchain node resources
                'blockchain_cpu_sum': ['blockchain_cpu'],  # Multi-process CPU sum (can be >100%)
                'blockchain_cpu_percent': ['blockchain_cpu_percent'],  # System-wide CPU percentage
                'blockchain_memory_percent': ['blockchain_memory_percent'],
                'blockchain_memory_mb': ['blockchain_memory_mb', 'blockchain_memory'],
                'blockchain_process_count': ['blockchain_process_count'],

                # System static resources
                'system_cpu_cores': ['system_cpu_cores', 'cpu_cores'],
                'system_memory_gb': ['system_memory_gb', 'memory_gb'],
                'system_disk_gb': ['system_disk_gb', 'disk_gb'],

                # System dynamic resources
                'system_cpu_usage': ['system_cpu_usage', 'cpu_usage'],
                'system_memory_usage': ['system_memory_usage', 'memory_usage'],
                'system_disk_usage': ['system_disk_usage', 'disk_usage'],

                'monitoring_iops': ['monitoring_iops', 'monitor_iops', 'overhead_iops'],
                'monitoring_throughput_mibs': ['monitoring_throughput_mibs', 'monitor_throughput', 'overhead_throughput']
            }

            # Try to find matching fields
            data: Dict[str, Union[int, float]] = {'sample_count': sample_count}
            for target_field, possible_fields in field_mappings.items():
                for field in possible_fields:
                    if field in df.columns:
                        # Calculate average and max values
                        data[f'{target_field}_avg'] = df[field].mean()
                        data[f'{target_field}_max'] = df[field].max()
                        # For percentage fields, calculate ratio
                        if 'percent' in target_field or 'usage' in target_field:
                            data[f'{target_field}_p90'] = df[field].quantile(0.9)
                        break

            # Convert blockchain_cpu_sum to system-wide percentage if needed
            if 'blockchain_cpu_sum_avg' in data and 'system_cpu_cores_avg' in data and data['system_cpu_cores_avg'] > 0:
                # blockchain_cpu_sum is multi-process CPU sum (can be >100%)
                # Convert to system-wide percentage: (sum / cores) = percentage
                data['blockchain_cpu_percent_avg'] = data['blockchain_cpu_sum_avg'] / data['system_cpu_cores_avg']
                data['blockchain_cpu_percent_max'] = data.get('blockchain_cpu_sum_max', 0) / data['system_cpu_cores_avg']

            # Calculate monitoring overhead ratio
            if 'monitoring_cpu_percent_avg' in data and 'system_cpu_usage_avg' in data and data['system_cpu_usage_avg'] > 0:
                data['monitoring_cpu_ratio'] = data['monitoring_cpu_percent_avg'] / data['system_cpu_usage_avg']

            if 'monitoring_memory_percent_avg' in data and 'system_memory_usage_avg' in data and data['system_memory_usage_avg'] > 0:
                data['monitoring_memory_ratio'] = data['monitoring_memory_percent_avg'] / data['system_memory_usage_avg']

            # Calculate blockchain node ratio
            if 'blockchain_cpu_percent_avg' in data and 'system_cpu_usage_avg' in data and data['system_cpu_usage_avg'] > 0:
                data['blockchain_cpu_ratio'] = data['blockchain_cpu_percent_avg'] / data['system_cpu_usage_avg']

            if 'blockchain_memory_percent_avg' in data and 'system_memory_usage_avg' in data and data['system_memory_usage_avg'] > 0:
                data['blockchain_memory_ratio'] = data['blockchain_memory_percent_avg'] / data['system_memory_usage_avg']

            return data
        except Exception as e:
            print(f"Error loading from overhead CSV: {e}")
            return None

    def _find_latest_monitoring_overhead_file(self):
        """Auto-discover the latest monitoring overhead file"""
        try:

            # Search for monitoring overhead files
            pattern = os.path.join(self.logs_dir, 'monitoring_overhead_*.csv')
            files = glob.glob(pattern)

            if not files:
                return None

            # Return the latest file (sorted by creation time, consistent with comprehensive_analysis.py)
            latest_file = max(files, key=os.path.getctime)
            return latest_file

        except Exception as e:
            print(f"Warning: Failed to find monitoring overhead file: {e}")
            return None

    def _extract_iops_from_performance_csv(self):
        """Extract IOPS and throughput data from performance CSV"""
        try:
            df = pd.read_csv(self.performance_csv)
            data = {}

            # Extract IOPS data
            if 'monitoring_iops_per_sec' in df.columns:
                iops_data = pd.to_numeric(df['monitoring_iops_per_sec'], errors='coerce').dropna()
                if not iops_data.empty:
                    data['monitoring_iops_avg'] = iops_data.mean()
                    data['monitoring_iops_max'] = iops_data.max()

            # Extract throughput data
            if 'monitoring_throughput_mibs_per_sec' in df.columns:
                throughput_data = pd.to_numeric(df['monitoring_throughput_mibs_per_sec'], errors='coerce').dropna()
                if not throughput_data.empty:
                    data['monitoring_throughput_mibs_avg'] = throughput_data.mean()
                    data['monitoring_throughput_mibs_max'] = throughput_data.max()

            return data if data else None

        except Exception as e:
            # Log error but don't affect main functionality
            print(f"Warning: Failed to extract IOPS data from performance CSV: {e}")
            return None

    def _validate_overhead_csv_format(self):
        """Validate monitoring overhead CSV format"""
        if not self.overhead_csv:
            print("⚠️ Monitoring overhead CSV file not specified")
            return False

        if not os.path.exists(self.overhead_csv):
            print(f"⚠️ Monitoring overhead CSV file does not exist: {self.overhead_csv}")
            return False

        try:
            with open(self.overhead_csv, 'r') as f:
                header = f.readline().strip()
                if not header:
                    print("⚠️ Monitoring overhead CSV file missing header")
                    return False

                field_count = len(header.split(','))
                expected_fields = 20  # Based on configured field count

                if field_count < 10:  # Should have at least 10 basic fields
                    print(f"⚠️ Monitoring overhead CSV has too few fields, expected at least 10, got {field_count}")
                    return False
                elif field_count != expected_fields:
                    print(f"ℹ️ Monitoring overhead CSV field count: {field_count} (expected {expected_fields})")

                # Check if there are data rows
                data_line = f.readline().strip()
                if not data_line:
                    print("⚠️ Monitoring overhead CSV file has no data rows")
                    return False

                print(f"✅ Monitoring overhead CSV format validation passed: {field_count} fields")
                return True

        except Exception as e:
            print(f"❌ CSV format validation failed: {e}")
            return False

    def validate_data_integrity(self):
        """Validate data integrity"""
        validation_results = {
            'performance_csv': False,
            'overhead_csv': False,
            'bottleneck_data': False,
            'config': False
        }

        # Validate performance CSV
        if os.path.exists(self.performance_csv):
            try:
                df = pd.read_csv(self.performance_csv)
                if not df.empty:
                    validation_results['performance_csv'] = True
                    print(f"✅ Performance CSV validation passed: {len(df)} rows of data")
                else:
                    print("⚠️ Performance CSV file is empty")
            except Exception as e:
                print(f"❌ Performance CSV validation failed: {e}")
        else:
            print(f"❌ Performance CSV file does not exist: {self.performance_csv}")

        # Validate overhead CSV
        validation_results['overhead_csv'] = self._validate_overhead_csv_format()

        # Validate bottleneck data
        if self.bottleneck_data and isinstance(self.bottleneck_data, dict):
            # Support both new format (bottleneck_detected) and old format (bottlenecks)
            if 'bottleneck_detected' in self.bottleneck_data or 'bottlenecks' in self.bottleneck_data:
                validation_results['bottleneck_data'] = True
                print("✅ Bottleneck data validation passed")
            else:
                print("⚠️ Bottleneck data format incomplete")
        else:
            print("ℹ️ Using default bottleneck data")
            validation_results['bottleneck_data'] = True  # Default data also counts as passed

        # Validate configuration
        if self.config and isinstance(self.config, dict) and len(self.config) > 0:
            validation_results['config'] = True
            print("✅ Configuration data validation passed")
        else:
            validation_results['config'] = False
            print("ℹ️ Configuration data incomplete (does not affect core functionality)")

        # Output validation summary
        passed = sum(validation_results.values())
        total = len(validation_results)
        print(f"\n📊 Data integrity validation results: {passed}/{total} items passed")

        return validation_results

    def parse_disk_analyzer_log(self):
        """Parse Disk bottleneck detector log file"""
        warnings = []
        performance_metrics = {}

        if not os.path.exists(self.disk_log_path):
            return warnings, performance_metrics

        try:
            with open(self.disk_log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                i = 0
                while i < len(lines):
                    line = lines[i].strip()

                    # Parse bottleneck warning: [time] Disk BOTTLENECK DETECTED: device - type, (Severity: level)
                    if '⚠️' in line and 'Disk BOTTLENECK DETECTED' in line:
                        try:
                            # Extract timestamp
                            timestamp = line.split('[')[1].split(']')[0] if '[' in line and ']' in line else ''

                            # Extract device and type: "nvme2n1 - IOPS"
                            main_part = line.split('Disk BOTTLENECK DETECTED:')[1].split('(Severity:')[0].strip()
                            device = main_part.split('-')[0].strip()
                            bottleneck_type = main_part.split('-')[1].strip().rstrip(',')

                            # Extract severity
                            severity = line.split('Severity:')[1].split(')')[0].strip()

                            # Read next line for detailed metrics
                            if i + 1 < len(lines):
                                detail_line = lines[i + 1].strip()
                                iops_value = ''
                                throughput_value = ''

                                # Extract IOPS: "IOPS: 29788.00/30000 (99%)"
                                if 'IOPS:' in detail_line:
                                    iops_part = detail_line.split('IOPS:')[1].split(',')[0].strip()
                                    iops_value = iops_part.split('/')[0].strip()

                                # Extract Throughput: "Throughput: 2779.61/4000.00 MiB/s (69%)"
                                if 'Throughput:' in detail_line:
                                    throughput_part = detail_line.split('Throughput:')[1].strip()
                                    throughput_value = throughput_part.split('/')[0].strip()

                                # Determine value based on bottleneck type
                                if bottleneck_type == 'IOPS':
                                    value = iops_value
                                    type_label = 'High IOPS'
                                elif bottleneck_type == 'THROUGHPUT':
                                    value = throughput_value
                                    type_label = 'High Throughput'
                                else:
                                    value = iops_value
                                    type_label = 'High Utilization'

                                warnings.append({
                                    'type': type_label,
                                    'device': device,
                                    'value': value,
                                    'timestamp': timestamp,
                                    'data_time': timestamp,
                                    'severity': severity,
                                    'bottleneck_type': bottleneck_type
                                })
                        except (IndexError, ValueError) as e:
                            pass

                    i += 1

        except Exception as e:
            print(f"⚠️ Error parsing Disk log: {e}")

        return warnings, performance_metrics

    def _calculate_data_completeness(self):
        """Calculate monitoring data completeness - file exists, has data, fields complete = 100%"""
        try:
            # Check performance CSV file
            if not os.path.exists(self.performance_csv):
                return 0.0

            perf_df = pd.read_csv(self.performance_csv)

            # Check if there are data rows
            if len(perf_df) == 0:
                return 0.0

            # Check if key fields exist (field completeness)
            required_fields = ['timestamp', 'cpu_usage', 'mem_usage']
            missing_fields = [f for f in required_fields if f not in perf_df.columns]

            if missing_fields:
                # Has missing fields, calculate partial completeness
                return (len(required_fields) - len(missing_fields)) / len(required_fields) * 100

            # File exists + has data + fields complete = 100%
            return 100.0

        except Exception as e:
            print(f"⚠️ Error calculating data completeness: {e}")
            return 0.0

    def _format_monitoring_io(self, value, metric_type='iops'):
        """Format monitoring IO value, display tiny values"""
        if value == 0:
            return "< 0.0001"
        elif value < 0.01:
            return f"{value:.4f}"
        elif metric_type == 'iops':
            return f"{value:.2f}"
        else:
            return f"{value:.4f}"

    def _format_stat_value(self, value, decimal=0):
        """Format statistical value"""
        if isinstance(value, (int, float)):
            if pd.isna(value) or np.isinf(value):
                return 'N/A'
            if decimal == 0:
                return f"{value:.0f}"
            else:
                return f"{value:.{decimal}f}"
        return 'N/A'

    def _numeric_series(self, df, column):
        """Return finite numeric values for one column."""
        if df is None or column not in df.columns:
            return pd.Series(dtype='float64')
        values = pd.to_numeric(df[column], errors='coerce')
        return values.replace([np.inf, -np.inf], np.nan).dropna()

    def _metric_stats(self, df, column, min_threshold=None):
        """Return min/avg/max for a metric column, or None when no valid samples exist."""
        values = self._numeric_series(df, column)
        if values.empty:
            return None
        if min_threshold is None:
            min_values = values
        else:
            min_values = values[values >= min_threshold]
        return {
            'min': min_values.min() if not min_values.empty else values.min(),
            'avg': values.mean(),
            'max': values.max(),
            'count': int(values.count()),
        }

    def _format_metric_stats(self, stats, decimal=0):
        if not stats:
            return ('N/A', 'N/A', 'N/A')
        return (
            self._format_stat_value(stats.get('min'), decimal),
            self._format_stat_value(stats.get('avg'), decimal),
            self._format_stat_value(stats.get('max'), decimal),
        )

    @staticmethod
    def _provider_from_df(df):
        """Provider is read from the CSV cloud_provider column; do not guess or hardcode it.

        Missing or empty cloud_provider falls back to 'other'; the registry uses
        the neutral normalized prefix for unknown providers.
        """
        if df is not None and 'cloud_provider' in df.columns and not df['cloud_provider'].empty:
            val = df['cloud_provider'].dropna()
            if not val.empty:
                provider = str(val.iloc[0]).strip().lower()
                if provider:
                    return provider
        return 'other'

    @classmethod
    def _resolve_disk_columns(cls, df, device_prefix, logical_name):
        """Resolve provider-aware disk columns through CSVSchemaRegistry.

        Physical columns look like '<device_prefix>_<dev>_<dfp>_<suffix>', where
        dev is the runtime device name. registry.resolve(logical_name, provider,
        '') returns the suffix used for exact matching, without retaining
        provider-specific literals.

        device_prefix: 'data' or 'accounts'.
        logical_name : 'disk_iops_provider_adjusted' or 'disk_throughput_provider_adjusted'.
        """
        if df is None:
            return []
        provider = cls._provider_from_df(df)
        # Suffix comes from registry: prefix='' -> '_{dfp}_iops' / '_{dfp}_throughput_mibs'.
        suffix = CSVSchemaRegistry.resolve(logical_name, provider, '')
        prefix = f'{device_prefix}_'
        return [col for col in df.columns if col.startswith(prefix) and col.endswith(suffix)]

    def generate_disk_analysis_section(self, warnings, performance_metrics):
        """Generate Disk analysis report HTML section - enhanced version with dual-layer statistics"""
        if not warnings and not performance_metrics:
            return ""

        # Read CSV data to calculate statistics
        try:
            df = pd.read_csv(self.performance_csv)
        except:
            df = None

        html = f"""
        <div class="section">
            <h2>&#128202; {self.t['disk_performance_analysis']}</h2>

            <div class="subsection">
                <h3>&#9888; {self.t['performance_warnings']}</h3>
        """

        if warnings:
            # Generate statistics summary
            summary = {}
            for w in warnings:
                key = (w['device'], w['type'])
                if key not in summary:
                    summary[key] = {
                        'count': 0,
                        'max_value': 0,
                        'first_time': w['timestamp'],
                        'last_time': w['timestamp']
                    }
                summary[key]['count'] += 1
                try:
                    summary[key]['max_value'] = max(summary[key]['max_value'], float(w['value']))
                except:
                    pass
                summary[key]['last_time'] = w['timestamp']

            # Display summary table
            html += f'''
            <div class="status-callout status-callout-warning">
                <h4>&#128202; {self.t.get('warning_statistics', 'Warning Statistics')}</h4>
                <table class="report-table disk-warning-table">
                    <thead>
                        <tr>
                            <th>{self.t.get('device', 'Device')}</th>
                            <th>{self.t.get('type', 'Type')}</th>
                            <th>{self.t.get('count', 'Count')}</th>
                            <th>{self.t.get('max_value', 'Max Value')}</th>
                            <th>{self.t.get('time_range', 'Time Range')}</th>
                        </tr>
                    </thead>
                    <tbody>
            '''

            for (device, type_label), stats in summary.items():
                html += f'''
                        <tr>
                            <td><strong>{device}</strong></td>
                            <td>{type_label}</td>
                            <td class="numeric-cell metric-value">{stats['count']}</td>
                            <td class="numeric-cell">{stats['max_value']:.2f}</td>
                            <td class="muted-cell">{stats['first_time']} - {stats['last_time']}</td>
                        </tr>
                '''

            html += '''
                    </tbody>
                </table>
                <p class="table-note">
                    &#128202; {tip_text}
                </p>
            </div>
            '''.format(tip_text=self.t.get('refer_to_disk_charts_hint',
                '💡 提示：警告的时间分布可在下方"Disk 专业图表"部分查看 → 点击"Disk 瓶颈分析"和"Disk 时间序列分析"图表' if self.language == 'zh'
                else '💡 Tip: View warning time distribution in "Disk Professional Charts" section below → Click "Disk Bottleneck Analysis" and "Disk Time Series Analysis" charts'))

            # Display detailed list (Top 20) as table
            display_warnings = warnings[:20]
            html += f'<h4>{self.t.get("detailed_warnings", "Detailed Warnings")} ({"Top 20" if len(warnings) > 20 else "All"} / {len(warnings)})</h4>'
            html += '''
            <table class="report-table disk-warning-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>{device}</th>
                        <th>{type}</th>
                        <th class="numeric-cell">{value}</th>
                        <th>{time}</th>
                    </tr>
                </thead>
                <tbody>
            '''.format(
                device=self.t.get('device', 'Device'),
                type=self.t.get('type', 'Type'),
                value=self.t.get('value', 'Value'),
                time=self.t.get('time', 'Time')
            )

            for idx, warning in enumerate(display_warnings, 1):
                color = "#dc3545" if warning['type'] == 'High Utilization' else "#fd7e14"
                unit = "%" if warning['type'] == 'High Utilization' else ""
                html += f'''
                <tr>
                    <td class="muted-cell">{idx}</td>
                    <td><strong>{warning['device']}</strong></td>
                    <td><span class="status-pill status-warn">{warning['type']}</span></td>
                    <td class="numeric-cell metric-value">{warning['value']}{unit}</td>
                    <td class="muted-cell">{warning.get('data_time', warning['timestamp'])}</td>
                </tr>
                '''

            html += '</tbody></table>'

            if len(warnings) > 20:
                html += f'<p style="color: #6c757d; font-style: italic; margin-top: 10px;">... {self.t.get("and", "and")} {len(warnings) - 20} {self.t.get("more_warnings", "more warnings")}. {self.t.get("check_full_log", "Check full log for details")}: <code>{self.disk_log_path}</code></p>'
        else:
            html += f'<p style="color: #28a745; font-weight: bold;">&#9989; {self.t["no_performance_anomaly"]}</p>'

        html += '</div>'

        # Layer 1: provider-neutral disk baseline statistics.
        html += f'''
            <div class="subsection">
                <h3>&#128200; {self.t['disk_baseline_stats']}</h3>
        '''

        if df is not None and not df.empty:
            # Get configured baseline values
            data_max_iops = self.config.get('DATA_VOL_MAX_IOPS', 'N/A')
            data_max_throughput = self.config.get('DATA_VOL_MAX_THROUGHPUT', 'N/A')
            accounts_max_iops = self.config.get('ACCOUNTS_VOL_MAX_IOPS', 'N/A')
            accounts_max_throughput = self.config.get('ACCOUNTS_VOL_MAX_THROUGHPUT', 'N/A')

            # DATA device provider-adjusted fields resolved through the registry.
            data_iops_col = self._resolve_disk_columns(df, 'data', 'disk_iops_provider_adjusted')
            data_throughput_col = self._resolve_disk_columns(df, 'data', 'disk_throughput_provider_adjusted')

            # ACCOUNTS device provider-adjusted fields resolved through the registry.
            accounts_iops_col = self._resolve_disk_columns(df, 'accounts', 'disk_iops_provider_adjusted')
            accounts_throughput_col = self._resolve_disk_columns(df, 'accounts', 'disk_throughput_provider_adjusted')

            # Format values
            data_iops_stats = self._metric_stats(df, data_iops_col[0], 10) if data_iops_col else None
            data_tp_stats = self._metric_stats(df, data_throughput_col[0], 1.0) if data_throughput_col else None
            data_iops_min, data_iops_avg, data_iops_max = self._format_metric_stats(data_iops_stats, 0)
            data_tp_min, data_tp_avg, data_tp_max = self._format_metric_stats(data_tp_stats, 1)

            html += f'''
                <table class="report-table disk-stats-table">
                    <thead>
                        <tr>
                            <th>{self.t['device']}</th>
                            <th>{self.t['metric']}</th>
                            <th>{self.t['baseline_config']}</th>
                            <th>{self.t['min']}</th>
                            <th>{self.t['avg']}</th>
                            <th>{self.t['max']}</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td rowspan="2" class="row-group-label">{self.t['data_device']}</td>
                            <td>{self.t['iops']}</td>
                            <td class="numeric-cell">{data_max_iops}</td>
                            <td class="numeric-cell">{data_iops_min}</td>
                            <td class="numeric-cell metric-value">{data_iops_avg}</td>
                            <td class="numeric-cell">{data_iops_max}</td>
                        </tr>
                        <tr>
                            <td>{self.t['throughput_mibs']}</td>
                            <td class="numeric-cell">{data_max_throughput}</td>
                            <td class="numeric-cell">{data_tp_min}</td>
                            <td class="numeric-cell metric-value">{data_tp_avg}</td>
                            <td class="numeric-cell">{data_tp_max}</td>
                        </tr>
            '''

            # If ACCOUNTS device data exists, add ACCOUNTS rows
            if accounts_iops_col or accounts_throughput_col:
                acc_iops_stats = self._metric_stats(df, accounts_iops_col[0], 10) if accounts_iops_col else None
                acc_tp_stats = self._metric_stats(df, accounts_throughput_col[0], 1.0) if accounts_throughput_col else None
                acc_iops_min, acc_iops_avg, acc_iops_max = self._format_metric_stats(acc_iops_stats, 0)
                acc_tp_min, acc_tp_avg, acc_tp_max = self._format_metric_stats(acc_tp_stats, 1)

                html += f'''
                        <tr>
                            <td rowspan="2" class="row-group-label">{self.t['accounts_device']}</td>
                            <td>{self.t['iops']}</td>
                            <td class="numeric-cell">{accounts_max_iops}</td>
                            <td class="numeric-cell">{acc_iops_min}</td>
                            <td class="numeric-cell metric-value">{acc_iops_avg}</td>
                            <td class="numeric-cell">{acc_iops_max}</td>
                        </tr>
                        <tr>
                            <td>{self.t['throughput_mibs']}</td>
                            <td class="numeric-cell">{accounts_max_throughput}</td>
                            <td class="numeric-cell">{acc_tp_min}</td>
                            <td class="numeric-cell metric-value">{acc_tp_avg}</td>
                            <td class="numeric-cell">{acc_tp_max}</td>
                        </tr>
                '''

            html += '''
                    </tbody>
                </table>
            '''
        else:
            html += f'<p style="color: #6c757d;">{self.t["no_disk_baseline"]}</p>'

        html += '</div>'

        # Second layer: iostat raw sampling data statistics
        html += f'''
            <div class="subsection">
                <h3>&#128200; {self.t["iostat_raw_sampling_stats"]}</h3>
        '''

        if df is not None and not df.empty:
            # DATA Device iostat fields
            iostat_stats = {}
            for metric in ['total_iops', 'total_throughput_mibs', 'util', 'avg_await']:
                data_col = [col for col in df.columns if col.startswith('data_') and col.endswith(f'_{metric}')]
                if data_col:
                    threshold = {'total_iops': 10, 'total_throughput_mibs': 1.0, 'util': 1.0, 'avg_await': 0.1}.get(metric)
                    stats = self._metric_stats(df, data_col[0], threshold)
                    if stats:
                        iostat_stats[f'DATA_{metric}'] = stats

            # ACCOUNTS Device iostat fields
            for metric in ['total_iops', 'total_throughput_mibs', 'util', 'avg_await']:
                accounts_col = [col for col in df.columns if col.startswith('accounts_') and col.endswith(f'_{metric}')]
                if accounts_col:
                    threshold = {'total_iops': 10, 'total_throughput_mibs': 1.0, 'util': 1.0, 'avg_await': 0.1}.get(metric)
                    stats = self._metric_stats(df, accounts_col[0], threshold)
                    if stats:
                        iostat_stats[f'ACCOUNTS_{metric}'] = stats

            html += f'''
                <table class="report-table disk-stats-table">
                    <thead>
                        <tr>
                            <th>{self.t['device']}</th>
                            <th>{self.t['metric']}</th>
                            <th>{self.t['min']}</th>
                            <th>{self.t['avg']}</th>
                            <th>{self.t['max']}</th>
                        </tr>
                    </thead>
                    <tbody>
            '''

            # DATA Device data
            if any(k.startswith('DATA_') for k in iostat_stats.keys()):
                d_iops_min, d_iops_avg, d_iops_max = self._format_metric_stats(iostat_stats.get('DATA_total_iops'), 0)
                d_tp_min, d_tp_avg, d_tp_max = self._format_metric_stats(iostat_stats.get('DATA_total_throughput_mibs'), 1)
                d_util_min, d_util_avg, d_util_max = self._format_metric_stats(iostat_stats.get('DATA_util'), 1)
                d_lat_min, d_lat_avg, d_lat_max = self._format_metric_stats(iostat_stats.get('DATA_avg_await'), 2)

                html += f'''
                        <tr>
                            <td rowspan="4" class="row-group-label">{self.t['data_device']}</td>
                            <td>{self.t['iops']}</td>
                            <td class="numeric-cell">{d_iops_min}</td>
                            <td class="numeric-cell metric-value">{d_iops_avg}</td>
                            <td class="numeric-cell">{d_iops_max}</td>
                        </tr>
                        <tr>
                            <td>{self.t['throughput_mibs']}</td>
                            <td class="numeric-cell">{d_tp_min}</td>
                            <td class="numeric-cell metric-value">{d_tp_avg}</td>
                            <td class="numeric-cell">{d_tp_max}</td>
                        </tr>
                        <tr>
                            <td>{self.t['utilization_pct']}</td>
                            <td class="numeric-cell">{d_util_min}</td>
                            <td class="numeric-cell metric-value">{d_util_avg}</td>
                            <td class="numeric-cell">{d_util_max}</td>
                        </tr>
                        <tr>
                            <td>{self.t['latency_ms']}</td>
                            <td class="numeric-cell">{d_lat_min}</td>
                            <td class="numeric-cell metric-value">{d_lat_avg}</td>
                            <td class="numeric-cell">{d_lat_max}</td>
                        </tr>
                '''

            # ACCOUNTS Device data
            if any(k.startswith('ACCOUNTS_') for k in iostat_stats.keys()):
                a_iops_min, a_iops_avg, a_iops_max = self._format_metric_stats(iostat_stats.get('ACCOUNTS_total_iops'), 0)
                a_tp_min, a_tp_avg, a_tp_max = self._format_metric_stats(iostat_stats.get('ACCOUNTS_total_throughput_mibs'), 1)
                a_util_min, a_util_avg, a_util_max = self._format_metric_stats(iostat_stats.get('ACCOUNTS_util'), 1)
                a_lat_min, a_lat_avg, a_lat_max = self._format_metric_stats(iostat_stats.get('ACCOUNTS_avg_await'), 2)

                html += f'''
                        <tr>
                            <td rowspan="4" class="row-group-label">{self.t['accounts_device']}</td>
                            <td>{self.t['iops']}</td>
                            <td class="numeric-cell">{a_iops_min}</td>
                            <td class="numeric-cell metric-value">{a_iops_avg}</td>
                            <td class="numeric-cell">{a_iops_max}</td>
                        </tr>
                        <tr>
                            <td>{self.t['throughput_mibs']}</td>
                            <td class="numeric-cell">{a_tp_min}</td>
                            <td class="numeric-cell metric-value">{a_tp_avg}</td>
                            <td class="numeric-cell">{a_tp_max}</td>
                        </tr>
                        <tr>
                            <td>{self.t['utilization_pct']}</td>
                            <td class="numeric-cell">{a_util_min}</td>
                            <td class="numeric-cell metric-value">{a_util_avg}</td>
                            <td class="numeric-cell">{a_util_max}</td>
                        </tr>
                        <tr>
                            <td>{self.t['latency_ms']}</td>
                            <td class="numeric-cell">{a_lat_min}</td>
                            <td class="numeric-cell metric-value">{a_lat_avg}</td>
                            <td class="numeric-cell">{a_lat_max}</td>
                        </tr>
                '''

            html += '''
                    </tbody>
                </table>
            '''
        else:
            html += f'<p style="color: #6c757d;">{self.t["no_iostat_data"]}</p>'

        html += '</div>'

        html += '''
        </div>
        '''

        return html
    def generate_html_report(self):
        """Generate HTML report - using safe field access"""
        try:
            df = pd.read_csv(self.performance_csv)

            html_content = self._generate_html_content(df)

            output_file = os.path.join(self.output_dir, f'performance_report_{self.language}_{os.environ.get("SESSION_TIMESTAMP")}.html')
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)

            print(f"✅ Enhanced HTML report generated: {output_file}")
            return output_file

        except Exception as e:
            print(f"❌ HTML report generation failed: {e}")
            return None

    def _generate_config_status_section(self):
        """Generate configuration status check section"""
        ledger_status = f"<span class='status-pill status-ok'>{self.t['configured']}</span>" if self.config.get('LEDGER_DEVICE') else f"<span class='status-pill status-bad'>{self.t['not_configured']}</span>"
        accounts_status = f"<span class='status-pill status-ok'>{self.t['configured']}</span>" if DeviceManager.is_accounts_configured() else f"<span class='status-pill status-warn'>{self.t['not_configured']}</span>"
        blockchain_node = self.config.get('BLOCKCHAIN_NODE', 'General')
        data_volume_status = f"<span class='status-pill status-ok'>{self.t['configured']}</span>" if self.config.get('DATA_VOL_TYPE') else f"<span class='status-pill status-warn'>{self.t['not_configured']}</span>"
        accounts_volume_status = f"<span class='status-pill status-ok'>{self.t['configured']}</span>" if self.config.get('ACCOUNTS_VOL_TYPE') else f"<span class='status-pill status-warn'>{self.t['not_configured']}</span>"

        accounts_note = ""
        if not DeviceManager.is_accounts_configured():
            accounts_note = f'<div class="warning"><strong>{self.t["note"]}:</strong> {self.t["accounts_not_configured_note"]}</div>'

        return f"""
        <div class="section">
            <h2>&#9881; {self.t['config_status_check']}</h2>
            <table class="report-table config-table">
                <thead>
                    <tr>
                        <th>{self.t['config_item']}</th>
                        <th>{self.t['status']}</th>
                        <th>{self.t['value']}</th>
                    </tr>
                </thead>
                <tbody>
                    <tr><td>{self.t['blockchain_node_type']}</td><td><span class="status-pill status-ok">{self.t['configured']}</span></td><td>{html.escape(str(blockchain_node))}</td></tr>
                    <tr><td>{self.t['data_device']}</td><td>{ledger_status}</td><td>{html.escape(str(self.config.get('LEDGER_DEVICE', 'N/A')))}</td></tr>
                    <tr><td>{self.t['accounts_device']}</td><td>{accounts_status}</td><td>{html.escape(str(self.config.get('ACCOUNTS_DEVICE', 'N/A')))}</td></tr>
                    <tr><td>{self.t['data_volume_type']}</td><td>{data_volume_status}</td><td>{html.escape(str(self.config.get('DATA_VOL_TYPE', 'N/A')))}</td></tr>
                    <tr><td>{self.t['accounts_volume_type']}</td><td>{accounts_volume_status}</td><td>{html.escape(str(self.config.get('ACCOUNTS_VOL_TYPE', 'N/A')))}</td></tr>
                </tbody>
            </table>
            {accounts_note}
        </div>
        """

    def _generate_monitoring_overhead_section(self):
        """Generate monitoring overhead section - enhanced with full resource analysis"""
        overhead_data = self.overhead_data  # Use cached data instead of reloading

        if overhead_data:
            # Monitoring process resources
            monitoring_cpu_avg = overhead_data.get('monitoring_cpu_percent_avg', 0)
            monitoring_memory_percent_avg = overhead_data.get('monitoring_memory_percent_avg', 0)
            monitoring_memory_mb_avg = overhead_data.get('monitoring_memory_mb_avg', 0)
            monitoring_process_count = overhead_data.get('monitoring_process_count_avg', 0)

            # Blockchain node resources
            blockchain_cpu_avg = overhead_data.get('blockchain_cpu_percent_avg', 0)
            blockchain_memory_percent_avg = overhead_data.get('blockchain_memory_percent_avg', 0)
            blockchain_memory_mb_avg = overhead_data.get('blockchain_memory_mb_avg', 0)
            blockchain_process_count = overhead_data.get('blockchain_process_count_avg', 0)

            # System resources
            system_cpu_cores = overhead_data.get('system_cpu_cores_avg', 0)
            system_memory_gb = overhead_data.get('system_memory_gb_avg', 0)
            system_cpu_usage_avg = overhead_data.get('system_cpu_usage_avg', 0)
            system_memory_usage_avg = overhead_data.get('system_memory_usage_avg', 0)

            # Resource ratio
            monitoring_cpu_ratio = overhead_data.get('monitoring_cpu_ratio', 0) * 100
            monitoring_memory_ratio = overhead_data.get('monitoring_memory_ratio', 0) * 100
            blockchain_cpu_ratio = overhead_data.get('blockchain_cpu_ratio', 0) * 100
            blockchain_memory_ratio = overhead_data.get('blockchain_memory_ratio', 0) * 100

            # Currently used I/O monitoring fields
            monitoring_iops_avg = overhead_data.get('monitoring_iops_avg', 0)
            monitoring_iops_max = overhead_data.get('monitoring_iops_max', 0)
            monitoring_throughput_avg = overhead_data.get('monitoring_throughput_mibs_avg', 0)
            monitoring_throughput_max = overhead_data.get('monitoring_throughput_mibs_max', 0)

            # Format to two decimal places
            format_num = lambda x: f"{x:.2f}"

            section_html = f"""
            <div class="section">
                <h2>&#128202; {self.t['monitoring_overhead_comprehensive_analysis']}</h2>

                <div class="info-card">
                    <h3>{self.t['system_resource_overview']}</h3>
                    <table class="data-table">
                        <tr>
                            <th>{self.t['metric_label']}</th>
                            <th>{self.t['value_label']}</th>
                        </tr>
                        <tr>
                            <td>{self.t['cpu_cores']}</td>
                            <td>{int(system_cpu_cores)}</td>
                        </tr>
                        <tr>
                            <td>{self.t['total_memory']}</td>
                            <td>{format_num(system_memory_gb)} GB</td>
                        </tr>
                        <tr>
                            <td>{self.t['avg_cpu_usage']}</td>
                            <td>{format_num(system_cpu_usage_avg)}%</td>
                        </tr>
                        <tr>
                            <td>{self.t['avg_memory_usage']}</td>
                            <td>{format_num(system_memory_usage_avg)}%</td>
                        </tr>
                    </table>
                </div>

                <div class="info-card">
                    <h3>{self.t['resource_usage_comparison']}</h3>
                    <table class="data-table">
                        <tr>
                            <th>{self.t['resource_type']}</th>
                            <th>{self.t['monitoring_system']}</th>
                            <th>{self.t['blockchain_node']}</th>
                            <th>{self.t['other_processes']}</th>
                        </tr>
                        <tr>
                            <td>{self.t['cpu_usage_rate']}</td>
                            <td>{format_num(monitoring_cpu_avg)}% ({format_num(monitoring_cpu_ratio)}%)</td>
                            <td>{format_num(blockchain_cpu_avg)}% ({format_num(blockchain_cpu_ratio)}%)</td>
                            <td>{format_num(max(0, system_cpu_usage_avg - monitoring_cpu_avg - blockchain_cpu_avg))}%</td>
                        </tr>
                        <tr>
                            <td>{self.t['memory_usage_rate']}</td>
                            <td>{format_num(monitoring_memory_percent_avg)}%</td>
                            <td>{format_num(blockchain_memory_percent_avg)}%</td>
                            <td>{format_num(max(0, system_memory_usage_avg - monitoring_memory_percent_avg - blockchain_memory_percent_avg))}%</td>
                        </tr>
                        <tr>
                            <td>{self.t['memory_usage_amount']}</td>
                            <td>{format_num(monitoring_memory_mb_avg)} MB</td>
                            <td>{format_num(blockchain_memory_mb_avg/1024)} GB</td>
                            <td>{format_num(system_memory_gb*1024 - monitoring_memory_mb_avg - blockchain_memory_mb_avg)} MB</td>
                        </tr>
                        <tr>
                            <td>{self.t['process_count']}</td>
                            <td>{int(monitoring_process_count)}</td>
                            <td>{int(blockchain_process_count)}</td>
                            <td>N/A</td>
                        </tr>
                    </table>
                    <p class="note">{self.t['percentage_note']}</p>
                </div>

                <div class="info-card">
                    <h3>{self.t['monitoring_io_overhead']}</h3>
                    <table class="data-table">
                        <tr>
                            <th>{self.t['metric_label']}</th>
                            <th>{self.t['average']}</th>
                            <th>{self.t['maximum']}</th>
                        </tr>
                        <tr>
                            <td>{self.t['iops']}</td>
                            <td>{format_num(monitoring_iops_avg)}</td>
                            <td>{format_num(monitoring_iops_max)}</td>
                        </tr>
                        <tr>
                            <td>{self.t['throughput_mibs']}</td>
                            <td>{format_num(monitoring_throughput_avg)}</td>
                            <td>{format_num(monitoring_throughput_max)}</td>
                        </tr>
                    </table>
                </div>

                <div class="conclusion">
                    <h3>&#128221; {self.t['monitoring_overhead_conclusion']}</h3>
                    <p>{self.t['monitoring_resource_analysis']}</p>
                    <ul>
                        <li>{self.t['cpu_overhead']}: {format_num(monitoring_cpu_ratio)}%</li>
                        <li>{self.t['memory_overhead']}: {format_num(monitoring_memory_percent_avg)}% ({format_num(monitoring_memory_mb_avg)} MB)</li>
                        <li>{self.t['io_overhead']}: {format_num(monitoring_iops_avg)} {self.t['iops']}</li>
                    </ul>

                    <p>{self.t['blockchain_resource_analysis']}</p>
                    <ul>
                        <li>{self.t['cpu_usage']}: {format_num(blockchain_cpu_ratio)}%</li>
                        <li>{self.t['memory_usage']}: {format_num(blockchain_memory_percent_avg)}% ({format_num(blockchain_memory_mb_avg/1024)} GB)</li>
                    </ul>

                    <p class="{'warning' if monitoring_cpu_avg > 5 else 'success'}">
                        {self.t['monitoring_impact']}
                        {'<strong>' + self.t['significant'] + '</strong> (' + self.t['monitoring_cpu_exceeds_5'] + ')' if monitoring_cpu_avg > 5 else '<strong>' + self.t['minor'] + '</strong> (' + self.t['monitoring_cpu_below_5'] + ')'}
                    </p>
                </div>
            </div>
            """
        else:
            section_html = f"""
            <div class="section">
                <h2>&#128202; {self.t['monitoring_overhead_comprehensive_analysis']}</h2>
                <div class="warning">
                    <h4>&#9888; {self.t['monitoring_data_unavailable']}</h4>
                    <p>{self.t['monitoring_data_not_found']}</p>
                    <p><strong>{self.t['expected_file']}</strong>: <code>logs/monitoring_overhead_YYYYMMDD_HHMMSS.csv</code></p>
                </div>
                <div class="info">
                    <h4>&#128161; {self.t['how_to_enable']}</h4>
                    <p>{self.t['monitoring_integrated']}</p>
                    <p>{self.t['check_config']}</p>
                    <ul>
                        <li>{self.t['ensure_variable_set']}</li>
                        <li>{self.t['ensure_function_calls']}</li>
                        <li>{self.t['check_permissions']}</li>
                    </ul>
                </div>
            </div>
            """

        return section_html

    def _generate_monitoring_overhead_detailed_section(self):
        """Generate detailed monitoring overhead analysis section"""
        overhead_data = self.overhead_data  # Use cached data instead of reloading

        if overhead_data and os.path.exists(os.path.join(self.output_dir, "monitoring_overhead_analysis.png")):
            # Generate resource usage trend charts
            self._generate_resource_usage_charts()

            section_html = f"""
            <div class="section">
                <h2>&#128200; {self.t['monitoring_overhead_detailed']}</h2>

                <div class="info-card">
                    <h3>&#128202; {self.t['resource_usage_trends']}</h3>
                    <div class="chart-container">
                        <img src="monitoring_overhead_analysis.png" alt="{self.t['monitoring_overhead_analysis']}" class="chart">
                    </div>
                    <div class="chart-info">
                        <p>{self.t['chart_shows_resource_usage_trend']}:</p>
                        <ul>
                            <li><strong>{self.t['monitoring_system_resource_usage']}</strong>: {self.t['cpu_memory_io_overhead_changes']}</li>
                            <li><strong>{self.t['blockchain_node_resource_usage']}</strong>: {self.t['cpu_memory_usage_trends']}</li>
                            <li><strong>{self.t['total_system_resource_usage']}</strong>: {self.t['cpu_memory_usage_entire_system']}</li>
                        </ul>
                    </div>
                </div>

                <div class="info-card">
                    <h3>&#128202; {self.t['resource_proportion_chart']}</h3>
                    <div class="chart-container">
                        <img src="resource_distribution_chart.png" alt="{self.t['resource_distribution_image']}" class="chart">
                    </div>
                    <div class="chart-info">
                        <p>{self.t['chart_shows_component_resources']}</p>
                        <ul>
                            <li><strong>{self.t['monitoring_system']}</strong>: {self.t['all_monitoring_processes']}</li>
                            <li><strong>{self.t['blockchain_node']}</strong>: {self.t['blockchain_related_processes']}</li>
                            <li><strong>{self.t['other_processes']}</strong>: {self.t['other_system_processes']}</li>
                        </ul>
                    </div>
                </div>

                <div class="info-card">
                    <h3>&#128202; {self.t['monitoring_performance_chart']}</h3>
                    <div class="chart-container">
                        <img src="monitoring_impact_chart.png" alt="{self.t['monitoring_impact_image']}" class="chart">
                    </div>
                    <div class="chart-info">
                        <p>{self.t['chart_analyzes_correlation']}</p>
                        <ul>
                            <li><strong>{self.t['monitoring_cpu_vs_qps']}</strong>: {self.t['monitoring_cpu_qps_relationship']}</li>
                            <li><strong>{self.t['monitoring_io_vs_disk']}</strong>: {self.t['monitoring_io_disk_relationship']}</li>
                        </ul>
                    </div>
                </div>
            </div>
            """
        else:
            section_html = f"""
            <div class="section">
                <h2>&#128200; {self.t['monitoring_overhead_detailed']}</h2>
                <div class="warning">
                    <h4>&#9888; {self.t['monitoring_detailed_unavailable']}</h4>
                    <p>{self.t['monitoring_file_not_found']}</p>
                    <ul>
                        <li>{self.t['overhead_csv_generated']}</li>
                        <li>{self.t['chart_script_executed']}</li>
                        <li>{self.t['output_dir_permissions']}</li>
                    </ul>
                </div>
                <div class="info">
                    <h4>&#128161; {self.t['how_to_generate_charts']}</h4>
                    <p>{self.t['use_command_to_generate']}</p>
                    <pre><code>python3 visualization/performance_visualizer.py --performance-csv logs/performance_data.csv --overhead-csv logs/monitoring_overhead.csv --output-dir reports</code></pre>
                </div>
            </div>
            """

        return section_html

    def _generate_resource_usage_charts(self):
        """Generate resource usage trend charts"""
        try:
            if not self.overhead_csv or not os.path.exists(self.overhead_csv):
                return

            df = pd.read_csv(self.overhead_csv)
            if df.empty:
                return

            # Resource distribution pie chart
            self._generate_resource_distribution_chart(df)

            # Monitoring impact analysis chart
            if self.performance_csv and os.path.exists(self.performance_csv):
                self._generate_monitoring_impact_chart(df)

        except Exception as e:
            print(f"Error generating resource usage charts: {e}")

    def _generate_resource_distribution_chart(self, df):
        """Generate resource distribution chart - 3x2 layout (using actual available data)"""
        try:

            UnifiedChartStyle.setup_matplotlib()

            # Read CPU data
            blockchain_cpu = df['blockchain_cpu'].mean() if 'blockchain_cpu' in df.columns else 0
            monitoring_cpu = df['monitoring_cpu'].mean() if 'monitoring_cpu' in df.columns else 0
            system_cpu_cores = df['system_cpu_cores'].mean() if 'system_cpu_cores' in df.columns else 96

            # Read Memory data - using basic fields
            blockchain_memory_mb = df['blockchain_memory_mb'].mean() if 'blockchain_memory_mb' in df.columns else 0
            monitoring_memory_mb = df['monitoring_memory_mb'].mean() if 'monitoring_memory_mb' in df.columns else 0
            system_memory_gb = df['system_memory_gb'].mean() if 'system_memory_gb' in df.columns else 739.70

            # Read basic memory data from performance CSV (unit: MB, needs conversion to GB)
            mem_used_mb = 0
            mem_total_mb = system_memory_gb * 1024
            if self.performance_csv and os.path.exists(self.performance_csv):
                try:
                    perf_df = pd.read_csv(self.performance_csv, usecols=['mem_used', 'mem_total'])
                    mem_used_mb = perf_df['mem_used'].mean() if 'mem_used' in perf_df.columns else 0
                    mem_total_mb = perf_df['mem_total'].mean() if 'mem_total' in perf_df.columns else system_memory_gb * 1024
                except Exception as e:
                    print(f"⚠️ {self.t['read_memory_data_failed']}: {e}")

            # Convert to GB
            mem_used_gb = mem_used_mb / 1024
            mem_total_gb = mem_total_mb / 1024

            # Read Network data
            net_total_gbps = 0
            network_max_gbps = 25
            if self.performance_csv and os.path.exists(self.performance_csv):
                try:
                    perf_df = pd.read_csv(self.performance_csv, usecols=['net_total_gbps'])
                    net_total_gbps = perf_df['net_total_gbps'].mean() if 'net_total_gbps' in perf_df.columns else 0
                    network_max_gbps = float(os.getenv('NETWORK_MAX_BANDWIDTH_GBPS', '25'))
                except Exception as e:
                    print(f"⚠️ {self.t['read_network_data_failed']}: {e}")

            # Calculate derived metrics
            blockchain_cores = blockchain_cpu / 100 if blockchain_cpu > 0 else 0
            monitoring_cores = monitoring_cpu / 100 if monitoring_cpu > 0 else 0
            idle_cores = max(0, system_cpu_cores - blockchain_cores - monitoring_cores)

            blockchain_memory_gb = blockchain_memory_mb / 1024
            monitoring_memory_gb = monitoring_memory_mb / 1024
            mem_free_gb = max(0, mem_total_gb - mem_used_gb)

            network_used_gbps = net_total_gbps
            network_available_gbps = max(0, network_max_gbps - net_total_gbps)
            network_utilization = (net_total_gbps / network_max_gbps * 100) if network_max_gbps > 0 else 0

            # Create 3x2 layout
            fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16, 18))
            fig.suptitle('System Resource Distribution Analysis',
                        fontsize=UnifiedChartStyle.FONT_CONFIG['title_size'], fontweight='bold', y=0.995)

            # Subplot 1: CPU Core Usage
            cpu_sizes = [blockchain_cores, monitoring_cores, idle_cores]
            cpu_labels = [f'Blockchain\n{blockchain_cores:.2f} cores',
                         f'Monitoring\n{monitoring_cores:.2f} cores',
                         f'Idle\n{idle_cores:.2f} cores']
            cpu_colors = [UnifiedChartStyle.COLORS['data_primary'], UnifiedChartStyle.COLORS['warning'],
                         UnifiedChartStyle.COLORS['success']]
            wedges1, texts1, autotexts1 = ax1.pie(cpu_sizes, labels=cpu_labels, colors=cpu_colors,
                                                   autopct='%1.1f%%', startangle=45, labeldistance=1.15,
                                                   textprops={'fontsize': UnifiedChartStyle.FONT_CONFIG['legend_size']})
            ax1.set_title(f'CPU Core Usage (Total: {system_cpu_cores:.0f} cores)',
                         fontsize=UnifiedChartStyle.FONT_CONFIG['subtitle_size'], pad=15)
            for autotext in autotexts1:
                autotext.set_fontsize(UnifiedChartStyle.FONT_CONFIG['text_size'])
                autotext.set_color('white')
                autotext.set_weight('bold')

            # Subplot 2: Memory Usage Distribution
            mem_sizes = [blockchain_memory_gb, monitoring_memory_gb, mem_free_gb]
            mem_labels = [f'Blockchain\n{blockchain_memory_gb:.2f} GB',
                         f'Monitoring\n{monitoring_memory_gb:.2f} GB',
                         f'Free\n{mem_free_gb:.2f} GB']
            mem_colors = [UnifiedChartStyle.COLORS['data_primary'], UnifiedChartStyle.COLORS['warning'],
                         UnifiedChartStyle.COLORS['success']]
            wedges2, texts2, autotexts2 = ax2.pie(mem_sizes, labels=mem_labels, colors=mem_colors,
                                                   autopct='%1.1f%%', startangle=45, labeldistance=1.15,
                                                   textprops={'fontsize': UnifiedChartStyle.FONT_CONFIG['legend_size']})
            ax2.set_title(f'Memory Usage Distribution (Total: {mem_total_gb:.0f} GB)',
                         fontsize=UnifiedChartStyle.FONT_CONFIG['subtitle_size'], pad=15)
            for autotext in autotexts2:
                autotext.set_fontsize(UnifiedChartStyle.FONT_CONFIG['text_size'])
                autotext.set_color('white')
                autotext.set_weight('bold')

            # Subplot 3: Memory Usage Comparison
            mem_categories = ['Blockchain', 'Monitoring', 'Free']
            mem_values = [blockchain_memory_gb, monitoring_memory_gb, mem_free_gb]
            mem_bar_colors = [UnifiedChartStyle.COLORS['data_primary'], UnifiedChartStyle.COLORS['warning'],
                             UnifiedChartStyle.COLORS['success']]
            bars3 = ax3.bar(mem_categories, mem_values, color=mem_bar_colors, alpha=0.7)
            ax3.set_ylabel('Memory (GB)', fontsize=UnifiedChartStyle.FONT_CONFIG['label_size'])
            ax3.set_title('Memory Usage Comparison', fontsize=UnifiedChartStyle.FONT_CONFIG['subtitle_size'], pad=15)
            ax3.grid(True, alpha=0.3, axis='y')
            for bar, val in zip(bars3, mem_values):
                pct = (val / mem_total_gb * 100) if mem_total_gb > 0 else 0
                ax3.text(bar.get_x() + bar.get_width()/2, val + max(mem_values)*0.02,
                        f'{val:.1f} GB\n({pct:.1f}%)', ha='center', fontsize=UnifiedChartStyle.FONT_CONFIG['text_size'])

            # Subplot 4: CPU Usage Comparison
            cpu_categories = ['Blockchain', 'Monitoring']
            cpu_values = [blockchain_cpu, monitoring_cpu]
            cpu_bar_colors = [UnifiedChartStyle.COLORS['data_primary'], UnifiedChartStyle.COLORS['warning']]
            bars4 = ax4.bar(cpu_categories, cpu_values, color=cpu_bar_colors, alpha=0.7)
            ax4.set_ylabel('CPU Usage (%)', fontsize=UnifiedChartStyle.FONT_CONFIG['label_size'])
            ax4.set_title('CPU Usage Comparison', fontsize=UnifiedChartStyle.FONT_CONFIG['subtitle_size'], pad=15)
            ax4.grid(True, alpha=0.3, axis='y')
            for bar, val in zip(bars4, cpu_values):
                ax4.text(bar.get_x() + bar.get_width()/2, val + max(cpu_values)*0.02,
                        f'{val:.2f}%', ha='center', fontsize=UnifiedChartStyle.FONT_CONFIG['text_size'])

            # Subplot 5: Network Bandwidth
            if net_total_gbps > 0:
                net_sizes = [network_used_gbps, network_available_gbps]
                net_labels = [f'Used\n{network_used_gbps:.2f} Gbps',
                             f'Available\n{network_available_gbps:.2f} Gbps']
                net_colors = [UnifiedChartStyle.COLORS['critical'], UnifiedChartStyle.COLORS['success']]
                wedges5, texts5, autotexts5 = ax5.pie(net_sizes, labels=net_labels, colors=net_colors,
                                                      autopct='%1.1f%%', startangle=45, labeldistance=1.15,
                                                      textprops={'fontsize': UnifiedChartStyle.FONT_CONFIG['legend_size']})
                ax5.set_title(f'Network Bandwidth (Max: {network_max_gbps:.0f} Gbps, Util: {network_utilization:.2f}%)',
                             fontsize=UnifiedChartStyle.FONT_CONFIG['subtitle_size'], pad=15)
                for autotext in autotexts5:
                    autotext.set_fontsize(UnifiedChartStyle.FONT_CONFIG['text_size'])
                    autotext.set_color('white')
                    autotext.set_weight('bold')
            else:
                ax5.text(0.5, 0.5, 'Network Data\nUnavailable',
                        ha='center', va='center', fontsize=UnifiedChartStyle.FONT_CONFIG['subtitle_size'],
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
                ax5.set_title('Network Bandwidth', fontsize=UnifiedChartStyle.FONT_CONFIG['subtitle_size'], pad=15)
                ax5.axis('off')

            # Subplot 6: Resource Overhead Summary
            overhead_categories = ['CPU\nOverhead', 'Memory\nOverhead']
            total_cpu = blockchain_cpu + monitoring_cpu
            total_mem = blockchain_memory_gb + monitoring_memory_gb
            cpu_overhead_pct = (monitoring_cpu / total_cpu * 100) if total_cpu > 0 else 0
            mem_overhead_pct = (monitoring_memory_gb / total_mem * 100) if total_mem > 0 else 0
            overhead_values = [cpu_overhead_pct, mem_overhead_pct]
            overhead_colors = [UnifiedChartStyle.COLORS['warning'], UnifiedChartStyle.COLORS['info']]
            bars6 = ax6.bar(overhead_categories, overhead_values, color=overhead_colors, alpha=0.7)
            ax6.set_ylabel('Overhead (%)', fontsize=UnifiedChartStyle.FONT_CONFIG['label_size'])
            ax6.set_title('Monitoring Overhead Comparison', fontsize=UnifiedChartStyle.FONT_CONFIG['subtitle_size'], pad=15)
            ax6.grid(True, alpha=0.3, axis='y')
            for bar, val in zip(bars6, overhead_values):
                ax6.text(bar.get_x() + bar.get_width()/2, val + max(overhead_values)*0.02, f'{val:.2f}%',
                        ha='center', fontsize=UnifiedChartStyle.FONT_CONFIG['text_size'])

            UnifiedChartStyle.apply_layout('auto')
            reports_dir = self.output_dir
            plt.savefig(os.path.join(reports_dir, 'resource_distribution_chart.png'), dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✅ Generation complete: resource_distribution_chart.png")

        except Exception as e:
            print(f"❌ {self.t['resource_distribution_failed']}: {e}")
            import traceback
            traceback.print_exc()

    def _generate_monitoring_impact_chart(self, overhead_df):
        """Generate monitoring impact analysis chart - 3x2 layout (using actual available data)"""
        try:

            UnifiedChartStyle.setup_matplotlib()

            # Read performance data
            perf_df = pd.read_csv(self.performance_csv) if self.performance_csv and os.path.exists(self.performance_csv) else pd.DataFrame()

            # Calculate averages - from overhead CSV
            blockchain_cpu = overhead_df['blockchain_cpu'].mean() if 'blockchain_cpu' in overhead_df.columns else 0
            monitoring_cpu = overhead_df['monitoring_cpu'].mean() if 'monitoring_cpu' in overhead_df.columns else 0
            blockchain_memory_mb = overhead_df['blockchain_memory_mb'].mean() if 'blockchain_memory_mb' in overhead_df.columns else 0
            monitoring_memory_mb = overhead_df['monitoring_memory_mb'].mean() if 'monitoring_memory_mb' in overhead_df.columns else 0
            system_cpu_cores = overhead_df['system_cpu_cores'].mean() if 'system_cpu_cores' in overhead_df.columns else 96
            system_memory_gb = overhead_df['system_memory_gb'].mean() if 'system_memory_gb' in overhead_df.columns else 739.70

            # Get I/O data and basic memory data from performance CSV
            monitoring_iops = perf_df['monitoring_iops_per_sec'].mean() if not perf_df.empty and 'monitoring_iops_per_sec' in perf_df.columns else 0
            monitoring_throughput = perf_df['monitoring_throughput_mibs_per_sec'].mean() if not perf_df.empty and 'monitoring_throughput_mibs_per_sec' in perf_df.columns else 0

            # Use basic memory data from performance CSV (unit: MB, needs conversion to GB)
            mem_used_mb = perf_df['mem_used'].mean() if not perf_df.empty and 'mem_used' in perf_df.columns else 0
            mem_total_mb = perf_df['mem_total'].mean() if not perf_df.empty and 'mem_total' in perf_df.columns else system_memory_gb * 1024
            mem_usage_pct = perf_df['mem_usage'].mean() if not perf_df.empty and 'mem_usage' in perf_df.columns else 0

            # Convert to GB
            mem_used = mem_used_mb / 1024
            mem_total = mem_total_mb / 1024

            # Convert to cores and GB
            blockchain_cores = blockchain_cpu / 100
            monitoring_cores = monitoring_cpu / 100
            blockchain_memory_gb = blockchain_memory_mb / 1024
            monitoring_memory_gb = monitoring_memory_mb / 1024

            # Calculate proportions
            total_cpu = blockchain_cpu + monitoring_cpu
            cpu_overhead_pct = (monitoring_cpu / total_cpu * 100) if total_cpu > 0 else 0
            total_memory = blockchain_memory_gb + monitoring_memory_gb
            memory_overhead_pct = (monitoring_memory_gb / total_memory * 100) if total_memory > 0 else 0

            # Create 3x2 layout
            fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16, 18))
            fig.suptitle('Monitoring Overhead Impact Analysis',
                        fontsize=UnifiedChartStyle.FONT_CONFIG['title_size'], fontweight='bold', y=0.995)

            # Subplot 1: CPU Core Usage
            cpu_categories = ['Blockchain', 'Monitoring']
            cpu_values = [blockchain_cores, monitoring_cores]
            cpu_colors = [UnifiedChartStyle.COLORS['data_primary'], UnifiedChartStyle.COLORS['warning']]
            bars1 = ax1.bar(cpu_categories, cpu_values, color=cpu_colors, alpha=0.7)
            ax1.set_ylabel('CPU Cores', fontsize=UnifiedChartStyle.FONT_CONFIG['label_size'])
            ax1.set_title(f'CPU Core Usage (Total: {system_cpu_cores:.0f} cores)',
                         fontsize=UnifiedChartStyle.FONT_CONFIG['subtitle_size'], pad=15)
            ax1.grid(True, alpha=0.3, axis='y')
            for bar, val in zip(bars1, cpu_values):
                pct = (val / system_cpu_cores * 100) if system_cpu_cores > 0 else 0
                ax1.text(bar.get_x() + bar.get_width()/2, val + max(cpu_values)*0.02,
                        f'{val:.2f}\n({pct:.1f}%)', ha='center', fontsize=UnifiedChartStyle.FONT_CONFIG['text_size'])

            # Subplot 2: Memory Usage
            mem_categories = ['Blockchain', 'Monitoring']
            mem_values = [blockchain_memory_gb, monitoring_memory_gb]
            mem_colors = [UnifiedChartStyle.COLORS['data_primary'], UnifiedChartStyle.COLORS['warning']]
            bars2 = ax2.bar(mem_categories, mem_values, color=mem_colors, alpha=0.7)
            ax2.set_ylabel('Memory (GB)', fontsize=UnifiedChartStyle.FONT_CONFIG['label_size'])
            ax2.set_title(f'Memory Usage (Total: {system_memory_gb:.1f} GB)',
                         fontsize=UnifiedChartStyle.FONT_CONFIG['subtitle_size'], pad=15)
            ax2.grid(True, alpha=0.3, axis='y')
            for bar, val in zip(bars2, mem_values):
                pct = (val / system_memory_gb * 100) if system_memory_gb > 0 else 0
                ax2.text(bar.get_x() + bar.get_width()/2, val + max(mem_values)*0.02,
                        f'{val:.2f}\n({pct:.1f}%)', ha='center', fontsize=UnifiedChartStyle.FONT_CONFIG['text_size'])

            # Subplot 3: Monitoring I/O Impact
            # Calculate I/O overhead percentage - use DeviceManager to dynamically get fields
            device_manager = DeviceManager(perf_df) if not perf_df.empty else None
            data_total_iops = 0
            accounts_total_iops = 0

            if device_manager:
                data_iops_field = device_manager.get_mapped_field('data_total_iops')
                if data_iops_field and data_iops_field in perf_df.columns:
                    data_total_iops = perf_df[data_iops_field].mean()

                if device_manager.is_accounts_configured():
                    accounts_iops_field = device_manager.get_mapped_field('accounts_total_iops')
                    if accounts_iops_field and accounts_iops_field in perf_df.columns:
                        accounts_total_iops = perf_df[accounts_iops_field].mean()

            total_system_iops = data_total_iops + accounts_total_iops
            io_overhead_pct = (monitoring_iops / total_system_iops * 100) if total_system_iops > 0 else 0

            if monitoring_iops > 0.01 or monitoring_throughput > 0.01:
                io_categories = ['IOPS/sec', 'Throughput\n(MiB/s)']
                io_values = [monitoring_iops, monitoring_throughput]
                io_colors = [UnifiedChartStyle.COLORS['data_primary'], UnifiedChartStyle.COLORS['success']]
                bars3 = ax3.bar(io_categories, io_values, color=io_colors, alpha=0.7)
                ax3.set_ylabel('Monitoring I/O', fontsize=UnifiedChartStyle.FONT_CONFIG['label_size'])
                ax3.set_title('Monitoring I/O Operations', fontsize=UnifiedChartStyle.FONT_CONFIG['subtitle_size'], pad=15)
                ax3.grid(True, alpha=0.3, axis='y')
                for bar, val in zip(bars3, io_values):
                    if max(io_values) > 0:
                        ax3.text(bar.get_x() + bar.get_width()/2, val + max(io_values)*0.02,
                                f'{val:.3f}', ha='center', fontsize=UnifiedChartStyle.FONT_CONFIG['text_size'])
            else:
                ax3.text(0.5, 0.5, 'Monitoring I/O Data Unavailable\n(All values are 0)',
                        ha='center', va='center', transform=ax3.transAxes,
                        fontsize=UnifiedChartStyle.FONT_CONFIG['subtitle_size'],
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
                ax3.set_title('Monitoring I/O Operations', fontsize=UnifiedChartStyle.FONT_CONFIG['subtitle_size'], pad=15)
                ax3.set_xlabel('I/O Type', fontsize=UnifiedChartStyle.FONT_CONFIG['label_size'])
                ax3.set_ylabel('Monitoring I/O', fontsize=UnifiedChartStyle.FONT_CONFIG['label_size'])
                ax3.grid(True, alpha=0.3)

            # Subplot 4: System Memory Overview (using basic memory data)
            if mem_used > 0 and mem_total > 0:
                mem_free = mem_total - mem_used
                mem_overview_labels = ['Used', 'Free']
                mem_overview_values = [mem_used, mem_free]
                mem_overview_colors = [UnifiedChartStyle.COLORS['warning'], UnifiedChartStyle.COLORS['success']]
                bars4 = ax4.bar(mem_overview_labels, mem_overview_values, color=mem_overview_colors, alpha=0.7)
                ax4.set_ylabel('Memory (GB)', fontsize=UnifiedChartStyle.FONT_CONFIG['label_size'])
                ax4.set_title(f'System Memory Overview (Usage: {mem_usage_pct:.1f}%)',
                             fontsize=UnifiedChartStyle.FONT_CONFIG['subtitle_size'], pad=15)
                ax4.grid(True, alpha=0.3, axis='y')
                for bar, val in zip(bars4, mem_overview_values):
                    pct = (val / mem_total * 100) if mem_total > 0 else 0
                    ax4.text(bar.get_x() + bar.get_width()/2, val + max(mem_overview_values)*0.02,
                            f'{val:.1f} GB\n({pct:.1f}%)', ha='center', fontsize=UnifiedChartStyle.FONT_CONFIG['text_size'])
            else:
                ax4.text(0.5, 0.5, 'Memory Data\nNot Available', ha='center', va='center',
                        fontsize=UnifiedChartStyle.FONT_CONFIG['subtitle_size'])
                ax4.set_title('System Memory Overview', fontsize=UnifiedChartStyle.FONT_CONFIG['subtitle_size'], pad=15)
                ax4.axis('off')

            # Subplot 5: CPU Overhead Trend
            if 'timestamp' in overhead_df.columns and 'monitoring_cpu' in overhead_df.columns and 'blockchain_cpu' in overhead_df.columns:
                if not pd.api.types.is_datetime64_any_dtype(overhead_df['timestamp']):
                    overhead_df['timestamp'] = pd.to_datetime(overhead_df['timestamp'])

                overhead_df['cpu_overhead_pct'] = (overhead_df['monitoring_cpu'] /
                                                   (overhead_df['blockchain_cpu'] + overhead_df['monitoring_cpu']) * 100)
                ax5.plot(overhead_df['timestamp'], overhead_df['cpu_overhead_pct'],
                        linewidth=2, color=UnifiedChartStyle.COLORS['warning'], alpha=0.7)
                ax5.axhline(y=overhead_df['cpu_overhead_pct'].mean(), color=UnifiedChartStyle.COLORS['critical'],
                           linestyle='--', alpha=0.7, label=f'Average: {overhead_df["cpu_overhead_pct"].mean():.2f}%')
                ax5.set_ylabel('Overhead (%)', fontsize=UnifiedChartStyle.FONT_CONFIG['label_size'])
                ax5.set_title('CPU Overhead Trend Over Time', fontsize=UnifiedChartStyle.FONT_CONFIG['subtitle_size'], pad=15)
                ax5.legend()
                ax5.grid(True, alpha=0.3)
                format_time_axis(ax5, overhead_df['timestamp'])
            else:
                ax5.text(0.5, 0.5, 'No Time Series Data', ha='center', va='center',
                        fontsize=UnifiedChartStyle.FONT_CONFIG['subtitle_size'])
                ax5.axis('off')

            # Subplot 6: Monitoring Efficiency Summary
            summary_lines = [
                "Monitoring Overhead Summary:",
                "",
                f"CPU Overhead:     {cpu_overhead_pct:.2f}%",
                f"Memory Overhead:  {memory_overhead_pct:.2f}%",
                f"I/O Overhead:     {io_overhead_pct:.2f}%",
                "",
                "Absolute Values:",
                f"  CPU:        {monitoring_cores:.2f} cores",
                f"  Memory:     {monitoring_memory_gb:.2f} GB",
                f"  IOPS:       {monitoring_iops:.0f}",
                f"  Throughput: {monitoring_throughput:.2f} MiB/s",
                "",
                f"Efficiency Rating: {'Excellent' if cpu_overhead_pct < 1 else 'Good' if cpu_overhead_pct < 3 else 'Needs Optimization'}"
            ]
            summary_text = "\n".join(summary_lines)
            UnifiedChartStyle.add_text_summary(ax6, summary_text, 'Monitoring Efficiency Summary')

            UnifiedChartStyle.apply_layout('auto')
            reports_dir = self.output_dir
            plt.savefig(os.path.join(reports_dir, 'monitoring_impact_chart.png'), dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✅ {self.t['generation_complete_monitoring']}")

        except Exception as e:
            print(f"❌ {self.t['monitoring_impact_failed']}: {e}")
            import traceback
            traceback.print_exc()

    def _generate_disk_bottleneck_section(self):
        """Generate Disk bottleneck analysis section - enhanced version with multi-device and root cause analysis"""
        bottleneck_info = self._load_bottleneck_data()
        overhead_data = self.overhead_data  # Use cached data instead of reloading

        # Device type list
        device_types = ['data', 'accounts']
        device_labels = {'data': 'DATA', 'accounts': 'ACCOUNTS'}

        if bottleneck_info and 'disk_bottlenecks' in bottleneck_info:
            disk_bottlenecks = bottleneck_info['disk_bottlenecks']

            # Group bottlenecks by device type
            device_bottlenecks = {}
            for bottleneck in disk_bottlenecks:
                device_type = bottleneck.get('device_type', 'data').lower()
                if device_type not in device_bottlenecks:
                    device_bottlenecks[device_type] = []
                device_bottlenecks[device_type].append(bottleneck)

            # Generate device bottleneck HTML
            devices_html = ""
            for device_type in device_types:
                if device_type in device_bottlenecks and device_bottlenecks[device_type]:
                    # This device has bottlenecks
                    bottlenecks = device_bottlenecks[device_type]

                    # Format bottleneck information
                    bottleneck_html = ""
                    for bottleneck in bottlenecks:
                        bottleneck_type = bottleneck.get('type', 'Unknown')
                        severity = bottleneck.get('severity', 'Medium')
                        details = bottleneck.get('details', {})

                        # Format details
                        details_html = ""
                        for key, value in details.items():
                            details_html += f"<li><strong>{key}:</strong> {value}</li>"

                        bottleneck_html += f"""
                        <div class="bottleneck-item severity-{severity.lower()}">
                            <h4>{bottleneck_type} <span class="severity">{severity}</span></h4>
                            <ul>
                                {details_html}
                            </ul>
                        </div>
                        """

                    # Get monitoring overhead data for root cause analysis
                    root_cause_html = self._generate_bottleneck_root_cause_analysis(device_type, overhead_data)

                    devices_html += f"""
                    <div class="device-bottleneck">
                        <h3>&#128192; {device_labels[device_type]}{self.t['device_bottleneck']}</h3>
                        <div class="bottleneck-container">
                            {bottleneck_html}
                        </div>
                        {root_cause_html}
                    </div>
                    """
                elif device_type == 'data':
                    # DATA device must be displayed even if no bottleneck
                    devices_html += f"""
                    <div class="device-bottleneck">
                        <h3>&#128192; {device_labels[device_type]}{self.t['device_label']}</h3>
                        <div class="success">
                            <h4>&#9989; {self.t['no_bottleneck_detected']}</h4>
                            <p>{device_labels[device_type]}{self.t['device_performance_good']}</p>
                        </div>
                    </div>
                    """

            section_html = f"""
            <div class="section">
                <h2>&#128192; {self.t['disk_bottleneck_analysis']}</h2>
                {devices_html}
                <div class="note">
                    <p>{self.t['disk_analysis_based_on']}</p>
                    <p>{self.t['root_cause_based_on']}</p>
                </div>
            </div>
            """
        else:
            section_html = f"""
            <div class="section">
                <h2>&#128192; {self.t['disk_bottleneck_analysis']}</h2>
                <div class="success">
                    <h4>&#9989; {self.t['no_disk_bottleneck_detected']}</h4>
                    <p>{self.t['no_disk_bottleneck_found']}</p>
                </div>
            </div>
            """

        return section_html

    def _generate_bottleneck_root_cause_analysis(self, device_type, overhead_data):
        """Generate bottleneck root cause analysis HTML"""
        if not overhead_data:
            return f"""
            <div class="warning">
                <h4>&#9888; {self.t['cannot_perform_root_cause']}</h4>
                <p>{self.t['missing_overhead_data']}</p>
            </div>
            """

        # Get monitoring overhead data
        monitoring_iops_avg = overhead_data.get('monitoring_iops_avg', 0)
        monitoring_throughput_avg = overhead_data.get('monitoring_throughput_mibs_avg', 0)

        # Estimate monitoring overhead impact on Disk
        # Note: Monitoring system reads /proc virtual filesystem, IOPS usually < 0.01
        if monitoring_iops_avg > 1.0:  # Very high (abnormal)
            impact_level = self.t['high_label']
            impact_percent = min(90, monitoring_iops_avg * 50)  # 1 IOPS = 50%
        elif monitoring_iops_avg > 0.1:  # High
            impact_level = self.t['medium_label']
            impact_percent = min(50, monitoring_iops_avg * 100)  # 0.1 IOPS = 10%
        else:  # Normal (< 0.1 IOPS)
            impact_level = self.t['low_label']
            impact_percent = min(20, monitoring_iops_avg * 200)  # 0.01 IOPS = 2%

        # Generate different HTML based on impact level
        if impact_level == self.t['high_label']:
            return f"""
            <div class="root-cause-analysis warning">
                <h4>&#128269; {self.t['root_cause_significant_impact']}</h4>
                <p>{self.t['monitoring_impact_level']}: <strong>{impact_level} ({self.t['about']}{impact_percent:.1f}%)</strong></p>
                <ul>
                    <li>{self.t['monitoring_avg_iops']}: <strong>{monitoring_iops_avg:.2f}</strong></li>
                    <li>{self.t['monitoring_avg_throughput']}: <strong>{monitoring_throughput_avg:.2f} MiB/s</strong></li>
                </ul>
                <p class="recommendation">{self.t['recommendation']}: {self.t['reduce_monitoring_frequency']}{device_type.upper()}{self.t['device_impact_suffix']}</p>
            </div>
            """
        elif impact_level == self.t['medium_label']:
            return f"""
            <div class="root-cause-analysis info">
                <h4>&#128269; {self.t['root_cause_moderate_impact']}</h4>
                <p>{self.t['monitoring_impact_level']}: <strong>{impact_level} ({self.t['about']}{impact_percent:.1f}%)</strong></p>
                <ul>
                    <li>{self.t['monitoring_avg_iops']}: <strong>{monitoring_iops_avg:.2f}</strong></li>
                    <li>{self.t['monitoring_avg_throughput']}: <strong>{monitoring_throughput_avg:.2f} MiB/s</strong></li>
                </ul>
                <p class="recommendation">{self.t['recommendation']}: {self.t['monitoring_has_some_impact']}</p>
            </div>
            """
        else:
            return f"""
            <div class="root-cause-analysis success">
                <h4>&#128269; {self.t['root_cause_minor_impact']}</h4>
                <p>{self.t['monitoring_impact_level']}: <strong>{impact_level} ({self.t['about']}{impact_percent:.1f}%)</strong></p>
                <ul>
                    <li>{self.t['monitoring_avg_iops']}: <strong>{monitoring_iops_avg:.2f}</strong></li>
                    <li>{self.t['monitoring_avg_throughput']}: <strong>{monitoring_throughput_avg:.2f} MiB/s</strong></li>
                </ul>
                <p class="recommendation">{self.t['recommendation']}: {device_type.upper()}{self.t['bottleneck_from_workload']}</p>
            </div>
            """



    def _generate_overhead_data_table(self):
        """Generate complete monitoring overhead data table"""
        if not self.overhead_data:
            return f"""
            <h2>&#128202; {self.t['monitoring_overhead_breakdown']}</h2>
            <div class="warning">
                <h4>&#9888; {self.t['overhead_data_not_available']}</h4>
                <p>{self.t['overhead_file_not_found']}</p>
                <p><strong>{self.t['expected_file_label']}</strong>: <code>logs/monitoring_overhead_YYYYMMDD_HHMMSS.csv</code></p>
                <p><strong>{self.t['description_label']}</strong>: {self.t['overhead_auto_generated']}</p>
            </div>
            """

        try:
            # Generate detailed monitoring overhead table
            table_html = f"""
            <h2>&#128202; {self.t['monitoring_overhead_breakdown']}</h2>
            <div class="info">
                <h4>&#128202; {self.t['overhead_detailed_data']}</h4>
                <p>{self.t['data_shows_component_consumption']}</p>
            </div>

            <table class="report-table monitoring-overhead-table">
                <thead>
                    <tr>
                        <th>{self.t['monitoring_component_label']}</th>
                        <th>{self.t['avg_cpu_usage_label']}</th>
                        <th>{self.t['peak_cpu_usage_label']}</th>
                        <th>{self.t['avg_memory_usage_label']}</th>
                        <th>{self.t['peak_memory_usage_label']}</th>
                        <th>{self.t['avg_iops_label']}</th>
                        <th>{self.t['peak_iops_label']}</th>
                        <th>{self.t['avg_throughput_label']}</th>
                        <th>{self.t['data_completeness_label']}</th>
                    </tr>
                </thead>
                <tbody>
            """

            # Monitoring component data (estimated based on overall monitoring data)
            # Calculate unified completeness
            data_completeness = self._calculate_data_completeness()

            monitoring_components = [
                {
                    'name': self.t['iostat_monitoring'],
                    'cpu_avg': self.overhead_data.get('monitoring_cpu_percent_avg', 0) * 0.3,
                    'cpu_max': self.overhead_data.get('monitoring_cpu_percent_max', 0) * 0.4,
                    'mem_avg': self.overhead_data.get('monitoring_memory_mb_avg', 0) * 0.2,
                    'mem_max': self.overhead_data.get('monitoring_memory_mb_max', 0) * 0.3,
                    'iops_avg': self.overhead_data.get('monitoring_iops_avg', 0) * 0.4,
                    'iops_max': self.overhead_data.get('monitoring_iops_max', 0) * 0.5,
                    'throughput_avg': self.overhead_data.get('monitoring_throughput_mibs_avg', 0) * 0.3,
                    'completeness': data_completeness
                },
                {
                    'name': self.t['sar_monitoring'],
                    'cpu_avg': self.overhead_data.get('monitoring_cpu_percent_avg', 0) * 0.2,
                    'cpu_max': self.overhead_data.get('monitoring_cpu_percent_max', 0) * 0.3,
                    'mem_avg': self.overhead_data.get('monitoring_memory_mb_avg', 0) * 0.15,
                    'mem_max': self.overhead_data.get('monitoring_memory_mb_max', 0) * 0.2,
                    'iops_avg': self.overhead_data.get('monitoring_iops_avg', 0) * 0.2,
                    'iops_max': self.overhead_data.get('monitoring_iops_max', 0) * 0.3,
                    'throughput_avg': self.overhead_data.get('monitoring_throughput_mibs_avg', 0) * 0.2,
                    'completeness': data_completeness
                },
                {
                    'name': self.t['vmstat_monitoring'],
                    'cpu_avg': self.overhead_data.get('monitoring_cpu_percent_avg', 0) * 0.1,
                    'cpu_max': self.overhead_data.get('monitoring_cpu_percent_max', 0) * 0.15,
                    'mem_avg': self.overhead_data.get('monitoring_memory_mb_avg', 0) * 0.1,
                    'mem_max': self.overhead_data.get('monitoring_memory_mb_max', 0) * 0.15,
                    'iops_avg': self.overhead_data.get('monitoring_iops_avg', 0) * 0.1,
                    'iops_max': self.overhead_data.get('monitoring_iops_max', 0) * 0.15,
                    'throughput_avg': self.overhead_data.get('monitoring_throughput_mibs_avg', 0) * 0.1,
                    'completeness': data_completeness
                },
                {
                    'name': self.t['data_collection_script'],
                    'cpu_avg': self.overhead_data.get('monitoring_cpu_percent_avg', 0) * 0.3,
                    'cpu_max': self.overhead_data.get('monitoring_cpu_percent_max', 0) * 0.4,
                    'mem_avg': self.overhead_data.get('monitoring_memory_mb_avg', 0) * 0.4,
                    'mem_max': self.overhead_data.get('monitoring_memory_mb_max', 0) * 0.5,
                    'iops_avg': self.overhead_data.get('monitoring_iops_avg', 0) * 0.2,
                    'iops_max': self.overhead_data.get('monitoring_iops_max', 0) * 0.3,
                    'throughput_avg': self.overhead_data.get('monitoring_throughput_mibs_avg', 0) * 0.3,
                    'completeness': data_completeness
                },
                {
                    'name': self.t['total_monitoring_overhead'],
                    'cpu_avg': self.overhead_data.get('monitoring_cpu_percent_avg', 0),
                    'cpu_max': self.overhead_data.get('monitoring_cpu_percent_max', 0),
                    'mem_avg': self.overhead_data.get('monitoring_memory_mb_avg', 0),
                    'mem_max': self.overhead_data.get('monitoring_memory_mb_max', 0),
                    'iops_avg': self.overhead_data.get('monitoring_iops_avg', 0),
                    'iops_max': self.overhead_data.get('monitoring_iops_max', 0),
                    'throughput_avg': self.overhead_data.get('monitoring_throughput_mibs_avg', 0),
                    'completeness': data_completeness
                }
            ]

            for i, component in enumerate(monitoring_components):
                # Set style based on whether it's a total row
                if component['name'] == self.t['total_monitoring_overhead']:
                    row_class = ' class="total-row"'
                else:
                    row_class = ''

                # Data completeness color
                completeness_color = 'green' if component['completeness'] > 95 else 'orange' if component['completeness'] > 90 else 'red'

                table_html += f"""
                <tr{row_class}>
                    <td>{component['name']}</td>
                    <td class="numeric-cell">{component['cpu_avg']:.2f}%</td>
                    <td class="numeric-cell">{component['cpu_max']:.2f}%</td>
                    <td class="numeric-cell">{component['mem_avg']:.1f} MB</td>
                    <td class="numeric-cell">{component['mem_max']:.1f} MB</td>
                    <td class="numeric-cell">{self._format_monitoring_io(component['iops_avg'], 'iops')}</td>
                    <td class="numeric-cell">{self._format_monitoring_io(component['iops_max'], 'iops')}</td>
                    <td class="numeric-cell">{self._format_monitoring_io(component['throughput_avg'], 'throughput')} MiB/s</td>
                    <td class="numeric-cell completeness-{completeness_color}">{component['completeness']:.1f}%</td>
                </tr>
                """

            table_html += f"""
                </tbody>
            </table>

            <div class="info" style="margin-top: 15px;">
                <h4>&#128202; {self.t['overhead_analysis_notes']}</h4>
                <ul>
                    <li><strong>{self.t['monitoring_component_label']}</strong>: {self.t['component_breakdown']}</li>
                    <li><strong>CPU Usage</strong>: {self.t['cpu_percentage_used']}</li>
                    <li><strong>{self.t['memory_usage_label']}</strong>: {self.t['memory_size_used']}</li>
                    <li><strong>IOPS</strong>: {self.t['disk_io_operations']}</li>
                    <li><strong>Throughput</strong>: {self.t['disk_throughput_generated']}</li>
                    <li><strong>{self.t['data_completeness_label']}</strong>: {self.t['data_completeness_percentage']}</li>
                </ul>
                <p>{self.t['total_overhead_usually']}</p>
                <p><strong>{self.t['iops_throughput_zero_reason']}</strong>:</p>
                <ul style="margin-top: 5px;">
                    <li>{self.t['monitoring_reads_proc']}</li>
                    <li>{self.t['actual_io_overhead']}</li>
                    <li>{self.t['proves_efficient_design']}</li>
                    <li>{self.t['to_view_tiny_values']}</li>
                </ul>
            </div>
            """

            return table_html

        except Exception as e:
            print(f"❌ {self.t['overhead_table_generation_failed']}: {e}")
            return f"""
            <div class="warning">
                <h4>❌ {self.t['overhead_table_generation_failed']}</h4>
                <p>{self.t['error_message_label']}: {str(e)[:100]}</p>
                <p>{self.t['check_data_format']}</p>
            </div>
            """



    def _generate_independent_tools_results(self):
        """Generate independent analysis tools results display"""
        return f"""
        <div class="info-grid">
            <div class="info-card">
                <h4>&#128269; {self.t['disk_bottleneck_detection']}</h4>
                <p><strong>{self.t['report_file_label']}</strong>: disk_bottleneck_analysis.txt</p>
                <p>{self.t['analyze_disk_under_qps']}</p>
            </div>
            <div class="info-card">
                <h4>&#128260; {self.t['disk_iops_conversion']}</h4>
                <p><strong>{self.t['report_file_label']}</strong>: disk_iops_conversion.json</p>
                <p>{self.t['convert_iostat_to_aws']}</p>
            </div>
            <div class="info-card">
                <h4>&#128202; {self.t['disk_comprehensive_analysis']}</h4>
                <p><strong>{self.t['report_file_label']}</strong>: disk_analysis.txt</p>
                <p>{self.t['disk_performance_report']}</p>
            </div>
            <div class="info-card">
                <h4>&#128187; {self.t['monitoring_overhead_calculation']}</h4>
                <p><strong>{self.t['data_file_label']}</strong>: monitoring_overhead_YYYYMMDD_HHMMSS.csv</p>
                <p>{self.t['detailed_overhead_data']}</p>
            </div>
        </div>
        """

    def _generate_ena_warnings_section(self, df):
        """Generate ENA network warning section - using ENAFieldAccessor"""
        try:
            # Check ENA data availability - using configuration-driven approach
            ena_columns = ENAFieldAccessor.get_available_ena_fields(df)
            if not ena_columns:
                return ""

            # Analyze ENA limitation data
            limitations = self._analyze_ena_limitations(df)

            if not limitations:
                return f"""
                <div class="info" style="background-color: #E8F8F5; padding: 15px; border-radius: 6px; margin: 15px 0; border-left: 4px solid #27AE60;">
                    <h4>&#9989; {self.t['ena_network_normal']}</h4>
                    <p>{self.t['no_ena_limitations']}</p>
                </div>
                """

            # Generate warning HTML
            html = f"""
            <div class="warning">
                <h4>&#128680; {self.t['ena_limitation_detected']}</h4>
                <p>{self.t['detected_ena_limitations']}</p>
                <ul>
            """

            for limit in limitations:
                duration = ""
                if limit['first_time'] != limit['last_time']:
                    duration = f" ({self.t['duration_time']}: {limit['first_time']} {self.t['to']} {limit['last_time']})"

                html += f"""
                <li>
                    <strong>{limit['type']}</strong>{duration}
                    <ul>
                        <li>{self.t['occurrence_count']}: {limit['occurrences']}{self.t['times']}</li>
                        <li>{self.t['max_value_label']}: {limit['max_value']}</li>
                        <li>{self.t['cumulative_impact']}: {limit['total_affected']}</li>
                    </ul>
                </li>
                """

            html += f"""
                </ul>
                <p><strong>{self.t['recommendation_label']}</strong>: {self.t['consider_optimize_network']}</p>
            </div>
            """

            return html

        except Exception as e:
            return f'<div class="error">{self.t["ena_warning_generation_failed"]}: {str(e)}</div>'

    def _analyze_ena_limitations(self, df):
        """Analyze ENA limitation occurrences - using ENAFieldAccessor"""
        limitations = []
        available_fields = ENAFieldAccessor.get_available_ena_fields(df)

        # Analyze exceeded type fields
        for field in available_fields:
            if 'exceeded' in field and field in df.columns:
                # Get field analysis information
                field_analysis = ENAFieldAccessor.analyze_ena_field(df, field)
                if field_analysis:
                    # Filter records with limitations (value > 0)
                    limited_records = df[df[field] > 0]

                    if not limited_records.empty:
                        limitations.append({
                            'type': field_analysis['description'],
                            'field': field,
                            'first_time': limited_records['timestamp'].min(),
                            'last_time': limited_records['timestamp'].max(),
                            'occurrences': len(limited_records),
                            'max_value': limited_records[field].max(),
                            'total_affected': limited_records[field].sum()
                        })

        # Special handling: Connection capacity insufficient warning - find available type fields
        available_field = None
        for field in available_fields:
            if 'available' in field and 'conntrack' in field:
                available_field = field
                break

        if available_field and available_field in df.columns:
            # Use dynamic threshold: calculate based on network threshold and data max value
            thresholds = get_visualization_thresholds()
            max_available = df[available_field].max() if not df[available_field].empty else 50000
            # Warning when available capacity is below (100-network threshold)% of max value
            low_connection_threshold = int(max_available * (100 - thresholds['io_warning']) / 100)
            low_connection_records = df[df[available_field] < low_connection_threshold]
            if not low_connection_records.empty:
                limitations.append({
                    'type': self.t['connection_capacity_warning'],
                    'field': available_field,
                    'first_time': low_connection_records['timestamp'].min(),
                    'last_time': low_connection_records['timestamp'].max(),
                    'occurrences': len(low_connection_records),
                    'max_value': f"{self.t['minimum_remaining']} {low_connection_records[available_field].min()} {self.t['connections']}",
                    'total_affected': f"{self.t['average_remaining']} {low_connection_records[available_field].mean():.0f} {self.t['connections']}" if available_field in low_connection_records.columns else self.t['data_not_available']
                })

        return limitations

    def _generate_ena_data_table(self, df):
        """Generate ENA data statistics table - using ENAFieldAccessor"""
        try:
            ena_columns = ENAFieldAccessor.get_available_ena_fields(df)
            if not ena_columns:
                return ""

            # Generate statistics - use ENAFieldAccessor to get field descriptions
            ena_stats = {}

            for col in ena_columns:
                field_analysis = ENAFieldAccessor.analyze_ena_field(df, col)
                if field_analysis:
                    # Select description based on language
                    description = field_analysis.get('aws_description', field_analysis['description']) if self.language == 'en' else field_analysis['description']
                    ena_stats[col] = {
                        'description': description,
                        'max': df[col].max(),
                        'mean': df[col].mean(),
                        'current': df[col].iloc[-1] if len(df) > 0 else 0
                    }

            # Generate HTML table
            table_rows = ""
            for field, stats in ena_stats.items():
                field_analysis = ENAFieldAccessor.analyze_ena_field(df, field)

                # Set different formats for different field types
                if field_analysis and field_analysis['type'] == 'gauge':  # available type fields
                    current_val = f"{stats['current']:,.0f}"
                    max_val = f"{stats['max']:,.0f}"
                    mean_val = f"{stats['mean']:,.0f}"
                else:  # counter type fields (exceeded)
                    current_val = f"{stats['current']}"
                    max_val = f"{stats['max']}"
                    mean_val = f"{stats['mean']:.1f}"

                # Status indicator
                status_class = "normal"
                if field_analysis and field_analysis['type'] == 'counter' and stats['current'] > 0:
                    status_class = "warning"
                elif field_analysis and field_analysis['type'] == 'gauge':
                    # Use dynamic threshold to determine connection capacity status
                    thresholds = get_visualization_thresholds()
                    max_available = max(stats['max'], 50000)  # Use max value or default value
                    warning_threshold = int(max_available * (100 - thresholds['io_warning']) / 100)
                    if stats['current'] < warning_threshold:
                        status_class = "warning"

                table_rows += f"""
                <tr class="{status_class}">
                    <td>{stats['description']}</td>
                    <td>{current_val}</td>
                    <td>{max_val}</td>
                    <td>{mean_val}</td>
                </tr>
                """

            return f"""
            <div class="section">
                <h3>&#127760; {self.t['ena_network_statistics']}</h3>
                <table class="performance-table">
                    <thead>
                        <tr>
                            <th>{self.t['ena_metric']}</th>
                            <th>{self.t['current_value']}</th>
                            <th>{self.t['max_value']}</th>
                            <th>{self.t['avg_value']}</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
                <p class="table-note">
                    <strong>{self.t['note_label']}</strong>:
                    • {self.t['exceeded_fields_show_drops']}
                    • {self.t['available_connections_show_capacity']}
                </p>
            </div>
            """

        except Exception as e:
            return f'<div class="error">{self.t["ena_table_generation_failed"]}: {str(e)}</div>'

    def _generate_cpu_disk_correlation_table(self, df):
        """Improved CPU and Disk correlation analysis table generation"""
        key_correlations = [
            ('cpu_iowait', 'util', self.t['cpu_iowait_vs_util']),
            ('cpu_iowait', 'aqu_sz', self.t['cpu_iowait_vs_queue']),
            ('cpu_iowait', 'r_await', self.t['cpu_iowait_vs_read_latency']),
            ('cpu_iowait', 'w_await', self.t['cpu_iowait_vs_write_latency']),
            ('cpu_usr', 'r_s', self.t['user_cpu_vs_read_requests']),
            ('cpu_sys', 'w_s', self.t['system_cpu_vs_write_requests']),
        ]

        correlation_data = []
        data_cols = [col for col in df.columns if col.startswith('data_')]
        accounts_cols = [col for col in df.columns if col.startswith('accounts_')]

        # Safe correlation analysis function
        def safe_correlation_analysis(cpu_col, iostat_col, description, device_type):
            """Safe correlation analysis"""
            try:
                if cpu_col not in df.columns:
                    return None, f"{self.t['missing_cpu_field']}: {cpu_col}"

                if iostat_col not in df.columns:
                    return None, f"{self.t['missing_disk_field']}: {iostat_col}"

                # Data validity check
                cpu_data = df[cpu_col].dropna()
                disk_data = df[iostat_col].dropna()

                if len(cpu_data) == 0 or len(disk_data) == 0:
                    return None, self.t['data_is_empty']

                # Align data and remove NaN
                combined_data = pd.concat([df[cpu_col], df[iostat_col]], axis=1).dropna()
                if len(combined_data) < 10:
                    return None, f"{self.t['insufficient_data_points']} ({self.t['only']}{len(combined_data)}{self.t['items']})"

                x_clean = combined_data.iloc[:, 0]
                y_clean = combined_data.iloc[:, 1]

                # Calculate correlation
                corr, p_value = pearsonr(x_clean, y_clean)

                # Check result validity
                if np.isnan(corr) or np.isnan(p_value):
                    return None, self.t['correlation_result_nan']

                # Improved correlation strength classification
                abs_corr = abs(corr)
                if abs_corr >= 0.8:
                    strength = self.t['very_strong_correlation']
                elif abs_corr >= 0.6:
                    strength = self.t['strong_correlation']
                elif abs_corr >= 0.4:
                    strength = self.t['moderate_correlation']
                elif abs_corr >= 0.2:
                    strength = self.t['weak_correlation']
                else:
                    strength = self.t['very_weak_correlation']

                # Improved statistical significance determination
                if p_value < 0.001:
                    significant = self.t['highly_significant_3']
                elif p_value < 0.01:
                    significant = self.t['highly_significant_2']
                elif p_value < 0.05:
                    significant = self.t['significant_1']
                else:
                    significant = self.t['not_significant']

                return {
                    self.t['device_type']: device_type,
                    self.t['analysis_item']: description,
                    self.t['cpu_metric']: cpu_col,
                    self.t['disk_metric']: iostat_col,
                    self.t['correlation_coefficient']: f"{corr:.4f}",
                    self.t['p_value']: f"{p_value:.4f}",
                    self.t['statistical_significance']: significant,
                    self.t['correlation_strength']: strength,
                    self.t['valid_sample_count']: len(combined_data),
                    self.t['data_integrity']: f"{len(combined_data)/len(df)*100:.1f}%"
                }, None

            except Exception as e:
                return None, f"{self.t['analysis_failed']}: {str(e)[:50]}"

        def find_matching_column(target_field, column_list):
            """Precise field matching"""
            # Exact match
            exact_matches = [col for col in column_list if col.endswith(f'_{target_field}')]
            if exact_matches:
                return exact_matches[0]

            # Fuzzy matching (stricter)
            fuzzy_matches = [col for col in column_list if target_field in col and not any(x in col for x in ['avg', 'max', 'min', 'sum'])]
            if fuzzy_matches:
                return fuzzy_matches[0]

            return None

        # Analyze DATA Device
        for cpu_field, iostat_field, description in key_correlations:
            iostat_col = find_matching_column(iostat_field, data_cols)

            if iostat_col:
                result, error = safe_correlation_analysis(cpu_field, iostat_col, description, 'DATA')
                if result:
                    correlation_data.append(result)
                else:
                    print(f"⚠️ DATA Device {description}: {error}")

        # Analyze ACCOUNTS Device
        if accounts_cols:
            for cpu_field, iostat_field, description in key_correlations:
                iostat_col = find_matching_column(iostat_field, accounts_cols)

                if iostat_col:
                    result, error = safe_correlation_analysis(cpu_field, iostat_col, description.replace('Device', 'ACCOUNTS Device'), 'ACCOUNTS')
                    if result:
                        correlation_data.append(result)
                    else:
                        print(f"⚠️ ACCOUNTS Device {description}: {error}")

        if not correlation_data:
            return f"""
            <div class="warning">
                <h4>&#9888; {self.t['correlation_data_unavailable']}</h4>
                <p>{self.t['possible_reasons']}</p>
                <ul>
                    <li>{self.t['missing_cpu_disk_fields']}</li>
                    <li>{self.t['data_quality_issues']}</li>
                    <li>{self.t['insufficient_data_less_10']}</li>
                </ul>
            </div>
            """

        # Generate improved HTML table
        table_html = f"""
        <table class="report-table correlation-table">
            <thead>
                <tr>
                    <th>{self.t['device_type']}</th>
                    <th>{self.t['analysis_item']}</th>
                    <th>{self.t['correlation_coefficient']}</th>
                    <th>{self.t['p_value']}</th>
                    <th>{self.t['statistical_significance']}</th>
                    <th>{self.t['correlation_strength']}</th>
                    <th>{self.t['valid_sample_count']}</th>
                    <th>{self.t['data_integrity']}</th>
                </tr>
            </thead>
            <tbody>
        """

        for i, data in enumerate(correlation_data):
            # Set row color based on correlation strength
            strength_val = data[self.t['correlation_strength']]
            if self.t['very_strong_correlation'] in strength_val:
                row_class = "correlation-very-strong"
            elif self.t['strong_correlation'] in strength_val:
                row_class = "correlation-strong"
            elif self.t['moderate_correlation'] in strength_val:
                row_class = "correlation-moderate"
            else:
                row_class = "correlation-weak"

            table_html += f"""
                <tr class="{row_class}">
                    <td>{data[self.t['device_type']]}</td>
                    <td>{data[self.t['analysis_item']]}</td>
                    <td class="metric-value">{data[self.t['correlation_coefficient']]}</td>
                    <td>{data[self.t['p_value']]}</td>
                    <td>{data[self.t['statistical_significance']]}</td>
                    <td class="metric-value">{data[self.t['correlation_strength']]}</td>
                    <td class="numeric-cell">{data[self.t['valid_sample_count']]}</td>
                    <td>{data[self.t['data_integrity']]}</td>
                </tr>
            """

        table_html += f"""
            </tbody>
        </table>
        <div class="info" style="margin-top: 15px;">
            <h4>&#128202; {self.t['correlation_analysis_notes']}</h4>
            <ul>
                <li><strong>{self.t['correlation_range']}</strong>: {self.t['larger_abs_stronger']}</li>
                <li><strong>{self.t['statistical_significance']}</strong>: {self.t['significance_levels']}</li>
                <li><strong>{self.t['strength_classification']}</strong></li>
                <li><strong>{self.t['data_integrity']}</strong>: {self.t['data_integrity_percentage']}</li>
            </ul>
        </div>
        """

        return table_html

    def _format_block_height_value(self, field_name, value):
        """Convert block_height related field values to human-readable format"""
        if 'health' in field_name.lower():
            return 'Healthy' if value == 1 else 'Unhealthy'
        elif 'data_loss' in field_name.lower():
            return 'No Data Loss' if value == 0 else 'Data Loss Detected'
        else:
            # For numeric fields (like block_height, block_height_diff), keep as is
            return f"{value:.0f}" if isinstance(value, (int, float)) else str(value)

    def _analyze_block_height_performance(self, df, block_height_fields):
        """Enhanced block height performance analysis - using comparison table display"""
        if not block_height_fields or df.empty:
            return f"<div class='info-card'><h4>{self.t['block_height_monitoring']}</h4><p>{self.t['no_block_height_data']}</p></div>"

        try:
            # Add time series chart display
            sync_chart_html = self._generate_block_height_chart_section()

            # Add data_loss_stats.json file display
            stats_file_html = self._generate_data_loss_stats_section()

            # Collect block height data
            block_height_data = {}
            for field in block_height_fields:
                if field in df.columns:
                    numeric_data = pd.to_numeric(df[field], errors='coerce').dropna()
                    if not numeric_data.empty:
                        block_height_data[field] = {
                            'current': numeric_data.iloc[-1] if len(numeric_data) > 0 else 0,
                            'average': numeric_data.mean(),
                            'min': numeric_data.min(),
                            'max': numeric_data.max(),
                            'data': numeric_data
                        }

            # Generate comparison table
            current_sync_mode = df['sync_mode'].dropna().iloc[-1] if 'sync_mode' in df.columns and not df['sync_mode'].dropna().empty else 'N/A'
            current_sync_status = df['sync_status'].dropna().iloc[-1] if 'sync_status' in df.columns and not df['sync_status'].dropna().empty else 'N/A'
            current_lag_value = df['lag_value'].dropna().iloc[-1] if 'lag_value' in df.columns and not df['lag_value'].dropna().empty else 'N/A'
            current_lag_unit = df['lag_unit'].dropna().iloc[-1] if 'lag_unit' in df.columns and not df['lag_unit'].dropna().empty else ''
            lag_display = f"{current_lag_value} {current_lag_unit}".strip()

            comparison_table = f"""
            <div class="sync-comparison-panel">
                <h3>📊 {self.t['block_height_data_comparison']}</h3>
                <div class="stats-grid sync-stats-grid">
                    <div class="stat-card"><div class="stat-label">{self.t['sync_mode']}</div><div class="stat-value">{current_sync_mode}</div></div>
                    <div class="stat-card"><div class="stat-label">{self.t['sync_status']}</div><div class="stat-value">{current_sync_status}</div></div>
                    <div class="stat-card"><div class="stat-label">{self.t['lag_value']}</div><div class="stat-value">{lag_display}</div></div>
                </div>
                <table class="report-table sync-health-table">
                    <thead>
                        <tr>
                            <th>{self.t['metric']}</th>
                            <th>{self.t['local_block_height']}</th>
                            <th>{self.t['mainnet_block_height']}</th>
                            <th>{self.t['block_height_diff']}</th>
                        </tr>
                    </thead>
                    <tbody>
            """

            # Add data rows
            metrics = [
                ('Current', 'current'),
                ('Average', 'average'),
                ('Min', 'min'),
                ('Max', 'max')
            ]

            for metric_name, metric_key in metrics:
                local_val = block_height_data.get('local_block_height', {}).get(metric_key, 'N/A')
                mainnet_val = block_height_data.get('mainnet_block_height', {}).get(metric_key, 'N/A')
                diff_val = block_height_data.get('block_height_diff', {}).get(metric_key, 'N/A')

                # Format numeric values
                local_str = f"{local_val:.0f}" if isinstance(local_val, (int, float)) else str(local_val)
                mainnet_str = f"{mainnet_val:.0f}" if isinstance(mainnet_val, (int, float)) else str(mainnet_val)
                diff_str = f"{diff_val:.0f}" if isinstance(diff_val, (int, float)) else str(diff_val)

                comparison_table += f"""
                        <tr>
                            <td class="metric-value">{metric_name}</td>
                            <td class="numeric-cell">{local_str}</td>
                            <td class="numeric-cell">{mainnet_str}</td>
                            <td class="numeric-cell metric-value">{diff_str}</td>
                        </tr>
                """

            comparison_table += """
                    </tbody>
                </table>
            </div>
            """

            # Combine all parts
            complete_html = f"""
            <div class="section">
                <h2>🔗 {self.t['blockchain_node_sync_analysis']}</h2>
                {sync_chart_html}
                {comparison_table}
                {stats_file_html}
            </div>
            """

            return complete_html

        except Exception as e:
            return f"<div class='error'>{self.t['block_height_analysis_failed']}: {str(e)}</div>"

    def _generate_block_height_chart_section(self):
        """Generate block height chart display section"""
        # Check multiple possible chart locations
        possible_paths = [
            os.path.join(self.output_dir, 'block_height_sync_chart.png'),
            os.path.join(os.path.dirname(self.output_dir), 'reports', 'block_height_sync_chart.png'),
            os.path.join(self.output_dir, 'current', 'reports', 'block_height_sync_chart.png')
        ]

        chart_src = None
        for path in possible_paths:
            if os.path.exists(path):
                # Calculate relative path
                chart_src = os.path.relpath(path, self.output_dir)
                break

        if chart_src:
            if self.language == 'zh':
                signal_items = """
                        <li><strong>主信号</strong>: 根据链模板的 sync_health.mode 展示高度差、reported lag 或健康状态</li>
                        <li><strong>同步状态</strong>: 当无法获取数值高度差时，显示 healthy / warning / unhealthy 状态时间线</li>
                        <li><strong>阈值</strong>: 数值型 gap/lag 使用 BLOCK_HEIGHT_DIFF_THRESHOLD；持续异常使用 BLOCK_HEIGHT_TIME_THRESHOLD</li>
                        <li><strong>异常区间</strong>: data_loss 或持续不健康状态会在图中标注</li>
                """
            else:
                signal_items = """
                        <li><strong>Primary signal</strong>: Shows height gap, reported lag, or health state based on sync_health.mode from the chain template</li>
                        <li><strong>Sync status</strong>: Shows healthy / warning / unhealthy timeline when numeric height gap is unavailable</li>
                        <li><strong>Thresholds</strong>: Numeric gap/lag uses BLOCK_HEIGHT_DIFF_THRESHOLD; sustained unhealthy state uses BLOCK_HEIGHT_TIME_THRESHOLD</li>
                        <li><strong>Anomaly regions</strong>: data_loss or sustained unhealthy state is highlighted when present</li>
                """
            return f"""
            <div class="info-card">
                <h3>📊 {self.t['block_height_sync_time_series']}</h3>
                <div class="chart-container">
                    <img src="{chart_src}" alt="{self.t['block_height_sync_status']}" class="chart-image">
                </div>
                <div class="chart-info">
                    <p>{self.t['chart_shows_sync_health']}:</p>
                    <ul>
                        {signal_items}
                    </ul>
                </div>
            </div>
            """
        else:
            return f"""
            <div class="info-card">
                <h3>📊 {self.t['block_height_sync_time_series']}</h3>
                <div class="warning">
                    <p>⚠️ {self.t['block_height_chart_not_generated']}</p>
                    <p>{self.t['possible_reason_node_data_unavailable']}</p>
                </div>
            </div>
            """

    def _generate_data_loss_stats_section(self):
        """Generate data_loss_stats.json file display section"""

        # Check stats file in archive
        stats_file = None
        possible_paths = [
            os.path.join(self.output_dir, 'stats', 'data_loss_stats.json'),
            os.path.join(self.output_dir, 'data_loss_stats.json'),
            os.path.join(os.path.dirname(self.output_dir), 'stats', 'data_loss_stats.json')
        ]

        for path in possible_paths:
            if os.path.exists(path):
                stats_file = path
                break

        if stats_file:
            try:
                with open(stats_file, 'r') as f:
                    stats_data = json.load(f)

                # Calculate derived metrics
                avg_duration = (stats_data['total_duration'] / stats_data['data_loss_periods']) if stats_data['data_loss_periods'] > 0 else 0

                return f"""
                <div class="info-card">
                    <h3>📋 {self.t['data_loss_stats_summary']}</h3>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <div class="stat-value">{stats_data['data_loss_count']}</div>
                            <div class="stat-label">{self.t['anomaly_sample_count']}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">{stats_data['data_loss_periods']}</div>
                            <div class="stat-label">{self.t['anomaly_event_count']}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">{stats_data['total_duration']}s</div>
                            <div class="stat-label">{self.t['total_anomaly_duration']}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">{avg_duration:.1f}s</div>
                            <div class="stat-label">{self.t['avg_event_duration']}</div>
                        </div>
                    </div>
                    <div class="file-info">
                        <p><strong>📁 {self.t['stats_file_location']}:</strong> <code>{os.path.relpath(stats_file, self.output_dir)}</code></p>
                        <p><strong>🕐 {self.t['last_updated']}:</strong> {stats_data.get('last_updated', 'Unknown')}</p>
                    </div>
                </div>
                """
            except Exception as e:
                return f"""
                <div class="warning">
                    <h3>⚠️ {self.t['data_loss_statistics']}</h3>
                    <p>{self.t['stats_file_read_failed']}: {str(e)}</p>
                    <p><strong>{self.t['file_location']}:</strong> <code>{os.path.relpath(stats_file, self.output_dir)}</code></p>
                </div>
                """
        else:
            return f"""
            <div class="warning">
                <h3>⚠️ {self.t['data_loss_statistics']}</h3>
                <p>{self.t['data_loss_stats_file_not_found']}:</p>
                <ul>
                    <li>{self.t['no_data_loss_detected_during_test']}</li>
                    <li>{self.t['stats_file_not_archived']}</li>
                    <li>{self.t['block_height_monitor_not_running']}</li>
                </ul>
            </div>
            """

    def _discover_chart_files(self):
        """Dynamically discover all generated chart files - scan multiple directories, support archive paths"""
        chart_patterns = ["*.png", "*.jpg", "*.svg"]
        chart_files = []

        # Scan directory list - support archived path structure
        scan_dirs = [
            self.output_dir,  # Main output directory (may be archive directory)
            os.path.join(self.output_dir, 'current', 'reports'),  # Advanced charts directory
            os.path.join(self.output_dir, 'reports'),  # Backup reports directory
            os.path.join(self.output_dir, 'logs'),  # Archived logs directory
        ]

        # If output_dir looks like archive directory, add special scan paths
        if 'archives' in self.output_dir or 'run_' in os.path.basename(self.output_dir):
            # This is archive directory, scan its subdirectories directly
            scan_dirs.extend([
                os.path.join(self.output_dir, 'logs'),
                os.path.join(self.output_dir, 'reports'),
                os.path.join(self.output_dir, 'current', 'reports'),
            ])

        # Add sibling reports directory scan (critical fix)
        parent_dir = os.path.dirname(self.output_dir)
        sibling_reports = os.path.join(parent_dir, 'reports')
        if os.path.exists(sibling_reports):
            scan_dirs.append(sibling_reports)

        for scan_dir in scan_dirs:
            if os.path.exists(scan_dir):
                for pattern in chart_patterns:
                    chart_files.extend(glob.glob(os.path.join(scan_dir, pattern)))

        # Deduplicate and sort
        unique_charts = list(set(chart_files))

        # Filter out timestamped duplicates (e.g., keep xxx.png, remove xxx_20251024_104814.png)
        import re
        timestamp_pattern = re.compile(r'_\d{8}_\d{6}\.png$')
        filtered_charts = []
        base_names = {}

        for chart in unique_charts:
            basename = os.path.basename(chart)
            # Check if this is a timestamped version
            if timestamp_pattern.search(basename):
                # Extract base name without timestamp
                base_name = re.sub(r'_\d{8}_\d{6}', '', basename)
                base_names[base_name] = base_names.get(base_name, []) + [chart]
            else:
                # Non-timestamped version, always keep
                filtered_charts.append(chart)

        # For timestamped files, only add if no non-timestamped version exists
        for base_name, charts in base_names.items():
            if base_name not in [os.path.basename(f) for f in filtered_charts]:
                # No non-timestamped version, keep the timestamped one
                filtered_charts.extend(charts)

        return sorted([f for f in filtered_charts if os.path.exists(f)])

    def _categorize_charts(self, chart_files):
        """Organize charts by category - based on filename patterns, exclude duplicate displayed charts"""
        # Exclude charts already displayed in fixed sections
        excluded_charts = {
            'block_height_sync_chart.png',  # Already displayed in block height analysis section
            'monitoring_overhead_analysis.png',  # Already displayed in monitoring overhead detailed analysis section
            'monitoring_impact_chart.png',  # Already displayed in monitoring overhead and performance relationship section
            'resource_distribution_chart.png'  # Already displayed in resource proportion analysis section
        }

        categories = {
            'advanced': {'title': self.t.get('advanced_analysis_charts', 'Advanced Analysis Charts'), 'charts': []},
            'disk': {'title': self.t.get('disk_professional_charts', 'Disk Professional Charts'), 'charts': []},
            'performance': {'title': self.t.get('core_performance_charts', 'Core Performance Charts'), 'charts': []},
            'monitoring': {'title': self.t.get('monitoring_overhead_charts', 'Monitoring & Overhead Charts'), 'charts': []},
            'network': {'title': self.t.get('network_ena_charts', 'Provider Network Charts'), 'charts': []},
            'other': {'title': self.t.get('additional_charts', 'Additional Charts'), 'charts': []}
        }

        for chart_file in chart_files:
            filename = os.path.basename(chart_file)
            filename_lower = filename.lower()

            # Skip excluded charts
            if filename in excluded_charts:
                continue

            # Advanced analysis charts
            if any(keyword in filename_lower for keyword in ['pearson', 'correlation', 'regression', 'heatmap', 'matrix']):
                categories['advanced']['charts'].append(chart_file)
            # Disk charts
            elif any(keyword in filename_lower for keyword in ['disk', 'iostat', 'bottleneck']):
                categories['disk']['charts'].append(chart_file)
            # Network/ENA charts
            elif any(keyword in filename_lower for keyword in ['ena', 'network', 'allowance']):
                categories['network']['charts'].append(chart_file)
            # Monitoring charts (exclude already displayed)
            elif any(keyword in filename_lower for keyword in ['monitoring', 'overhead']) and filename not in excluded_charts:
                categories['monitoring']['charts'].append(chart_file)
            # Performance charts
            elif any(keyword in filename_lower for keyword in ['performance', 'qps', 'trend', 'efficiency', 'threshold', 'util', 'await']):
                categories['performance']['charts'].append(chart_file)
            else:
                categories['other']['charts'].append(chart_file)

        return categories

    def _generate_chart_gallery_section(self):
        """Generate dynamic chart display area"""
        chart_files = self._discover_chart_files()
        if not chart_files:
            return f'<div class="section"><h2>📊 {self.t["performance_charts"]}</h2><p>{self.t["no_charts_found"]}</p></div>'

        categories = self._categorize_charts(chart_files)
        total_charts = len(chart_files)

        html = f'''
        <div class="section">
            <h2>📊 {self.t["performance_chart_gallery"]}</h2>
            <div class="chart-summary">
                <p><strong>{self.t["total_charts_generated"]}:</strong> {total_charts}</p>
            </div>
        '''

        for category_key, category_data in categories.items():
            if category_data['charts']:
                html += f'''
                <div class="chart-category">
                    <h3 style="color: #2C3E50;">📈 {category_data['title']} ({len(category_data['charts'])} charts)</h3>
                    <div class="chart-grid">
                '''

                for chart_file in category_data['charts']:
                    chart_name = os.path.basename(chart_file)
                    chart_key = chart_name.replace('.png', '')

                    # Get localized title and description
                    title_key = f'chart_{chart_key}'
                    desc_key = f'chart_{chart_key}_desc'
                    chart_title = self.t.get(title_key, chart_name.replace('_', ' ').replace('.png', '').title())
                    chart_desc = self.t.get(desc_key, '')

                    # Fix capitalization if using fallback title
                    if title_key not in self.t:
                        chart_title = chart_title.replace('Cpu', 'CPU').replace('Aws', 'AWS').replace('Qps', 'QPS').replace('Ena', 'ENA')

                    # Calculate relative path
                    rel_path = os.path.relpath(chart_file, self.output_dir)

                    html += f'''
                    <div class="chart-item">
                        <h4>{chart_title}</h4>
                        {f'<p style="color: #666; font-size: 0.9em; margin: 8px 0;">{chart_desc}</p>' if chart_desc else ''}
                        <div class="chart-container">
                            <img src="{rel_path}" alt="{chart_title}" class="chart">
                        </div>
                    </div>
                    '''

                html += '''
                    </div>
                </div>
                '''

        html += '</div>'
        return html

    def _generate_html_content(self, df):
        """Generate HTML content + bottleneck information display + image references"""
        try:
            # Identify block_height related fields
            block_height_fields = [col for col in df.columns if 'block_height' in col.lower() or 'height' in col.lower()]

            # Generate each section - using actually existing methods
            monitoring_overhead_analysis = self._generate_monitoring_overhead_section()
            monitoring_overhead_detailed = self._generate_monitoring_overhead_detailed_section()
            ena_warnings = self._generate_ena_warnings_section(df)
            ena_data_table = self._generate_ena_data_table(df)

            config_status_section = self._generate_config_status_section()
            block_height_analysis = self._analyze_block_height_performance(df, block_height_fields)

            correlation_table = self._generate_cpu_disk_correlation_table(df)
            overhead_table = self._generate_overhead_data_table()

            # Generate performance summary
            performance_summary = self._generate_performance_summary(df)
            environment_summary = self._generate_environment_summary_section(df)
            data_quality_summary = self._generate_data_quality_section(df)

            # Generate bottleneck information display (if available)
            bottleneck_section = self._generate_bottleneck_section()

            # Generate dynamic chart display section
            charts_section = self._generate_chart_gallery_section()

            # Generate Disk analysis results
            disk_warnings, disk_metrics = self.parse_disk_analyzer_log()
            disk_analysis_section = self.generate_disk_analysis_section(disk_warnings, disk_metrics)

            # Per-method attribution section (optional; empty if proxy data is absent)
            per_method_section = self._generate_per_method_section_safe()
            correlation_section = f"""
            <div class="section" id="cpu-disk-correlation">
                <h2>&#128202; {self.t['cpu_disk_correlation_analysis']}</h2>
                {correlation_table}
            </div>
            """

            section_specs = [
                ('run-environment', self.t['run_environment'], environment_summary),
                ('data-quality', self.t['data_quality_summary'], data_quality_summary),
                ('system-bottleneck', self.t['system_bottleneck_analysis'], bottleneck_section),
                ('performance-summary', self.t['performance_summary'], performance_summary),
                ('configuration', self.t['config_status_check'], config_status_section),
                ('sync-health', self.t['blockchain_node_sync_analysis'], block_height_analysis),
                ('disk-analysis', self.t['disk_performance_analysis'], disk_analysis_section),
                ('charts', self.t['performance_analysis_charts'], charts_section),
                ('monitoring-overhead', self.t['monitoring_overhead_comprehensive_analysis'], monitoring_overhead_analysis),
                ('monitoring-overhead-detail', self.t['monitoring_overhead_detailed'], monitoring_overhead_detailed),
                ('overhead-table', self.t['monitoring_overhead_breakdown'], overhead_table),
                ('ena-warnings', self.t['ena_network_statistics'], ena_warnings),
                ('ena-data', self.t['ena_network_statistics'], ena_data_table),
                ('per-method', 'Per-Method', per_method_section),
                ('cpu-disk-correlation', self.t['cpu_disk_correlation_analysis'], correlation_section),
            ]

            section_html = []
            nav_items = []
            for section_id, label, content in section_specs:
                available = bool(content and str(content).strip())
                nav_items.append((section_id, label, available))
                if available:
                    section_html.append(self._add_section_id(content, section_id))

            report_nav = self._generate_report_nav(nav_items)

            return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>&#128640; {self.t['report_title']}</title>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <style>
                    {self._get_css_styles()}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="report-header">
                        <h1>{self.t['report_title']}</h1>
                        <div class="report-meta">{self.t['generated_time']}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
                        <div class="report-capabilities">
                            <span>{self.t['unified_field_naming']}</span>
                            <span>{self.t['complete_device_support']}</span>
	                            <span>{self.t['monitoring_overhead_analysis']}</span>
	                            <span>{self.t['blockchain_node_specific_analysis']}</span>
	                            <span>{self.t['bottleneck_detection_analysis']}</span>
	                        </div>
                        {report_nav}
	                    </div>

                    {''.join(section_html)}
                </div>
            </body>
            </html>
            """
        except Exception as e:
            return f"<div class='error'>{self.t['html_content_generation_failed']}: {str(e)}</div>"

    def _generate_charts_section(self):
        """Generate chart display section"""
        try:
            # Define all possible generated charts and their descriptions
            chart_definitions = [
                # Charts generated by performance_visualizer.py
                {
                    'filename': 'performance_overview.png',
                    'title': f'&#128200; {self.t["chart_performance_overview"]}',
                    'description': self.t['chart_performance_overview_desc']
                },
                {
                    'filename': 'cpu_disk_correlation_visualization.png',
                    'title': f'&#128279; {self.t["chart_cpu_disk_correlation"]}',
                    'description': self.t['chart_cpu_disk_correlation_desc']
                },
                {
                    'filename': 'device_performance_comparison.png',
                    'title': f'&#128190; {self.t["chart_device_performance_comparison"]}',
                    'description': self.t['chart_device_performance_comparison_desc']
                },
                {
                    'filename': 'await_threshold_analysis.png',
                    'title': f'&#9202; {self.t["chart_await_threshold_analysis"]}',
                    'description': self.t['chart_await_threshold_analysis_desc']
                },
                {
                    'filename': 'util_threshold_analysis.png',
                    'title': f'&#128202; {self.t["chart_util_threshold_analysis"]}',
                    'description': self.t['chart_util_threshold_analysis_desc']
                },
                {
                    'filename': 'monitoring_overhead_analysis.png',
                    'title': f'&#128203; {self.t["chart_monitoring_overhead_analysis"]}',
                    'description': self.t['chart_monitoring_overhead_analysis_desc']
                },
                {
                    'filename': 'smoothed_trend_analysis.png',
                    'title': f'&#128200; {self.t["chart_smoothed_trend_analysis"]}',
                    'description': self.t['chart_smoothed_trend_analysis_desc']
                },
                {
                    'filename': 'qps_trend_analysis.png',
                    'title': f'&#128640; {self.t["chart_qps_trend_analysis"]}',
                    'description': self.t['chart_qps_trend_analysis_desc']
                },
                {
                    'filename': 'resource_efficiency_analysis.png',
                    'title': f'&#9889; {self.t["chart_resource_efficiency_analysis"]}',
                    'description': self.t['chart_resource_efficiency_analysis_desc']
                },
                {
                    'filename': 'bottleneck_identification.png',
                    'title': f'&#128680; {self.t["chart_bottleneck_identification"]}',
                    'description': self.t['chart_bottleneck_identification_desc']
                },

                # Charts generated by advanced_chart_generator.py
                {
                    'filename': 'pearson_correlation_analysis.png',
                    'title': f'&#128202; {self.t["chart_pearson_correlation_analysis"]}',
                    'description': self.t['chart_pearson_correlation_analysis_desc']
                },
                {
                    'filename': 'linear_regression_analysis.png',
                    'title': f'&#128200; {self.t["chart_linear_regression_analysis"]}',
                    'description': self.t['chart_linear_regression_analysis_desc']
                },
                {
                    'filename': 'negative_correlation_analysis.png',
                    'title': f'&#128201; {self.t["chart_negative_correlation_analysis"]}',
                    'description': self.t['chart_negative_correlation_analysis_desc']
                },
                {
                    'filename': 'comprehensive_correlation_matrix.png',
                    'title': f'&#128269; {self.t["chart_comprehensive_correlation_matrix"]}',
                    'description': self.t['chart_comprehensive_correlation_matrix_desc']
                },
                {
                    'filename': 'performance_trend_analysis.png',
                    'title': f'&#128202; {self.t["chart_performance_trend_analysis"]}',
                    'description': self.t['chart_performance_trend_analysis_desc']
                },
                {
                    'filename': 'ena_limitation_trends.png',
                    'title': f'&#128680; {self.t["chart_ena_limitation_trends"]}',
                    'description': self.t['chart_ena_limitation_trends_desc']
                },
                {
                    'filename': 'ena_connection_capacity.png',
                    'title': f'&#128279; {self.t["chart_ena_connection_capacity"]}',
                    'description': self.t['chart_ena_connection_capacity_desc']
                },
                {
                    'filename': 'ena_comprehensive_status.png',
                    'title': f'&#127760; {self.t["chart_ena_comprehensive_status"]}',
                    'description': self.t['chart_ena_comprehensive_status_desc']
                },
                {
                    'filename': 'performance_correlation_heatmap.png',
                    'title': f'&#128293; {self.t["chart_performance_correlation_heatmap"]}',
                    'description': self.t['chart_performance_correlation_heatmap_desc']
                },

                {
                    'filename': 'performance_cliff_analysis.png',
                    'title': f'&#128201; {self.t["chart_performance_cliff_analysis"]}',
                    'description': self.t['chart_performance_cliff_analysis_desc']
                },
                {
                    'filename': 'comprehensive_analysis_charts.png',
                    'title': f'&#128202; {self.t["chart_comprehensive_analysis_charts"]}',
                    'description': self.t['chart_comprehensive_analysis_charts_desc']
                },
                {
                    'filename': 'qps_performance_analysis.png',
                    'title': f'&#127919; {self.t["chart_qps_performance_analysis"]}',
                    'description': self.t['chart_qps_performance_analysis_desc']
                },

                # Disk professional analysis chart group
                {
                    'filename': 'disk_capacity_planning.png',
                    'title': f'&#128202; {self.t["chart_disk_capacity_planning"]}',
                    'description': self.t['chart_disk_capacity_planning_desc']
                },
                {
                    'filename': 'disk_iostat_performance.png',
                    'title': f'&#128190; {self.t["chart_disk_iostat_performance"]}',
                    'description': self.t['chart_disk_iostat_performance_desc']
                },
                {
                    'filename': 'disk_bottleneck_correlation.png',
                    'title': f'&#128279; {self.t["chart_disk_bottleneck_correlation"]}',
                    'description': self.t['chart_disk_bottleneck_correlation_desc']
                },
                {
                    'filename': 'disk_performance_overview.png',
                    'title': f'&#128200; {self.t["chart_disk_performance_overview"]}',
                    'description': self.t['chart_disk_performance_overview_desc']
                },
                {
                    'filename': 'disk_bottleneck_analysis.png',
                    'title': f'&#128680; {self.t["chart_disk_bottleneck_analysis"]}',
                    'description': self.t['chart_disk_bottleneck_analysis_desc']
                },
                {
                    'filename': 'disk_normalized_comparison.png',
                    'title': f'&#9878;️ {self.t["chart_disk_normalized_comparison"]}',
                    'description': self.t['chart_disk_normalized_comparison_desc']
                },
                {
                    'filename': 'disk_time_series_analysis.png',
                    'title': f'&#128202; {self.t["chart_disk_time_series_analysis"]}',
                    'description': self.t['chart_disk_time_series_analysis_desc']
                },
                {
                    'filename': 'block_height_sync_chart.png',
                    'title': f'🔗 {self.t["chart_block_height_sync_chart"]}',
                    'description': self.t['chart_block_height_sync_chart_desc']
                }
            ]

            # Check chart file existence and generate HTML
            charts_html = f"""
            <div class="section">
                <h2>&#128202; {self.t['performance_analysis_charts']}</h2>
                <div class="info">
                    <p>{self.t['charts_provide_comprehensive_visualization']}</p>
                </div>
            """

            # Get report output directory - use environment variable or current/reports structure
            reports_dir = os.getenv('REPORTS_DIR', os.path.join(self.output_dir, 'current', 'reports'))
            if not os.path.exists(reports_dir):
                reports_dir = self.output_dir

            available_charts = []
            missing_charts = []

            for chart in chart_definitions:
                chart_path = os.path.join(reports_dir, chart['filename'])
                # Also check charts directly in output_dir
                alt_chart_path = os.path.join(self.output_dir, os.path.basename(chart['filename']))

                if os.path.exists(chart_path):
                    available_charts.append((chart, chart['filename']))
                elif os.path.exists(alt_chart_path):
                    available_charts.append((chart, os.path.basename(chart['filename'])))
                else:
                    missing_charts.append(chart)

            # Generate HTML for available charts
            if available_charts:
                charts_html += """
                <div class="charts-grid">
                """

                for chart, relative_path in available_charts:
                    charts_html += f"""
                    <div class="chart-item">
                        <h3>{chart['title']}</h3>
                        <div class="chart-description">
                            <p>{chart['description']}</p>
                        </div>
                        <div class="chart-container">
                            <img src="{relative_path}" alt="{chart['title']}" class="chart-image">
                        </div>
                    </div>
                    """

                charts_html += """
                </div>
                """

                # Add chart statistics
                charts_html += f"""
                <div class="charts-summary">
                    <h3>&#128200; {self.t['chart_statistics']}</h3>
                    <ul>
                        <li>&#9989; {self.t['available_charts']}: {len(available_charts)} {self.t['items']}</li>
                        <li>&#8987; {self.t['pending_charts']}: {len(missing_charts)} {self.t['items']}</li>
                        <li>&#128202; {self.t['chart_coverage']}: {len(available_charts)/(len(available_charts)+len(missing_charts))*100:.1f}%</li>
                    </ul>
                </div>
                """
            else:
                charts_html += f"""
                <div class="warning">
                    <h3>&#9888; {self.t['chart_generation_notice']}</h3>
                    <p>{self.t['no_chart_files_found']}:</p>
                    <ul>
                        <li>{self.t['run_performance_visualizer']}</li>
                        <li>{self.t['run_advanced_chart_generator']}</li>
                        <li>{self.t['run_comprehensive_analysis']}</li>
                    </ul>
                    <p>{self.t['ensure_run_chart_scripts_before_report']}</p>
                </div>
                """

            # If there are missing charts, show notice
            if missing_charts:
                charts_html += f"""
                <div class="missing-charts">
                    <h3>&#128203; {self.t['pending_charts']}</h3>
                    <p>{self.t['following_charts_not_generated']}:</p>
                    <ul>
                """
                for chart in missing_charts[:5]:  # Show only first 5
                    charts_html += f"<li>{chart['title']} - {chart['description']}</li>"

                if len(missing_charts) > 5:
                    charts_html += f"<li>... {self.t['and']} {len(missing_charts) - 5} {self.t['more_charts']}</li>"

                charts_html += """
                    </ul>
                </div>
                """

            charts_html += """
            </div>
            """

            return charts_html

        except Exception as e:
            return f"""
            <div class="section error">
                <h2>&#9888; {self.t['chart_display_error']}</h2>
                <p>{self.t['chart_section_generation_failed']}: {str(e)}</p>
            </div>
            """

    def _generate_per_method_section_safe(self):
        """Generate per-method attribution section if proxy data is available.

        Looks for:
          - env PROXY_METHOD_CSV or LOGS_DIR/proxy_method.csv
          - {output_dir}/unified_monitor.csv  (or env UNIFIED_MONITOR_CSV, falls back to
                                               the same self.performance_csv if absent)

        Returns empty string when either file missing — silent degradation so old runs
        without proxy data still render unchanged.
        """
        try:
            proxy_csv = next(
                (path for path in self._runtime_file_candidates(
                    'PROXY_METHOD_CSV',
                    os.path.join(self.logs_dir, 'proxy_method.csv'),
                    os.path.join(self.output_dir, 'proxy_method.csv'),
                ) if os.path.exists(path)),
                None,
            )
            if not proxy_csv:
                return ""  # no proxy data — silent degradation

            monitor_csv = os.environ.get('UNIFIED_MONITOR_CSV')
            if not monitor_csv:
                # fall back to the performance_csv we already have (it has timestamps)
                monitor_csv = self.performance_csv
            if not os.path.exists(monitor_csv):
                return ""

            # Lazy import to avoid coupling when feature not used
            from analysis.per_method_attribution import (
                compute_per_method_qps,
                compute_per_method_resource,
                filter_proxy_records_by_methods,
                read_monitor_csv,
                read_proxy_csv,
            )
            from visualization.per_method_charts import generate_all_charts
            from visualization.per_method_report import (
                compute_summary,
                get_chart_titles_for_language,
                render_per_method_section,
            )

            proxy_recs = list(read_proxy_csv(proxy_csv))
            allowed_methods = self._load_configured_workload_methods()
            proxy_recs = filter_proxy_records_by_methods(proxy_recs, allowed_methods)
            if not proxy_recs:
                return ""
            qps_rows = compute_per_method_qps(proxy_recs)
            # Re-read proxy (Iterator already consumed)
            resource_rows = compute_per_method_resource(
                filter_proxy_records_by_methods(read_proxy_csv(proxy_csv), allowed_methods),
                # The unified monitor CSV memory column is 'mem_used', while
                # read_monitor_csv defaults to 'mem_used_mb' for unit fixtures.
                # Passing mem_col explicitly keeps production memory attribution non-zero.
                list(read_monitor_csv(monitor_csv, mem_col="mem_used")),
            )

            chain_name = self.config.get('BLOCKCHAIN_NODE', 'chain') if hasattr(self, 'config') else 'chain'
            chart_dir = os.path.join(self.output_dir, 'per_method_charts')
            titles = get_chart_titles_for_language(self.language)
            paths = generate_all_charts(
                qps_rows, resource_rows, chart_dir, chain_name=chain_name, titles=titles,
            )
            summary = compute_summary(qps_rows, resource_rows)
            # Use relative paths so report can be copied around
            rel_paths = {k: os.path.relpath(str(p), self.output_dir) for k, p in paths.items()}
            return render_per_method_section(
                self.language, chain_name, rel_paths, summary,
            )
        except Exception as e:
            import html as _html_mod
            return f'<!-- per_method section skipped: {_html_mod.escape(str(e))} -->'

    def _load_configured_workload_methods(self):
        """Return methods configured for the active single/mixed workload.

        Health probes also pass through the proxy, but per-method attribution is
        only for user workload methods declared in the chain template.
        """
        chain_name = self.config.get('BLOCKCHAIN_NODE') if hasattr(self, 'config') else None
        chain_name = chain_name or os.getenv('BLOCKCHAIN_NODE')
        if not chain_name:
            return None

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        chain_file = os.path.join(root, 'config', 'chains', f'{chain_name}.json')
        if not os.path.exists(chain_file):
            return None

        try:
            with open(chain_file, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
        except Exception:
            return None

        rpc_methods = cfg.get('rpc_methods', {})
        methods = set()
        single = rpc_methods.get('single')
        if isinstance(single, str) and single.strip():
            methods.add(single.strip())

        mixed = rpc_methods.get('mixed')
        if isinstance(mixed, str):
            methods.update(m.strip() for m in mixed.split(',') if m.strip())

        weighted = rpc_methods.get('mixed_weighted')
        if isinstance(weighted, list):
            for item in weighted:
                if isinstance(item, dict) and item.get('method'):
                    methods.add(str(item['method']).strip())

        return methods or None

    def _bottleneck_criteria_items(self):
        """Render bottleneck criteria from the active threshold variables."""
        cpu = os.getenv('BOTTLENECK_CPU_THRESHOLD', '85')
        memory = os.getenv('BOTTLENECK_MEMORY_THRESHOLD', '90')
        disk_iops = os.getenv('BOTTLENECK_DISK_IOPS_THRESHOLD', '90')
        disk_tp = os.getenv('BOTTLENECK_DISK_THROUGHPUT_THRESHOLD', '90')
        network = os.getenv('BOTTLENECK_NETWORK_THRESHOLD', '80')
        success = os.getenv('SUCCESS_RATE_THRESHOLD', '95')
        latency = os.getenv('MAX_LATENCY_THRESHOLD', '1000')
        errors = os.getenv('BOTTLENECK_ERROR_RATE_THRESHOLD', '5')
        sync_gap = os.getenv('BLOCK_HEIGHT_DIFF_THRESHOLD', '50')
        sync_time = os.getenv('BLOCK_HEIGHT_TIME_THRESHOLD', '300')

        if self.language == 'zh':
            return [
                (
                    f"<strong>检测维度（8个指标）：</strong><br>"
                    f"&nbsp;&nbsp;• 资源指标：CPU>{cpu}%、内存>{memory}%、"
                    f"Disk provider-normalized IOPS>{disk_iops}%、"
                    f"Disk provider-normalized 吞吐量>{disk_tp}%、网络>{network}%<br>"
                    f"&nbsp;&nbsp;• <strong>RPC 性能：</strong>成功率&lt;{success}% 或 P99 延迟>{latency}ms<br>"
                    f"&nbsp;&nbsp;• RPC 错误：错误率>{errors}%、连接失败"
                ),
                (
                    "<strong>五场景判断逻辑：</strong><br>"
                    "&nbsp;&nbsp;• <strong>场景A-Resource：</strong>资源超标 + 节点健康 → 误判 → 重置计数器 → 继续测试<br>"
                    "&nbsp;&nbsp;• <strong>场景A-RPC：</strong>RPC性能违规 → 真瓶颈 → 累积计数 → 连续阈值后停止<br>"
                    "&nbsp;&nbsp;• <strong>场景B：</strong>任意瓶颈 + 节点不健康 → 真瓶颈 → 累积计数 → 连续阈值后停止<br>"
                    "&nbsp;&nbsp;• <strong>场景C：</strong>无资源瓶颈 + 节点持续不健康 → 节点故障 → 停止<br>"
                    "&nbsp;&nbsp;• <strong>场景D：</strong>全部指标正常 + 节点健康 → 继续测试"
                ),
                (
                    f"<strong>节点同步健康验证：</strong><br>"
                    f"&nbsp;&nbsp;• 支持 absolute_gap、conditional_gap、reported_lag、freshness_only、health_only 等模式<br>"
                    f"&nbsp;&nbsp;• 数值型 gap/lag 复用 BLOCK_HEIGHT_DIFF_THRESHOLD={sync_gap}；持续异常复用 BLOCK_HEIGHT_TIME_THRESHOLD={sync_time}s<br>"
                    f"&nbsp;&nbsp;• RPC 性能违规可以直接触发瓶颈判断，不要求必须同时出现高度差异常"
                ),
            ]

        return [
            (
                f"<strong>Detection Dimensions (8 metrics):</strong><br>"
                f"&nbsp;&nbsp;• Resource metrics: CPU>{cpu}%, Memory>{memory}%, "
                f"Disk provider-normalized IOPS>{disk_iops}%, "
                f"Disk provider-normalized throughput>{disk_tp}%, Network>{network}%<br>"
                f"&nbsp;&nbsp;• <strong>RPC performance:</strong> Success rate&lt;{success}% or P99 latency>{latency}ms<br>"
                f"&nbsp;&nbsp;• RPC errors: Error rate>{errors}%, connection failures"
            ),
            (
                "<strong>5-scenario decision logic:</strong><br>"
                "&nbsp;&nbsp;• <strong>Scenario A-Resource:</strong> Resource exceeded + node healthy → false positive → reset counter → continue<br>"
                "&nbsp;&nbsp;• <strong>Scenario A-RPC:</strong> RPC performance violation → true bottleneck → accumulate count → stop after consecutive threshold<br>"
                "&nbsp;&nbsp;• <strong>Scenario B:</strong> Any bottleneck + node unhealthy → true bottleneck → accumulate count → stop after consecutive threshold<br>"
                "&nbsp;&nbsp;• <strong>Scenario C:</strong> No resource bottleneck + node persistently unhealthy → node failure → stop<br>"
                "&nbsp;&nbsp;• <strong>Scenario D:</strong> All metrics normal + node healthy → continue testing"
            ),
            (
                f"<strong>Node sync-health verification:</strong><br>"
                f"&nbsp;&nbsp;• Supports absolute_gap, conditional_gap, reported_lag, freshness_only, and health_only modes<br>"
                f"&nbsp;&nbsp;• Numeric gap/lag reuses BLOCK_HEIGHT_DIFF_THRESHOLD={sync_gap}; sustained unhealthy state reuses BLOCK_HEIGHT_TIME_THRESHOLD={sync_time}s<br>"
                f"&nbsp;&nbsp;• RPC performance violations can trigger bottleneck classification without requiring a height-gap anomaly"
            ),
        ]

    def _bottleneck_criteria_list_html(self):
        return "\n".join(f"<li>{item}</li>" for item in self._bottleneck_criteria_items())

    def _generate_bottleneck_section(self):
        """Generate system-level bottleneck analysis section - always display"""
        try:
            bottleneck_detected = False
            if self.bottleneck_data:
                bottleneck_detected = self.bottleneck_data.get('bottleneck_detected', False)

            if bottleneck_detected:
                # Has bottleneck: display detailed information
                max_qps = self.bottleneck_data.get('max_successful_qps', 0)
                bottleneck_qps = self.bottleneck_data.get('bottleneck_qps', 0)
                reasons = self.bottleneck_data.get('bottleneck_reasons', self.t['unknown'])
                severity = self.bottleneck_data.get('severity', 'medium')
                detection_time = self.bottleneck_data.get('detection_time', self.t['unknown'])
                consecutive_detections = self.bottleneck_data.get('consecutive_detections', 0)

                performance_drop = 0.0
                if max_qps > 0:
                    performance_drop = ((bottleneck_qps - max_qps) / max_qps) * 100

                severity_color = {
                    'low': '#28a745',
                    'medium': '#ffc107',
                    'high': '#dc3545'
                }.get(severity.lower(), '#ffc107')

                return f"""
                <div class="section bottleneck-section bottleneck-detected">
                    <h2>&#128680; {self.t['system_bottleneck_analysis']}</h2>

                    <div class="status-callout status-callout-warning">
                        <h3>&#9888; {self.t['system_bottleneck_detected']}</h3>
                    </div>

                    <div class="metric-card-grid">
                        <div class="metric-card">
                            <h4>&#127942; {self.t['max_successful_qps']}</h4>
                            <div class="metric-card-value good">{max_qps}</div>
                        </div>
                        <div class="metric-card">
                            <h4>&#128680; {self.t['bottleneck_trigger_qps']}</h4>
                            <div class="metric-card-value bad">{bottleneck_qps}</div>
                        </div>
                        <div class="metric-card">
                            <h4>&#128201; {self.t['performance_drop']}</h4>
                            <div class="metric-card-value bad">{performance_drop:.1f}%</div>
                        </div>
                    </div>

                    <div class="detail-panel">
                        <h3>&#128269; {self.t['bottleneck_details']}</h3>
                        <p><strong>{self.t['detection_time']}:</strong> {detection_time}</p>
                        <p><strong>{self.t['severity_level']}:</strong> <span style="color: {severity_color}; font-weight: bold;">{severity.upper()}</span></p>
                        <p><strong>{self.t['bottleneck_reason']}:</strong> {reasons}</p>
                        <p><strong>{self.t['consecutive_detections']}:</strong> {consecutive_detections}</p>
                    </div>

                    <div class="bottleneck-criteria-card">
                        <h3 style="margin-top: 0;">&#128203; {self.t['bottleneck_criteria_title']}</h3>
                        <p><strong>{self.t['bottleneck_criteria_note']}</strong></p>
                        <ol class="bottleneck-criteria-list">
                            {self._bottleneck_criteria_list_html()}
                        </ol>
                    </div>
                </div>
                """
            else:
                # No bottleneck: display criteria
                return f"""
                <div class="section bottleneck-section">
                    <h2>&#9989; {self.t['system_bottleneck_analysis']}</h2>

                    <div class="status-callout status-callout-success">
                        <h3>&#9989; {self.t['no_system_bottleneck_detected']}</h3>
                    </div>

                    <div class="bottleneck-criteria-card">
                        <h3 style="margin-top: 0;">&#128203; {self.t['bottleneck_criteria_title']}</h3>
                        <p><strong>{self.t['bottleneck_criteria_note']}</strong></p>
                        <ol class="bottleneck-criteria-list">
                            {self._bottleneck_criteria_list_html()}
                        </ol>
                    </div>

                    <div class="detail-panel">
                        <h3>&#128202; {self.t['bottleneck_current_status']}</h3>
                        <p>{self.t['bottleneck_status_normal']}</p>
                    </div>
                </div>
                """

        except Exception as e:
            return f"""
            <div class="section error">
                <h2>&#9888; {self.t['bottleneck_info_display_error']}</h2>
                <p>{self.t['bottleneck_info_processing_failed']}: {str(e)}</p>
            </div>
            """

    def _get_css_styles(self):
        """Get CSS styles - enhanced version with chart display support"""
        return """
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 24px;
            background: #f4f6f8;
            color: #1f2933;
            line-height: 1.6;
        }
        .container {
            width: min(1440px, 96%);
            margin: 0 auto;
            background-color: #ffffff;
            padding: 32px;
            border-radius: 10px;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
        }
        .report-header {
            border-bottom: 1px solid #d9e2ec;
            padding: 4px 0 22px 0;
            margin-bottom: 20px;
            text-align: center;
        }
        .report-header h1 {
            margin-bottom: 8px;
        }
        .report-meta {
            color: #627d98;
            font-size: 0.95em;
            margin-bottom: 14px;
        }
        .report-capabilities {
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 8px;
        }
        .report-capabilities span {
            display: inline-flex;
            align-items: center;
            min-height: 28px;
            padding: 4px 10px;
            border: 1px solid #dbe3ed;
            border-radius: 999px;
            background: #f8fafc;
            color: #52606d;
            font-size: 0.86em;
            line-height: 1.25;
        }
        .report-nav {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 8px;
            margin-top: 18px;
            padding-top: 16px;
            border-top: 1px solid #eef2f6;
        }
        .report-nav:empty {
            display: none;
        }
        .report-nav a {
            display: inline-flex;
            align-items: center;
            max-width: 260px;
            min-height: 30px;
            padding: 5px 11px;
            border: 1px solid #cbd5e1;
            border-radius: 999px;
            background: #ffffff;
            color: #334e68;
            font-size: 0.86em;
            line-height: 1.25;
            text-decoration: none;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .report-nav a:hover {
            background: #f0f7ff;
            border-color: #9fb3c8;
            color: #102a43;
        }
        .section {
            margin: 26px 0;
            padding: 22px;
            border: 1px solid #d9e2ec;
            border-radius: 8px;
            background-color: #ffffff;
        }
        .section p {
            line-height: 1.72;
            margin: 10px 0;
        }
        .section li {
            line-height: 1.72;
            margin: 7px 0;
        }
        .section ul, .section ol {
            padding-left: 24px;
        }
        .bottleneck-criteria-card {
            background: #f0f7ff;
            border: 1px solid #b9d7f4;
            border-left: 4px solid #2b6cb0;
            padding: 18px 20px;
            border-radius: 8px;
            margin: 18px 0;
        }
        .bottleneck-criteria-card p {
            line-height: 1.78;
            margin: 10px 0 14px 0;
        }
        .bottleneck-criteria-list {
            margin: 12px 0 0 0;
            padding-left: 24px;
        }
        .bottleneck-criteria-list li {
            line-height: 1.9;
            margin: 13px 0;
            padding-left: 4px;
        }
        .bottleneck-criteria-list strong {
            color: #243b53;
        }
        .bottleneck-section {
            border-left: 4px solid #2f855a;
        }
        .bottleneck-detected {
            border-left-color: #c53030;
            background: #fffafa;
        }
        .status-callout {
            border: 1px solid #dbe3ed;
            border-radius: 8px;
            padding: 14px 16px;
            margin: 16px 0;
        }
        .status-callout h3,
        .status-callout h4 {
            margin: 0;
        }
        .status-callout-success {
            background: #f0fff4;
            border-color: #c6f6d5;
        }
        .status-callout-success h3 {
            color: #276749;
        }
        .status-callout-warning {
            background: #fffaf0;
            border-color: #f6e05e;
        }
        .status-callout-warning h3 {
            color: #975a16;
        }
        .metric-card-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin: 18px 0;
        }
        .metric-card,
        .detail-panel {
            background: #ffffff;
            border: 1px solid #dbe3ed;
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
        }
        .metric-card h4,
        .detail-panel h3 {
            margin-top: 0;
        }
        .metric-card-value {
            color: #102a43;
            font-size: 2em;
            font-weight: 700;
            line-height: 1.2;
        }
        .metric-card-value.good {
            color: #2f855a;
        }
        .metric-card-value.bad {
            color: #c53030;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            min-height: 24px;
            padding: 3px 9px;
            border-radius: 999px;
            font-size: 0.86em;
            font-weight: 600;
            line-height: 1.2;
            white-space: nowrap;
        }
        .status-ok {
            background: #f0fff4;
            color: #276749;
            border: 1px solid #c6f6d5;
        }
        .status-warn {
            background: #fffaf0;
            color: #975a16;
            border: 1px solid #f6e05e;
        }
        .status-bad {
            background: #fff5f5;
            color: #c53030;
            border: 1px solid #fed7d7;
        }
        .environment-section {
            background: #f8fafc;
            border-color: #cbd5e1;
        }
        .environment-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 16px;
            margin-top: 14px;
        }
        .environment-card {
            background: #ffffff;
            border: 1px solid #dbe3ed;
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
        }
        .environment-card h3 {
            color: #243b53;
            margin: 0 0 12px 0;
            font-size: 1.02em;
            border-bottom: 1px solid #e5ebf2;
            padding-bottom: 8px;
        }
        .env-item {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: 12px;
            padding: 7px 0;
            border-bottom: 1px solid #eef2f6;
        }
        .env-item:last-child {
            border-bottom: 0;
        }
        .env-label {
            color: #627d98;
            font-size: 0.88em;
        }
        .env-value {
            color: #102a43;
            font-weight: 600;
            text-align: right;
            word-break: break-word;
        }
        .data-quality-section {
            background: #ffffff;
        }
        .quality-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 14px;
            margin-top: 14px;
        }
        .quality-item {
            border: 1px solid #dbe3ed;
            border-radius: 8px;
            padding: 14px;
            background: #f8fafc;
        }
        .quality-label {
            display: block;
            color: #627d98;
            font-size: 0.86em;
            margin-bottom: 8px;
        }
        .quality-value {
            display: block;
            color: #102a43;
            font-size: 1.55em;
            font-weight: 700;
            line-height: 1.1;
        }
        .quality-ok {
            border-left: 4px solid #2f855a;
        }
        .quality-warn {
            border-left: 4px solid #d69e2e;
            background: #fffaf0;
        }
        .quality-neutral {
            border-left: 4px solid #718096;
        }
        .quality-notes {
            margin-top: 14px;
            color: #52606d;
            font-size: 0.94em;
        }
        .quality-notes p {
            margin: 6px 0;
        }
        .info {
            background-color: #EBF5FB;
            padding: 15px;
            border-radius: 6px;
            margin: 15px 0;
            border-left: 4px solid #3498DB;
        }
        .success {
            background-color: #E8F8F5;
            padding: 15px;
            border-radius: 6px;
            margin: 15px 0;
            border-left: 4px solid #27AE60;
        }
        .warning {
            background-color: #FEF5E7;
            padding: 15px;
            border-radius: 6px;
            margin: 15px 0;
            border-left: 4px solid #F39C12;
        }
        .warning tr.warning {
            background-color: #FEF5E7 !important;
        }
        .table-note {
            font-size: 0.9em;
            color: #666;
            margin-top: 10px;
            font-style: italic;
        }
        .error {
            background-color: #FADBD8;
            padding: 15px;
            border-radius: 6px;
            margin: 15px 0;
            border-left: 4px solid #E74C3C;
        }
        .bottleneck-alert {
            background-color: #FEF5E7 !important;
            border-left: 5px solid #E74C3C !important;
        }
        .bottleneck-summary {
            margin: 20px 0;
        }
        .bottleneck-stats {
            display: flex;
            justify-content: space-around;
            margin: 25px 0;
            flex-wrap: wrap;
        }
        .stat-item {
            text-align: center;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 10px;
            min-width: 180px;
            margin: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .stat-value {
            margin-top: 10px;
            font-weight: bold;
        }
        .bottleneck-details, .bottleneck-recommendations, .bottleneck-actions {
            margin: 25px 0;
        }

        /* Chart display styles */
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 30px;
            margin: 25px 0;
        }
        .chart-item {
            background-color: white;
            border: 1px solid #e1e8ed;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .chart-item:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(0,0,0,0.12);
        }
        .chart-item h3 {
            color: #2c3e50;
            margin-top: 0;
            margin-bottom: 15px;
            font-size: 1.2em;
            border-bottom: 2px solid #3498db;
            padding-bottom: 8px;
        }
        .chart-description {
            margin-bottom: 20px;
        }
        .chart-description p {
            color: #5a6c7d;
            font-size: 0.95em;
            margin: 0;
            line-height: 1.5;
        }
        .chart-container {
            text-align: center;
            background-color: #fafbfc;
            border-radius: 8px;
            padding: 15px;
            border: 1px solid #e9ecef;
        }
        .chart-image {
            max-width: 100%;
            height: auto;
            border-radius: 6px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.2s ease;
        }
        .chart-image:hover {
            transform: scale(1.02);
        }
        .charts-summary {
            background-color: #e8f5e8;
            padding: 20px;
            border-radius: 8px;
            margin: 25px 0;
            border-left: 4px solid #27AE60;
        }
        .charts-summary h3 {
            color: #155724;
            margin-top: 0;
        }
        .charts-summary ul {
            margin: 10px 0;
            padding-left: 20px;
        }
        .charts-summary li {
            margin: 8px 0;
            color: #155724;
        }
        .missing-charts {
            background-color: #FEF5E7;
            padding: 20px;
            border-radius: 8px;
            margin: 25px 0;
            border-left: 4px solid #ffc107;
        }
        .missing-charts h3 {
            color: #856404;
            margin-top: 0;
        }
        .missing-charts ul {
            margin: 10px 0;
            padding-left: 20px;
        }
        .missing-charts li {
            margin: 8px 0;
            color: #856404;
        }

        /* Block height statistics styles */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stat-item {
            text-align: center;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #e9ecef;
        }
        .stat-value {
            font-size: 1.8em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }
        .stat-label {
            font-size: 0.9em;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .file-info {
            margin-top: 20px;
            padding: 15px;
            background-color: #e8f4fd;
            border-radius: 6px;
            border-left: 4px solid #007bff;
        }
        .file-info code {
            background-color: #f1f3f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }

        /* Table styles */
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            background-color: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 1px 4px rgba(15, 23, 42, 0.08);
        }
        th, td {
            border: 1px solid #dbe3ed;
            padding: 12px 15px;
            text-align: left;
        }
        th {
            background-color: #edf2f7;
            font-weight: 600;
            color: #243b53;
        }
        tr:nth-child(even) {
            background-color: #f8fafc;
        }
        tr:hover {
            background-color: #eef5ff;
        }

        /* Heading styles */
        h1 {
            color: #102a43;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.2em;
        }
        h2 {
            color: #243b53;
            border-bottom: 2px solid #9fb3c8;
            padding-bottom: 8px;
            margin-top: 30px;
        }
        h3 {
            color: #5a6c7d;
            margin-top: 20px;
        }

        /* Responsive design */
        @media (max-width: 1200px) {
            .container {
                width: 98%;
                padding: 20px;
            }
            table {
                font-size: 0.9em;
            }
        }

        /* Print styles */
        @media print {
            body {
                background: #ffffff;
                padding: 0;
            }
            .container {
                box-shadow: none;
                border: 1px solid #ccc;
                width: 100%;
                padding: 18px;
            }
            .report-nav,
            .report-capabilities {
                display: none;
            }
            .section,
            .chart-item,
            .environment-card,
            .quality-item,
            .metric-card,
            .detail-panel {
                box-shadow: none !important;
                break-inside: avoid;
            }
            table,
            img {
                break-inside: avoid;
            }
            .chart-item:hover {
                transform: none;
                box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            }
            .chart-image:hover {
                transform: none;
            }
        }

        /* Chart Gallery Styles */
        .chart-summary {
            background-color: #e3f2fd;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #2196f3;
        }
        .chart-category {
            margin-bottom: 30px;
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }
        .chart-category h3 {
            color: #1976d2;
            border-bottom: 2px solid #e3f2fd;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .chart-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-top: 15px;
        }
        .chart-item {
            background-color: #fafafa;
            border-radius: 8px;
            padding: 15px;
            border: 1px solid #e0e0e0;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .chart-item:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.12);
        }
        .chart-item h4 {
            color: #424242;
            margin-bottom: 10px;
            font-size: 1.1em;
        }
        .chart-container {
            text-align: center;
            background-color: white;
            border-radius: 6px;
            padding: 10px;
            border: 1px solid #e8e8e8;
        }
        .chart {
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            transition: transform 0.2s ease;
        }
        .chart:hover {
            transform: scale(1.02);
        }

        /* Modern report polish overrides */
        .container > .section:first-of-type {
            margin-top: 22px;
        }
        .section h2 {
            display: flex;
            align-items: center;
            gap: 8px;
            color: #102a43;
            border-bottom: 1px solid #d9e2ec;
            padding-bottom: 12px;
            margin: 0 0 18px 0;
            font-size: 1.34em;
            line-height: 1.35;
        }
        .section h3 {
            color: #243b53;
            margin: 18px 0 12px 0;
            line-height: 1.45;
        }
        .section h4 {
            color: #334e68;
            line-height: 1.45;
        }
        .info,
        .success,
        .warning,
        .error,
        .missing-charts,
        .charts-summary,
        .file-info {
            border-radius: 8px;
            padding: 16px 18px;
            box-shadow: none;
        }
        .info {
            background: #f0f7ff;
            border-left-color: #2b6cb0;
        }
        .success {
            background: #f0fff4;
            border-left-color: #2f855a;
        }
        .warning,
        .missing-charts {
            background: #fffaf0;
            border-left-color: #d69e2e;
        }
        .error {
            background: #fff5f5;
            border-left-color: #c53030;
        }
        .info-card,
        .chart-category,
        .chart-item,
        .stat-card,
        .stat-item {
            border: 1px solid #dbe3ed !important;
            border-radius: 8px !important;
            background: #ffffff !important;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05) !important;
        }
        .info-card {
            padding: 18px;
            margin: 16px 0;
        }
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
        }
        .stats-grid {
            gap: 16px;
        }
        .stat-card,
        .stat-item {
            padding: 16px !important;
        }
        .stat-card .stat-label,
        .stat-item .stat-label {
            color: #627d98;
            font-size: 0.84em;
            letter-spacing: 0;
            text-transform: none;
        }
        .stat-card .stat-value,
        .stat-item .stat-value {
            color: #102a43;
            font-size: 1.45em;
            line-height: 1.2;
        }
        table,
        .report-table,
        .data-table,
        .performance-table {
            border: 1px solid #dbe3ed !important;
            border-radius: 8px !important;
            box-shadow: 0 1px 4px rgba(15, 23, 42, 0.06) !important;
            overflow: hidden;
        }
        .report-table {
            width: 100%;
            min-width: 680px;
            border-collapse: collapse;
            margin-top: 15px;
            background: #ffffff;
        }
        .report-table th,
        .report-table td {
            padding: 12px 14px;
        }
        .config-table td:first-child,
        .summary-table td:first-child {
            color: #52606d;
            font-weight: 500;
        }
        .numeric-cell {
            text-align: right;
        }
        .metric-value {
            color: #102a43;
            font-weight: 700;
        }
        .muted-cell {
            color: #627d98;
            font-size: 0.92em;
        }
        .total-row {
            background: #f0f7ff !important;
            font-weight: 700;
            border-top: 2px solid #9fb3c8;
        }
        .completeness-green {
            color: #2f855a;
            font-weight: 700;
        }
        .completeness-orange {
            color: #975a16;
            font-weight: 700;
        }
        .completeness-red {
            color: #c53030;
            font-weight: 700;
        }
        .correlation-very-strong {
            background: #f0fff4 !important;
        }
        .correlation-strong {
            background: #f7fffa !important;
        }
        .correlation-moderate {
            background: #fffaf0 !important;
        }
        .correlation-weak {
            background: #f8fafc !important;
        }
        .monitoring-overhead-table,
        .correlation-table,
        .disk-warning-table,
        .disk-stats-table,
        .sync-health-table {
            font-size: 0.95em;
        }
        .disk-warning-table {
            margin-top: 12px;
        }
        .row-group-label {
            color: #243b53;
            font-weight: 700;
            background: #f8fafc;
            vertical-align: middle;
        }
        .disk-stats-table td:nth-child(n+3),
        .sync-health-table td:nth-child(n+2) {
            font-variant-numeric: tabular-nums;
        }
        .sync-comparison-panel {
            margin: 20px 0;
        }
        .sync-stats-grid {
            margin: 10px 0 16px 0;
        }
        th {
            background: #edf2f7 !important;
            color: #243b53 !important;
            border-color: #dbe3ed !important;
            line-height: 1.45;
        }
        td {
            border-color: #e5ebf2 !important;
            color: #334e68;
            line-height: 1.55;
        }
        tr:nth-child(even) {
            background: #f8fafc !important;
        }
        tr:hover {
            background: #eef5ff !important;
        }
        .charts-grid,
        .chart-grid {
            gap: 20px;
        }
        .chart-category {
            padding: 18px;
            margin-bottom: 22px;
        }
        .chart-category h3,
        .chart-item h3,
        .chart-item h4 {
            border-bottom: 1px solid #d9e2ec !important;
            color: #243b53 !important;
            padding-bottom: 10px;
            margin-top: 0;
        }
        .chart-item {
            padding: 18px !important;
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }
        .chart-item:hover {
            transform: translateY(-1px);
            box-shadow: 0 12px 26px rgba(15, 23, 42, 0.08) !important;
        }
        .chart-container {
            background: #f8fafc !important;
            border-color: #e5ebf2 !important;
            border-radius: 8px !important;
        }
        .chart-image,
        .chart {
            border-radius: 6px;
            box-shadow: none;
        }
        .chart-image:hover,
        .chart:hover {
            transform: none;
        }
        .table-note {
            color: #627d98;
            line-height: 1.6;
            font-style: normal;
        }
        div[style*="background: #d4edda"],
        div[style*="background: #e7f3ff"],
        div[style*="background: #f8f9fa"],
        div[style*="background: white"] {
            border-radius: 8px !important;
        }
        """

    def _generate_performance_summary(self, df):
        """Generate performance summary section"""
        try:
            # Calculate basic statistics
            cpu_avg = df['cpu_usage'].mean() if 'cpu_usage' in df.columns and len(df) > 0 else 0
            cpu_max = df['cpu_usage'].max() if 'cpu_usage' in df.columns and len(df) > 0 else 0
            mem_avg = df['mem_usage'].mean() if 'mem_usage' in df.columns and len(df) > 0 else 0

            # DATA Device statistics - using unified field format matching.
            data_iops_cols = [col for col in df.columns if col.startswith('data_') and col.endswith('_total_iops')]
            data_iops_stats = self._metric_stats(df, data_iops_cols[0]) if data_iops_cols else None
            data_iops_avg = self._format_metric_stats(data_iops_stats, 0)[1]

            # ACCOUNTS Device statistics - only show when runtime data is present.
            accounts_iops_cols = [col for col in df.columns if col.startswith('accounts_') and col.endswith('_total_iops')]
            accounts_iops_stats = self._metric_stats(df, accounts_iops_cols[0]) if accounts_iops_cols else None
            accounts_iops_row = ""
            if accounts_iops_cols:
                accounts_iops_avg = self._format_metric_stats(accounts_iops_stats, 0)[1]
                accounts_iops_row = (
                    f"<tr><td>{self.t['accounts_device_avg_iops']}</td>"
                    f"<td class=\"numeric-cell metric-value\">{accounts_iops_avg}</td></tr>"
                )

            return f"""
            <div class="section">
                <h2>&#128202; {self.t['performance_summary']}</h2>
                <table class="report-table summary-table">
                    <thead>
                        <tr>
                            <th>{self.t['metric']}</th>
                            <th class="numeric-cell">{self.t['value']}</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr><td>{self.t['avg_cpu_usage']}</td><td class="numeric-cell metric-value">{cpu_avg:.1f}%</td></tr>
                        <tr><td>{self.t['peak_cpu_usage']}</td><td class="numeric-cell metric-value">{cpu_max:.1f}%</td></tr>
                        <tr><td>{self.t['avg_memory_usage']}</td><td class="numeric-cell metric-value">{mem_avg:.1f}%</td></tr>
                        <tr><td>{self.t['data_device_avg_iops']}</td><td class="numeric-cell metric-value">{data_iops_avg}</td></tr>
                        {accounts_iops_row}
                        <tr><td>{self.t['monitoring_data_points']}</td><td class="numeric-cell metric-value">{len(df):,}</td></tr>
                    </tbody>
                </table>
            </div>
            """
        except Exception as e:
            return f"<div class='error'>{self.t['performance_summary_generation_failed']}: {str(e)}</div>"

def main():
    parser = argparse.ArgumentParser(description='Report Generator - Enhanced version + bottleneck mode support')
    parser.add_argument('performance_csv', help='System performance monitoring CSV file')
    parser.add_argument('-c', '--config', help='Configuration file', default='config_loader.sh')
    parser.add_argument('-o', '--overhead-csv', help='Monitoring overhead CSV file')
    parser.add_argument('--bottleneck-mode', action='store_true', help='Enable bottleneck analysis mode')
    parser.add_argument('--bottleneck-info', help='Bottleneck information JSON file path')
    parser.add_argument('--language', choices=['en', 'zh'], default='en', help='Report language (en or zh)')

    args = parser.parse_args()

    if not os.path.exists(args.performance_csv):
        print(f"❌ File does not exist: {args.performance_csv}")
        return 1

    # Check bottleneck information file
    bottleneck_info_file = None
    if args.bottleneck_mode or args.bottleneck_info:
        if args.bottleneck_info and os.path.exists(args.bottleneck_info):
            bottleneck_info_file = args.bottleneck_info
            print(f"📊 Using bottleneck information file: {bottleneck_info_file}")
        else:
            print("⚠️ Bottleneck mode enabled but bottleneck information file not found, will generate standard report")

    # Create generator with specified language
    generator = ReportGenerator(args.performance_csv, args.config, args.overhead_csv, bottleneck_info_file, args.language)

    result = generator.generate_html_report()

    if result:
        if bottleneck_info_file:
            print("🎉 Bottleneck mode HTML report generated successfully!")
        else:
            print("🎉 Enhanced HTML report generated successfully!")
        return 0
    else:
        print("❌ HTML report generation failed")
        return 1

if __name__ == "__main__":
    exit(main())
