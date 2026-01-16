"""
Data Processing Utilities
Handles CSV uploads, data validation, and preprocessing
"""

import pandas as pd
import numpy as np
from io import StringIO
import json


class DataProcessor:
    """Data processing and validation utilities"""
    
    @staticmethod
    def read_csv(file_path):
        """Read CSV file and return DataFrame"""
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {str(e)}")
    
    @staticmethod
    def validate_data(df, min_rows=10, min_cols=2):
        """Validate uploaded data"""
        errors = []
        
        if df.empty:
            errors.append("Dataset is empty")
        
        if len(df) < min_rows:
            errors.append(f"Insufficient data: Dataset has only {len(df)} rows. Minimum {min_rows} rows required for reliable machine learning analysis.")
        
        if len(df.columns) < min_cols:
            errors.append(f"Insufficient columns: Dataset has only {len(df.columns)} columns. Minimum {min_cols} columns required.")
        
        # Check for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < min_cols:
            errors.append(f"Insufficient numeric data: Found only {len(numeric_cols)} numeric columns. At least {min_cols} numeric columns are required for machine learning algorithms.")
        
        # Check for too many missing values
        missing_pct = (df.isnull().sum() / len(df)) * 100
        high_missing = missing_pct[missing_pct > 50]
        if len(high_missing) > 0:
            errors.append(f"Columns with >50% missing values: {', '.join(high_missing.index.tolist())}. Please clean your data before uploading.")
        
        # Check if entire dataset has too many missing values
        overall_missing = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        if overall_missing > 70:
            errors.append(f"Too many missing values overall: {overall_missing:.1f}% of entire dataset is missing.")
        
        return errors
    
    @staticmethod
    def get_data_info(df):
        """Get detailed information about the dataset"""
        info = {
            'shape': df.shape,
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': ((df.isnull().sum() / len(df)) * 100).round(2).to_dict()
        }
        
        # Numeric columns statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        info['numeric_columns'] = numeric_cols
        
        if numeric_cols:
            stats = df[numeric_cols].describe().to_dict()
            info['statistics'] = stats
        
        return info
    
    @staticmethod
    def get_preview(df, n_rows=10):
        """Get preview of the dataset"""
        preview = {
            'head': df.head(n_rows).to_dict(orient='records'),
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict()
        }
        return preview
    
    @staticmethod
    def get_numeric_columns(df):
        """Get list of numeric columns"""
        return df.select_dtypes(include=[np.number]).columns.tolist()
    
    @staticmethod
    def prepare_regression_data(df, target_column, feature_columns=None):
        """Prepare data for regression algorithms"""
        # Get numeric columns
        numeric_cols = DataProcessor.get_numeric_columns(df)
        
        if target_column not in numeric_cols:
            raise ValueError(f"Target column '{target_column}' must be numeric")
        
        # Remove rows with missing values in target
        df_clean = df.dropna(subset=[target_column])
        
        # If no feature columns specified, use all numeric columns except target
        if feature_columns is None:
            feature_columns = [col for col in numeric_cols if col != target_column]
        
        # Validate feature columns
        invalid_features = [col for col in feature_columns if col not in numeric_cols]
        if invalid_features:
            raise ValueError(f"Feature columns must be numeric: {', '.join(invalid_features)}")
        
        # Remove rows with missing values in features
        df_clean = df_clean.dropna(subset=feature_columns)
        
        if len(df_clean) < 10:
            raise ValueError("Insufficient data after removing missing values")
        
        X = df_clean[feature_columns].values
        y = df_clean[target_column].values
        
        return X, y, feature_columns
    
    @staticmethod
    def prepare_clustering_data(df, feature_columns=None):
        """Prepare data for clustering algorithms"""
        # Get numeric columns
        numeric_cols = DataProcessor.get_numeric_columns(df)
        
        if not numeric_cols:
            raise ValueError("No numeric columns found for clustering")
        
        # If no feature columns specified, use all numeric columns
        if feature_columns is None:
            feature_columns = numeric_cols
        
        # Validate feature columns
        invalid_features = [col for col in feature_columns if col not in numeric_cols]
        if invalid_features:
            raise ValueError(f"Feature columns must be numeric: {', '.join(invalid_features)}")
        
        # Remove rows with missing values
        df_clean = df[feature_columns].dropna()
        
        if len(df_clean) < 10:
            raise ValueError("Insufficient data after removing missing values")
        
        X = df_clean.values
        
        return X, feature_columns
    
    @staticmethod
    def save_results(results, file_path):
        """Save results to JSON file"""
        try:
            with open(file_path, 'w') as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            raise ValueError(f"Error saving results: {str(e)}")
    
    @staticmethod
    def load_results(file_path):
        """Load results from JSON file"""
        try:
            with open(file_path, 'r') as f:
                results = json.load(f)
            return results
        except Exception as e:
            raise ValueError(f"Error loading results: {str(e)}")
    
    @staticmethod
    def get_correlation_matrix(df):
        """Calculate correlation matrix for numeric columns"""
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty or len(numeric_df.columns) < 2:
            return None
        return numeric_df.corr()
    
    @staticmethod
    def auto_detect_target(df):
        """
        Automatically detect the best target column for regression.
        Priority:
        1. Last numeric column (common convention)
        2. Column with highest variance
        3. Column with most correlation to other features
        """
        numeric_cols = DataProcessor.get_numeric_columns(df)
        
        if not numeric_cols:
            return None
        
        if len(numeric_cols) == 1:
            return numeric_cols[0]
        
        # Default: use last column (common ML convention)
        target = numeric_cols[-1]
        
        # Check if last column has reasonable variance
        target_variance = df[target].var()
        
        # If variance is very low, find column with highest variance
        if target_variance < 0.01:
            variances = df[numeric_cols].var()
            target = variances.idxmax()
        
        return target
    
    @staticmethod
    def auto_select_features(df, target_column=None, max_features=20):
        """
        Automatically select best features for modeling.
        Excludes target column and selects based on:
        1. Correlation with target (for regression)
        2. Variance (remove low-variance features)
        3. Missing values (remove columns with >50% missing)
        """
        numeric_cols = DataProcessor.get_numeric_columns(df)
        
        if not numeric_cols:
            return []
        
        # Remove target from features
        if target_column and target_column in numeric_cols:
            numeric_cols = [col for col in numeric_cols if col != target_column]
        
        # Remove columns with >50% missing
        missing_pct = (df[numeric_cols].isnull().sum() / len(df)) * 100
        valid_cols = missing_pct[missing_pct <= 50].index.tolist()
        
        if not valid_cols:
            return numeric_cols[:max_features]
        
        # Remove low-variance features (variance < 0.01)
        variances = df[valid_cols].var()
        high_var_cols = variances[variances >= 0.01].index.tolist()
        
        if not high_var_cols:
            return valid_cols[:max_features]
        
        # If we have a target, sort by correlation
        if target_column and target_column in df.columns:
            correlations = df[high_var_cols + [target_column]].corr()[target_column].abs()
            correlations = correlations.drop(target_column)
            sorted_features = correlations.sort_values(ascending=False).index.tolist()
            return sorted_features[:max_features]
        
        return high_var_cols[:max_features]
    
    @staticmethod
    def detect_problem_type(df):
        """
        Detect if data is better suited for regression or clustering.
        Returns: 'regression', 'clustering', or 'both'
        """
        numeric_cols = DataProcessor.get_numeric_columns(df)
        
        if len(numeric_cols) < 2:
            return 'unknown'
        
        # If we have a clear target (last column has good correlation with others)
        if len(numeric_cols) >= 2:
            target = numeric_cols[-1]
            features = numeric_cols[:-1]
            
            if len(features) > 0:
                corr_matrix = df[numeric_cols].corr()
                target_corr = corr_matrix[target].drop(target).abs()
                
                # If any feature has correlation > 0.3 with target, suggest regression
                if target_corr.max() > 0.3:
                    return 'regression'
        
        # Otherwise suggest both
        return 'both'
    
    @staticmethod
    def get_data_insights(df):
        """Get comprehensive data insights for automatic analysis"""
        insights = {
            'shape': list(df.shape),  # Convert tuple to list for JSON serialization
            'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
            'duplicates': int(df.duplicated().sum()),
            'numeric_columns': DataProcessor.get_numeric_columns(df),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'missing_summary': {
                'total_missing': int(df.isnull().sum().sum()),
                'columns_with_missing': df.columns[df.isnull().any()].tolist(),
                'missing_percentage': float((df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100)
            }
        }
        
        # Auto-detect target and features (ensure they're JSON-serializable)
        auto_target = DataProcessor.auto_detect_target(df)
        auto_features = DataProcessor.auto_select_features(df, auto_target)
        
        insights['auto_target'] = auto_target if auto_target is None else str(auto_target)
        insights['auto_features'] = auto_features if auto_features is None else [str(f) for f in auto_features]
        insights['problem_type'] = DataProcessor.detect_problem_type(df)
        
        # Basic statistics for numeric columns
        if insights['numeric_columns']:
            numeric_df = df[insights['numeric_columns']]
            insights['statistics'] = {
                'mean': {str(k): float(v) for k, v in numeric_df.mean().to_dict().items()},
                'std': {str(k): float(v) for k, v in numeric_df.std().to_dict().items()},
                'min': {str(k): float(v) for k, v in numeric_df.min().to_dict().items()},
                'max': {str(k): float(v) for k, v in numeric_df.max().to_dict().items()},
                'median': {str(k): float(v) for k, v in numeric_df.median().to_dict().items()}
            }
        
        return insights
