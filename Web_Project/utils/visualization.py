"""
Visualization Utilities
Generates plots for regression and clustering results
"""

import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import BytesIO
import base64


class VisualizationGenerator:
    """Generate visualizations for ML results"""
    
    @staticmethod
    def set_style():
        """Set matplotlib style"""
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10
    
    @staticmethod
    def fig_to_base64(fig):
        """Convert matplotlib figure to base64 string"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        return f"data:image/png;base64,{image_base64}"
    
    @staticmethod
    def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted"):
        """Plot actual vs predicted values for regression"""
        VisualizationGenerator.set_style()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return VisualizationGenerator.fig_to_base64(fig)
    
    @staticmethod
    def plot_residuals(y_true, y_pred, title="Residual Plot"):
        """Plot residuals for regression"""
        VisualizationGenerator.set_style()
        
        residuals = y_true - y_pred
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        return VisualizationGenerator.fig_to_base64(fig)
    
    @staticmethod
    def plot_coefficients(coefficients, feature_names, title="Feature Coefficients"):
        """Plot feature coefficients for regression"""
        VisualizationGenerator.set_style()
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.4)))
        
        # Sort by absolute value
        indices = np.argsort(np.abs(coefficients))[::-1]
        sorted_coef = coefficients[indices]
        sorted_names = [feature_names[i] for i in indices]
        
        colors = ['green' if c > 0 else 'red' for c in sorted_coef]
        
        ax.barh(range(len(sorted_coef)), sorted_coef, color=colors, alpha=0.7)
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel('Coefficient Value')
        ax.set_title(title)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='x')
        
        return VisualizationGenerator.fig_to_base64(fig)
    
    @staticmethod
    def plot_feature_importance(coefficients, feature_names, title="Feature Importance"):
        """Plot feature importance (absolute coefficients)"""
        VisualizationGenerator.set_style()
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.4)))
        
        importance = np.abs(coefficients)
        indices = np.argsort(importance)[::-1]
        sorted_importance = importance[indices]
        sorted_names = [feature_names[i] for i in indices]
        
        ax.barh(range(len(sorted_importance)), sorted_importance, color='steelblue', alpha=0.7)
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel('Absolute Coefficient Value')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='x')
        
        return VisualizationGenerator.fig_to_base64(fig)
    
    @staticmethod
    def plot_clustering_scatter(X, labels, centers=None, title="Clustering Results"):
        """Plot clustering results (2D or first 2 dimensions)"""
        VisualizationGenerator.set_style()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Use first 2 dimensions if data is high-dimensional
        X_plot = X[:, :2] if X.shape[1] >= 2 else X
        
        # Create scatter plot
        scatter = ax.scatter(X_plot[:, 0], X_plot[:, 1], c=labels, 
                            cmap='viridis', alpha=0.6, edgecolors='k', linewidth=0.5)
        
        # Plot centers if provided
        if centers is not None:
            centers_plot = centers[:, :2] if centers.shape[1] >= 2 else centers
            ax.scatter(centers_plot[:, 0], centers_plot[:, 1], 
                      c='red', marker='X', s=200, edgecolors='black', linewidth=2,
                      label='Centroids')
        
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title(title)
        plt.colorbar(scatter, ax=ax, label='Cluster')
        if centers is not None:
            ax.legend()
        
        return VisualizationGenerator.fig_to_base64(fig)
    
    @staticmethod
    def plot_elbow_curve(k_values, inertias, title="Elbow Method"):
        """Plot elbow curve for K-Means"""
        VisualizationGenerator.set_style()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('Inertia (Within-Cluster Sum of Squares)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        return VisualizationGenerator.fig_to_base64(fig)
    
    @staticmethod
    def plot_silhouette_scores(k_values, silhouettes, title="Silhouette Scores"):
        """Plot silhouette scores"""
        VisualizationGenerator.set_style()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(k_values, silhouettes, 'go-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('Silhouette Score')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
        
        return VisualizationGenerator.fig_to_base64(fig)
    
    @staticmethod
    def plot_bic_aic(n_values, bic_scores, aic_scores, title="Model Selection Criteria"):
        """Plot BIC and AIC scores for GMM"""
        VisualizationGenerator.set_style()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(n_values, bic_scores, 'bo-', linewidth=2, markersize=8, label='BIC')
        ax.plot(n_values, aic_scores, 'ro-', linewidth=2, markersize=8, label='AIC')
        
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Score (Lower is Better)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return VisualizationGenerator.fig_to_base64(fig)
    
    @staticmethod
    def plot_cluster_distribution(cluster_stats, title="Cluster Distribution"):
        """Plot cluster size distribution"""
        VisualizationGenerator.set_style()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        clusters = [stat['cluster'] for stat in cluster_stats]
        sizes = [stat['size'] for stat in cluster_stats]
        percentages = [stat['percentage'] for stat in cluster_stats]
        
        # Bar chart
        ax1.bar(clusters, sizes, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Cluster')
        ax1.set_ylabel('Number of Points')
        ax1.set_title('Cluster Sizes')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Pie chart
        ax2.pie(percentages, labels=[f'Cluster {c}' for c in clusters], 
               autopct='%1.1f%%', startangle=90)
        ax2.set_title('Cluster Distribution (%)')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        return VisualizationGenerator.fig_to_base64(fig)
    
    @staticmethod
    def plot_correlation_heatmap(df, title="Correlation Heatmap"):
        """Plot correlation heatmap for numeric features"""
        VisualizationGenerator.set_style()
        
        # Get numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty or len(numeric_df.columns) < 2:
            # Return empty plot
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'Insufficient numeric data for correlation',
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return VisualizationGenerator.fig_to_base64(fig)
        
        # Calculate correlation
        corr_matrix = numeric_df.corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(max(10, len(corr_matrix) * 0.8), 
                                        max(8, len(corr_matrix) * 0.7)))
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, square=True,
                   linewidths=1, cbar_kws={"shrink": 0.8},
                   vmin=-1, vmax=1, ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        return VisualizationGenerator.fig_to_base64(fig)
    
    @staticmethod
    def plot_distribution_grid(df, max_features=6, title="Feature Distributions"):
        """Plot distribution of numeric features"""
        VisualizationGenerator.set_style()
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'No numeric features found',
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return VisualizationGenerator.fig_to_base64(fig)
        
        # Limit number of features to plot
        cols_to_plot = numeric_cols[:max_features]
        n_cols = len(cols_to_plot)
        n_rows = (n_cols + 2) // 3  # 3 columns max
        
        fig, axes = plt.subplots(n_rows, min(3, n_cols), 
                                figsize=(15, n_rows * 4))
        
        if n_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_cols > 1 else [axes]
        
        for idx, col in enumerate(cols_to_plot):
            ax = axes[idx]
            data = df[col].dropna()
            
            # Histogram with KDE
            ax.hist(data, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {col}', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add statistics text
            stats_text = f'Mean: {data.mean():.2f}\\nStd: {data.std():.2f}'
            ax.text(0.7, 0.95, stats_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='wheat', alpha=0.5))
        
        # Hide extra subplots
        for idx in range(n_cols, len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        return VisualizationGenerator.fig_to_base64(fig)
    
    @staticmethod
    def plot_box_plots(df, max_features=6, title="Box Plots - Outlier Detection"):
        """Plot box plots for outlier detection"""
        VisualizationGenerator.set_style()
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'No numeric features found',
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return VisualizationGenerator.fig_to_base64(fig)
        
        # Limit features
        cols_to_plot = numeric_cols[:max_features]
        
        fig, ax = plt.subplots(figsize=(max(10, len(cols_to_plot) * 1.5), 6))
        
        # Prepare data for box plot
        data_to_plot = [df[col].dropna() for col in cols_to_plot]
        
        bp = ax.boxplot(data_to_plot, labels=cols_to_plot, patch_artist=True,
                       notch=True, showmeans=True)
        
        # Color the boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(cols_to_plot)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Features', fontweight='bold')
        ax.set_ylabel('Values', fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return VisualizationGenerator.fig_to_base64(fig)
