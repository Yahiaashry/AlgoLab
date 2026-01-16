"""
Clustering Models Handler
Implements K-Means and Gaussian Mixture Models (GMM)
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score


class BaseClusteringModel:
    """Base class for clustering models"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
    
    def scale_data(self, X):
        """Scale data"""
        return self.scaler.fit_transform(X)
    
    def calculate_metrics(self, X, labels):
        """Calculate clustering metrics"""
        metrics = {}
        
        if len(np.unique(labels)) > 1:
            metrics['silhouette'] = float(silhouette_score(X, labels))
            metrics['davies_bouldin'] = float(davies_bouldin_score(X, labels))
        else:
            metrics['silhouette'] = 0.0
            metrics['davies_bouldin'] = 0.0
        
        return metrics
    
    def get_cluster_stats(self, X, labels):
        """Get statistics for each cluster"""
        unique_labels = np.unique(labels)
        stats = []
        
        for label in unique_labels:
            cluster_points = X[labels == label]
            stats.append({
                'cluster': int(label),
                'size': int(len(cluster_points)),
                'percentage': float(len(cluster_points) / len(X) * 100)
            })
        
        return stats


class KMeansModel(BaseClusteringModel):
    """K-Means Clustering Model"""
    
    def train(self, X, params=None):
        """Train K-Means model"""
        if params is None:
            params = {}
        
        n_clusters = params.get('n_clusters', 3)
        max_iter = params.get('max_iter', 300)
        n_init = params.get('n_init', 10)
        
        # Scale data
        X_scaled = self.scale_data(X)
        
        # Train model
        self.model = KMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            n_init=n_init,
            random_state=42
        )
        
        labels = self.model.fit_predict(X_scaled)
        
        # Calculate metrics
        metrics = self.calculate_metrics(X_scaled, labels)
        cluster_stats = self.get_cluster_stats(X_scaled, labels)
        
        # Elbow method data
        inertias = []
        silhouettes = []
        k_range = range(2, min(11, len(X) // 2))
        
        for k in k_range:
            kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
            temp_labels = kmeans_temp.fit_predict(X_scaled)
            inertias.append(float(kmeans_temp.inertia_))
            if k > 1:
                silhouettes.append(float(silhouette_score(X_scaled, temp_labels)))
        
        return {
            'algorithm': 'K-Means Clustering',
            'n_clusters': n_clusters,
            'labels': labels.tolist(),
            'centers': self.model.cluster_centers_.tolist(),
            'inertia': float(self.model.inertia_),
            'metrics': metrics,
            'cluster_stats': cluster_stats,
            'elbow_data': {
                'k_values': list(k_range),
                'inertias': inertias,
                'silhouettes': silhouettes
            },
            'X_scaled': X_scaled.tolist()
        }


class GMMModel(BaseClusteringModel):
    """Gaussian Mixture Model"""
    
    def train(self, X, params=None):
        """Train GMM model"""
        if params is None:
            params = {}
        
        n_components = params.get('n_components', 3)
        covariance_type = params.get('covariance_type', 'full')
        max_iter = params.get('max_iter', 100)
        
        # Scale data
        X_scaled = self.scale_data(X)
        
        # Train model
        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            random_state=42
        )
        
        self.model.fit(X_scaled)
        labels = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Calculate metrics
        metrics = self.calculate_metrics(X_scaled, labels)
        cluster_stats = self.get_cluster_stats(X_scaled, labels)
        
        # Additional GMM metrics
        metrics['aic'] = float(self.model.aic(X_scaled))
        metrics['bic'] = float(self.model.bic(X_scaled))
        metrics['log_likelihood'] = float(self.model.score(X_scaled))
        
        # Optimal components analysis
        bic_scores = []
        aic_scores = []
        n_range = range(2, min(11, len(X) // 2))
        
        for n in n_range:
            gmm_temp = GaussianMixture(n_components=n, random_state=42)
            gmm_temp.fit(X_scaled)
            bic_scores.append(float(gmm_temp.bic(X_scaled)))
            aic_scores.append(float(gmm_temp.aic(X_scaled)))
        
        return {
            'algorithm': 'Gaussian Mixture Model',
            'n_components': n_components,
            'covariance_type': covariance_type,
            'labels': labels.tolist(),
            'probabilities': probabilities.tolist(),
            'means': self.model.means_.tolist(),
            'metrics': metrics,
            'cluster_stats': cluster_stats,
            'converged': bool(self.model.converged_),
            'n_iter': int(self.model.n_iter_),
            'optimal_components_data': {
                'n_values': list(n_range),
                'bic_scores': bic_scores,
                'aic_scores': aic_scores
            },
            'X_scaled': X_scaled.tolist()
        }
