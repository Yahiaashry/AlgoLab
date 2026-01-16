"""
Regression Models Handler
Implements Linear Regression, Ridge, and Lasso
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


class BaseRegressionModel:
    """Base class for regression models"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def split_and_scale(self, X, y, test_size=0.2, random_state=42):
        """Split and scale data"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate regression metrics"""
        return {
            'r2': float(r2_score(y_true, y_pred)),
            'mse': float(mean_squared_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae': float(mean_absolute_error(y_true, y_pred))
        }


class LinearRegressionModel(BaseRegressionModel):
    """Linear Regression Model"""
    
    def train(self, X, y, params=None):
        """Train linear regression model"""
        if params is None:
            params = {}
        
        test_size = params.get('test_size', 0.2)
        
        # Split and scale
        X_train, X_test, y_train, y_test = self.split_and_scale(X, y, test_size)
        
        # Train model
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Metrics
        train_metrics = self.calculate_metrics(y_train, y_train_pred)
        test_metrics = self.calculate_metrics(y_test, y_test_pred)
        
        return {
            'algorithm': 'Linear Regression',
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'coefficients': self.model.coef_.tolist(),
            'intercept': float(self.model.intercept_),
            'test_predictions': {
                'y_true': y_test.tolist(),
                'y_pred': y_test_pred.tolist()
            },
            'train_predictions': {
                'y_true': y_train.tolist(),
                'y_pred': y_train_pred.tolist()
            }
        }


class RidgeModel(BaseRegressionModel):
    """Ridge Regression Model"""
    
    def train(self, X, y, params=None):
        """Train ridge regression model"""
        if params is None:
            params = {}
        
        test_size = params.get('test_size', 0.2)
        alpha = params.get('alpha', 1.0)
        
        # Split and scale
        X_train, X_test, y_train, y_test = self.split_and_scale(X, y, test_size)
        
        # Train model
        self.model = Ridge(alpha=alpha, max_iter=10000)
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Metrics
        train_metrics = self.calculate_metrics(y_train, y_train_pred)
        test_metrics = self.calculate_metrics(y_test, y_test_pred)
        
        return {
            'algorithm': 'Ridge Regression',
            'alpha': alpha,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'coefficients': self.model.coef_.tolist(),
            'intercept': float(self.model.intercept_),
            'test_predictions': {
                'y_true': y_test.tolist(),
                'y_pred': y_test_pred.tolist()
            },
            'train_predictions': {
                'y_true': y_train.tolist(),
                'y_pred': y_train_pred.tolist()
            },
            'max_coef': float(np.max(np.abs(self.model.coef_))),
            'mean_coef': float(np.mean(np.abs(self.model.coef_)))
        }


class LassoModel(BaseRegressionModel):
    """Lasso Regression Model"""
    
    def train(self, X, y, params=None):
        """Train lasso regression model"""
        if params is None:
            params = {}
        
        test_size = params.get('test_size', 0.2)
        alpha = params.get('alpha', 1.0)
        
        # Split and scale
        X_train, X_test, y_train, y_test = self.split_and_scale(X, y, test_size)
        
        # Train model
        self.model = Lasso(alpha=alpha, max_iter=10000)
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Metrics
        train_metrics = self.calculate_metrics(y_train, y_train_pred)
        test_metrics = self.calculate_metrics(y_test, y_test_pred)
        
        # Feature selection
        non_zero_features = int(np.sum(self.model.coef_ != 0))
        zero_features = int(np.sum(self.model.coef_ == 0))
        
        return {
            'algorithm': 'Lasso Regression',
            'alpha': alpha,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'coefficients': self.model.coef_.tolist(),
            'intercept': float(self.model.intercept_),
            'test_predictions': {
                'y_true': y_test.tolist(),
                'y_pred': y_test_pred.tolist()
            },
            'train_predictions': {
                'y_true': y_train.tolist(),
                'y_pred': y_train_pred.tolist()
            },
            'non_zero_features': non_zero_features,
            'zero_features': zero_features,
            'feature_selection': {
                'n_selected': non_zero_features,
                'n_total': non_zero_features + zero_features,
                'selected_features': [],
                'eliminated': zero_features
            }
        }
