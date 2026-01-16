"""
ML Web Application - Main Flask Application
Author: AI Assistant
Description: Complete ML web app with 5 algorithms (Linear, Ridge, Lasso, K-Means, GMM)
Features: Automatic workflow, dark mode, CSRF protection, comprehensive analysis
"""

from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for, flash
from flask_wtf.csrf import CSRFProtect
import os
import sys
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
import json
from datetime import datetime
import secrets

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model handlers
from models.regression_models import LinearRegressionModel, RidgeModel, LassoModel
from models.clustering_models import KMeansModel, GMMModel

# Import utilities
from utils.data_processor import DataProcessor
from utils.visualization import VisualizationGenerator

# Initialize Flask app
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = Flask(__name__,
            template_folder=os.path.join(parent_dir, 'templates'),
            static_folder=os.path.join(parent_dir, 'static'))
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(16))
app.config['UPLOAD_FOLDER'] = os.path.join(parent_dir, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'csv'}
app.config['WTF_CSRF_TIME_LIMIT'] = None  # No time limit on CSRF tokens

# Initialize CSRF protection
csrf = CSRFProtect(app)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# ==================== ROUTES ====================

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Data upload page with automatic processing"""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded', 'error')
            return redirect(url_for('upload'))
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('upload'))
        
        if file and allowed_file(file.filename):
            try:
                # Save file
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                unique_filename = f"{timestamp}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(filepath)
                
                # Process data
                df = DataProcessor.read_csv(filepath)
                errors = DataProcessor.validate_data(df)
                
                if errors:
                    flash(f"Data validation errors: {'; '.join(errors)}", 'error')
                    return redirect(url_for('upload'))
                
                # Store filepath in session
                session['filepath'] = filepath
                session['filename'] = filename
                
                # Redirect to automatic analysis
                flash(f'File "{filename}" uploaded successfully! Analyzing...', 'success')
                return redirect(url_for('analyze'))
            
            except Exception as e:
                flash(f'Error processing file: {str(e)}', 'error')
                return redirect(url_for('upload'))
        
        flash('Invalid file type. Please upload a CSV file.', 'error')
        return redirect(url_for('upload'))
    
    return render_template('upload.html')


@app.route('/analyze')
def analyze():
    """Automatic analysis - runs all algorithms and shows results"""
    if 'filepath' not in session:
        flash('Please upload data first', 'error')
        return redirect(url_for('upload'))
    
    try:
        # Load data
        df = DataProcessor.read_csv(session['filepath'])
        
        # Get comprehensive insights
        insights = DataProcessor.get_data_insights(df)
        preview = DataProcessor.get_preview(df, n_rows=5)
        
        # Generate data visualizations
        data_viz = {
            'correlation': VisualizationGenerator.plot_correlation_heatmap(df),
            'distributions': VisualizationGenerator.plot_distribution_grid(df, max_features=6),
            'boxplots': VisualizationGenerator.plot_box_plots(df, max_features=6)
        }
        
        # Auto-detect problem type and run appropriate algorithms
        problem_type = insights['problem_type']
        all_results = {}
        
        # Run regression algorithms if applicable
        if problem_type in ['regression', 'both'] and insights['auto_target'] and insights['auto_features']:
            target_col = insights['auto_target']
            feature_cols = insights['auto_features']
            
            try:
                X, y, feature_names = DataProcessor.prepare_regression_data(df, target_col, feature_cols)
                
                # Run Linear Regression
                linear_model = LinearRegressionModel()
                all_results['linear'] = linear_model.train(X, y, {})
                all_results['linear']['feature_names'] = feature_names
                all_results['linear']['target_name'] = target_col
                
                # Run Ridge Regression
                ridge_model = RidgeModel()
                all_results['ridge'] = ridge_model.train(X, y, {'alpha': 1.0})
                all_results['ridge']['feature_names'] = feature_names
                all_results['ridge']['target_name'] = target_col
                
                # Run Lasso Regression
                lasso_model = LassoModel()
                all_results['lasso'] = lasso_model.train(X, y, {'alpha': 0.1})
                all_results['lasso']['feature_names'] = feature_names
                all_results['lasso']['target_name'] = target_col
                
            except Exception as e:
                flash(f'Regression analysis skipped: {str(e)}', 'warning')
        
        # Run clustering algorithms
        if problem_type in ['clustering', 'both'] or not all_results:
            try:
                cluster_features = insights['auto_features'] if insights['auto_features'] else insights['numeric_columns']
                X, feature_names = DataProcessor.prepare_clustering_data(df, cluster_features)
                
                # Auto-detect optimal clusters (between 2 and min(10, n_samples//5))
                optimal_k = min(5, max(2, len(X) // 10))
                
                # Run K-Means
                kmeans_model = KMeansModel()
                all_results['kmeans'] = kmeans_model.train(X, {'n_clusters': optimal_k})
                all_results['kmeans']['feature_names'] = feature_names
                
                # Run GMM
                gmm_model = GMMModel()
                all_results['gmm'] = gmm_model.train(X, {'n_components': optimal_k, 'covariance_type': 'full'})
                all_results['gmm']['feature_names'] = feature_names
                
            except Exception as e:
                flash(f'Clustering analysis skipped: {str(e)}', 'warning')
        
        # Generate visualizations for each algorithm
        all_viz = {}
        for algo_name, result in all_results.items():
            if algo_name in ['linear', 'ridge', 'lasso']:
                y_test = np.array(result['test_predictions']['y_true'])
                y_pred = np.array(result['test_predictions']['y_pred'])
                coefficients = np.array(result['coefficients'])
                feature_names = result.get('feature_names', [])
                
                all_viz[algo_name] = {
                    'actual_vs_predicted': VisualizationGenerator.plot_actual_vs_predicted(y_test, y_pred),
                    'residuals': VisualizationGenerator.plot_residuals(y_test, y_pred),
                    'coefficients': VisualizationGenerator.plot_coefficients(coefficients, feature_names)
                }
                
            elif algo_name in ['kmeans', 'gmm']:
                X_scaled = np.array(result['X_scaled'])
                labels = np.array(result['labels'])
                centers = np.array(result.get('centers', [])) if 'centers' in result else None
                
                all_viz[algo_name] = {
                    'clusters': VisualizationGenerator.plot_clustering_scatter(X_scaled, labels, centers),
                    'distribution': VisualizationGenerator.plot_cluster_distribution(result['cluster_stats'])
                }
                
                if 'elbow_data' in result:
                    all_viz[algo_name]['elbow'] = VisualizationGenerator.plot_elbow_curve(
                        result['elbow_data']['k_values'],
                        result['elbow_data']['inertias']
                    )
                
                if 'optimal_components_data' in result:
                    all_viz[algo_name]['bic_aic'] = VisualizationGenerator.plot_bic_aic(
                        result['optimal_components_data']['n_values'],
                        result['optimal_components_data']['bic_scores'],
                        result['optimal_components_data']['aic_scores']
                    )
        
        # Store in session (only store serializable data)
        session['insights'] = insights
        session['filepath'] = session.get('filepath')  # Keep filepath
        session['filename'] = session.get('filename')  # Keep filename
        # Note: Not storing all_results, all_viz in session due to size/serialization issues
        # They are passed directly to template instead
        
        # Render results page with everything
        return render_template('auto_results.html',
                             insights=insights,
                             preview=preview,
                             data_viz=data_viz,
                             all_results=all_results,
                             all_viz=all_viz)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        flash(f'Error during analysis: {str(e)}', 'error')
        return redirect(url_for('upload'))


@app.route('/select_algorithm')
def select_algorithm():
    """Algorithm selection page"""
    if 'filepath' not in session:
        return render_template('select_algorithm.html', error='Please upload data first')
    
    df = DataProcessor.read_csv(session['filepath'])
    data_info = DataProcessor.get_data_info(df)
    preview = DataProcessor.get_preview(df)
    
    return render_template('select_algorithm.html', data_info=data_info, preview=preview)


@app.route('/train_model', methods=['POST'])
def train_model():
    """Train selected model"""
    if 'filepath' not in session:
        flash('Please upload data first', 'error')
        return redirect(url_for('upload'))
    
    try:
        algorithm = request.form.get('algorithm')
        target_col = request.form.get('target_column')
        feature_cols = request.form.getlist('feature_columns') if request.form.getlist('feature_columns') else None
        cluster_features = request.form.getlist('cluster_features') if request.form.getlist('cluster_features') else None
        
        # Parameters
        alpha = float(request.form.get('alpha', 1.0)) if request.form.get('alpha') else 1.0
        n_clusters = int(request.form.get('n_clusters', 3)) if request.form.get('n_clusters') else 3
        n_components = int(request.form.get('n_components', 3)) if request.form.get('n_components') else 3
        covariance_type = request.form.get('covariance_type', 'full')
        
        # Load data
        df = DataProcessor.read_csv(session['filepath'])
        
        # Prepare data and train
        if algorithm in ['linear', 'ridge', 'lasso']:
            X, y, feature_names = DataProcessor.prepare_regression_data(df, target_col, feature_cols)
            params = {'alpha': alpha} if algorithm in ['ridge', 'lasso'] else {}
            
            if algorithm == 'linear':
                model = LinearRegressionModel()
            elif algorithm == 'ridge':
                model = RidgeModel()
            else:
                model = LassoModel()
            
            results = model.train(X, y, params)
            
            # Generate visualizations
            y_test = np.array(results['test_predictions']['y_true'])
            y_pred = np.array(results['test_predictions']['y_pred'])
            coefficients = np.array(results['coefficients'])
            
            plots = {
                'actual_vs_predicted': VisualizationGenerator.plot_actual_vs_predicted(y_test, y_pred),
                'residuals': VisualizationGenerator.plot_residuals(y_test, y_pred),
                'coefficients': VisualizationGenerator.plot_coefficients(coefficients, feature_names)
            }
            
        else:  # Clustering
            X, feature_names = DataProcessor.prepare_clustering_data(df, cluster_features)
            
            if algorithm == 'kmeans':
                model = KMeansModel()
                params = {'n_clusters': n_clusters}
            else:  # GMM
                model = GMMModel()
                params = {'n_components': n_components, 'covariance_type': covariance_type}
            
            results = model.train(X, params)
            
            # Generate visualizations
            X_scaled = np.array(results['X_scaled'])
            labels = np.array(results['labels'])
            centers = np.array(results['centers']) if 'centers' in results else None
            
            plots = {
                'clusters': VisualizationGenerator.plot_clustering_scatter(X_scaled, labels, centers),
                'distribution': VisualizationGenerator.plot_cluster_distribution(results['cluster_stats'])
            }
            
            if 'elbow_data' in results:
                plots['elbow'] = VisualizationGenerator.plot_elbow_curve(
                    results['elbow_data']['k_values'],
                    results['elbow_data']['inertias']
                )
                plots['silhouette'] = VisualizationGenerator.plot_silhouette_scores(
                    results['elbow_data']['k_values'],
                    results['elbow_data']['silhouettes']
                )
            
            if 'optimal_components_data' in results:
                plots['bic_aic'] = VisualizationGenerator.plot_bic_aic(
                    results['optimal_components_data']['n_values'],
                    results['optimal_components_data']['bic_scores'],
                    results['optimal_components_data']['aic_scores']
                )
        
        # Store in session
        session['results'] = results
        session['plots'] = plots
        session['algorithm'] = algorithm
        
        return render_template('results.html', results=results, plots=plots)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        flash(f'Error training model: {str(e)}', 'error')
        return redirect(url_for('select_algorithm'))


@app.route('/results')
def results():
    """Results page"""
    if 'results' not in session:
        flash('No results available', 'error')
        return redirect(url_for('upload'))
    
    return render_template('results.html', results=session['results'], plots=session.get('plots', {}))


@app.route('/download_results')
def download_results():
    """Download results as JSON - regenerate from current session data"""
    if 'filepath' not in session:
        flash('No data available. Please upload data first.', 'error')
        return redirect(url_for('upload'))
    
    try:
        import io
        
        # Regenerate results from current session data
        df = DataProcessor.read_csv(session['filepath'])
        insights = session.get('insights', DataProcessor.get_data_insights(df))
        
        # Prepare comprehensive results object
        results_data = {
            'metadata': {
                'filename': session.get('filename', 'unknown'),
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'application': 'AlgoLab ML Analysis'
            },
            'dataset_info': {
                'shape': insights['shape'],
                'memory_usage': insights['memory_usage'],
                'duplicates': insights['duplicates'],
                'missing_summary': insights['missing_summary']
            },
            'analysis': {
                'auto_detected_target': insights.get('auto_target'),
                'auto_selected_features': insights.get('auto_features'),
                'problem_type': insights.get('problem_type'),
                'numeric_columns': insights.get('numeric_columns', [])
            }
        }
        
        # Add statistics if available
        if 'statistics' in insights:
            results_data['statistics'] = insights['statistics']
        
        # Convert to JSON
        output = io.BytesIO()
        json_str = json.dumps(results_data, indent=2, default=str)
        output.write(json_str.encode('utf-8'))
        output.seek(0)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = session.get('filename', 'data').replace('.csv', '')
        
        return send_file(
            output,
            mimetype='application/json',
            as_attachment=True,
            download_name=f'algolab_results_{filename}_{timestamp}.json'
        )
    
    except Exception as e:
        flash(f'Error downloading results: {str(e)}', 'error')
        return redirect(url_for('upload'))


@app.route('/download_plot/<plot_name>')
def download_plot(plot_name):
    """Download plot as PNG"""
    if 'plots' not in session or plot_name not in session['plots']:
        return jsonify({'error': 'Plot not found'}), 404
    
    try:
        import io
        import base64
        
        # Get base64 plot
        plot_base64 = session['plots'][plot_name]
        
        # Remove data URI prefix if present
        if ',' in plot_base64:
            plot_base64 = plot_base64.split(',')[1]
        
        # Decode base64 to bytes
        plot_bytes = base64.b64decode(plot_base64)
        
        # Create BytesIO object
        output = io.BytesIO(plot_bytes)
        output.seek(0)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return send_file(
            output,
            mimetype='image/png',
            as_attachment=True,
            download_name=f'{plot_name}_{timestamp}.png'
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/compare')
def compare():
    """Algorithm comparison page"""
    return render_template('compare.html')


@app.route('/about')
def about():
    """About/Help page"""
    return render_template('about.html')


@app.route('/clear_session')
def clear_session():
    """Clear session data"""
    session.clear()
    flash('Session cleared successfully', 'success')
    return redirect(url_for('index'))


@app.route('/load_sample/<filename>')
def load_sample(filename):
    """Load sample dataset and redirect to analysis"""
    sample_folder = os.path.join(parent_dir, 'sample_data')
    source_filepath = os.path.join(sample_folder, filename)
    
    if not os.path.exists(source_filepath) or not allowed_file(filename):
        flash('Sample file not found', 'error')
        return redirect(url_for('upload'))
    
    try:
        # Copy sample file to uploads folder
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_sample_{filename}"
        dest_filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Copy file
        import shutil
        shutil.copy2(source_filepath, dest_filepath)
        
        # Validate
        df = DataProcessor.read_csv(dest_filepath)
        errors = DataProcessor.validate_data(df)
        
        if errors:
            flash(f"Sample data validation errors: {'; '.join(errors)}", 'error')
            return redirect(url_for('upload'))
        
        # Store in session
        session['filepath'] = dest_filepath
        session['filename'] = filename
        
        flash(f'Sample dataset "{filename}" loaded successfully!', 'success')
        return redirect(url_for('analyze'))
    
    except Exception as e:
        flash(f'Error loading sample data: {str(e)}', 'error')
        return redirect(url_for('upload'))


# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500


# ==================== MAIN ====================

if __name__ == '__main__':
    print("="*60)
    print("ML Web Application Starting...")
    print("="*60)
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False') == 'True'
    app.run(debug=debug, host='0.0.0.0', port=port)