import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                           explained_variance_score)
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data():
    # Load the dataset
    df = pd.read_csv('../data/kinase.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nData types:")
    print(df.dtypes)
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
    print("\nTarget variable (pIC50) statistics:")
    print(df['pIC50'].describe())
    
    return df

def prepare_features(df):
    # Select numerical features for modeling
    numerical_features = ['MW', 'LogP', 'NumHDonors', 'NumHAcceptors', 'Druglikeness']
    
    # Check for missing values in selected features
    print("Missing values in selected features:")
    print(df[numerical_features].isnull().sum())
    
    # Handle missing values
    df_clean = df[numerical_features + ['pIC50']].dropna()
    print(f"\nShape after removing missing values: {df_clean.shape}")
    
    # Separate features and target
    X = df_clean[numerical_features]
    y = df_clean['pIC50']
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    print("\nFeature correlations with pIC50:")
    correlations = df_clean.corr()['pIC50'].sort_values(ascending=False)
    print(correlations)
    
    return X, y, numerical_features

def basic_regression_comparison(X, y, feature_names):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0, random_state=42),
        'Lasso Regression': Lasso(alpha=0.1, random_state=42),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1)
    }
    
    # Train and evaluate models
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        if name == 'SVR':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'y_pred': y_pred
        }
        
        print(f"MSE: {mse:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"MAE: {mae:.3f}")
        print(f"R²: {r2:.3f}")
    
    return results, X_test, y_test

def plot_results_comparison(results):
    # Extract metrics for plotting
    models = list(results.keys())
    rmse_scores = [results[model]['rmse'] for model in models]
    mae_scores = [results[model]['mae'] for model in models]
    r2_scores = [results[model]['r2'] for model in models]
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # RMSE
    axes[0].bar(models, rmse_scores, color='skyblue')
    axes[0].set_title('RMSE Comparison')
    axes[0].set_ylabel('RMSE')
    axes[0].tick_params(axis='x', rotation=45)
    
    # MAE
    axes[1].bar(models, mae_scores, color='lightgreen')
    axes[1].set_title('MAE Comparison')
    axes[1].set_ylabel('MAE')
    axes[1].tick_params(axis='x', rotation=45)
    
    # R²
    axes[2].bar(models, r2_scores, color='lightcoral')
    axes[2].set_title('R² Comparison')
    axes[2].set_ylabel('R²')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    plt.close()

def detailed_model_analysis(X, y, feature_names):
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Use Random Forest for detailed analysis
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    
    # Print detailed metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Detailed Metrics:")
    print(f"MSE: {mse:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"R²: {r2:.3f}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual pIC50')
    plt.ylabel('Predicted pIC50')
    plt.title('Actual vs Predicted pIC50')
    plt.tight_layout()
    plt.show()
    plt.close()
    
    # Plot residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted pIC50')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.tight_layout()
    plt.show()
    plt.close()
    
    # Plot feature importance
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance')
    plt.bar(range(X.shape[1]), importances[indices])
    plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45)
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.show()
    plt.close()

def polynomial_regression_demo(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    print(f"Original features: {X_train.shape[1]}")
    print(f"Polynomial features: {X_train_poly.shape[1]}")
    
    # Train polynomial regression
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    y_pred_poly = poly_model.predict(X_test_poly)
    
    # Compare with linear regression
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred_linear = linear_model.predict(X_test)
    
    # Calculate metrics
    poly_r2 = r2_score(y_test, y_pred_poly)
    linear_r2 = r2_score(y_test, y_pred_linear)
    
    print(f"Linear Regression R²: {linear_r2:.3f}")
    print(f"Polynomial Regression R²: {poly_r2:.3f}")
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred_linear, alpha=0.6, label='Linear')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual pIC50')
    plt.ylabel('Predicted pIC50')
    plt.title('Linear Regression')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred_poly, alpha=0.6, label='Polynomial')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual pIC50')
    plt.ylabel('Predicted pIC50')
    plt.title('Polynomial Regression')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    plt.close()

def regularization_comparison(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test different alpha values
    alphas = [0.001, 0.01, 0.1, 1, 10, 100]
    
    ridge_scores = []
    lasso_scores = []
    
    for alpha in alphas:
        # Ridge regression
        ridge = Ridge(alpha=alpha, random_state=42)
        ridge.fit(X_train_scaled, y_train)
        ridge_pred = ridge.predict(X_test_scaled)
        ridge_scores.append(r2_score(y_test, ridge_pred))
        
        # Lasso regression
        lasso = Lasso(alpha=alpha, random_state=42)
        lasso.fit(X_train_scaled, y_train)
        lasso_pred = lasso.predict(X_test_scaled)
        lasso_scores.append(r2_score(y_test, lasso_pred))
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.semilogx(alphas, ridge_scores, 'o-', label='Ridge')
    plt.semilogx(alphas, lasso_scores, 's-', label='Lasso')
    plt.xlabel('Alpha (Regularization Strength)')
    plt.ylabel('R² Score')
    plt.title('Regularization Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()
    
    # Print best results
    best_ridge_alpha = alphas[np.argmax(ridge_scores)]
    best_lasso_alpha = alphas[np.argmax(lasso_scores)]
    print(f"Best Ridge alpha: {best_ridge_alpha}")
    print(f"Best Lasso alpha: {best_lasso_alpha}")

def hyperparameter_tuning(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Grid search for Random Forest
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10]
    }
    
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {-grid_search.best_score_:.3f}")
    
    # Evaluate best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"Test R² with best model: {r2:.3f}")
    print(f"Test RMSE with best model: {rmse:.3f}")
    
    return best_model

def cross_validation_analysis(X, y):
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1)
    }
    
    # Perform cross-validation
    cv_results = {}
    
    for name, model in models.items():
        if name == 'SVR':
            scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
        else:
            scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        
        cv_results[name] = scores
        print(f"{name}: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
    
    # Plot cross-validation results
    plt.figure(figsize=(10, 6))
    plt.boxplot([cv_results[name] for name in models.keys()], labels=models.keys())
    plt.title('Cross-Validation R² Comparison')
    plt.ylabel('R² Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.close()

def practical_insights(X, y, feature_names):
    # Train a Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    
    # Feature importance analysis
    importances = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("Feature Importance Ranking:")
    print(feature_importance_df)
    
    # Analyze feature relationships with target
    X_df = pd.DataFrame(X, columns=feature_names)
    X_df['pIC50'] = y
    
    print("\nFeature Correlations with pIC50:")
    correlations = X_df.corr()['pIC50'].sort_values(ascending=False)
    print(correlations)
    
    # Visualize feature relationships
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, feature in enumerate(feature_names):
        if i < 6:  # Limit to 6 features for visualization
            axes[i].scatter(X_df[feature], X_df['pIC50'], alpha=0.6)
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('pIC50')
            axes[i].set_title(f'{feature} vs pIC50')
    
    plt.tight_layout()
    plt.show()
    plt.close()

if __name__ == "__main__":
    print("Regression Examples with Kinase Dataset")
    print("=" * 50)
    
    # Load and explore data
    df = load_and_explore_data()
    
    # Prepare features
    X, y, feature_names = prepare_features(df)
    
    # Basic regression comparison
    results, X_test, y_test = basic_regression_comparison(X, y, feature_names)
    
    # Plot results comparison
    plot_results_comparison(results)
    
    # Detailed model analysis
    detailed_model_analysis(X, y, feature_names)
    
    # Polynomial regression demo
    polynomial_regression_demo(X, y)
    
    # Regularization comparison
    regularization_comparison(X, y)
    
    # Hyperparameter tuning
    best_model = hyperparameter_tuning(X, y)
    
    # Cross-validation analysis
    cross_validation_analysis(X, y)
    
    # Practical insights
    practical_insights(X, y, feature_names)
    
    print("\n" + "=" * 50)
    print("All regression demonstrations completed!") 