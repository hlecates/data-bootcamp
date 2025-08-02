import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                           precision_score, recall_score, f1_score, roc_curve, auc)
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
    
    print("\nTarget variable distribution:")
    print(df['class'].value_counts())
    
    return df

def prepare_features(df):
    # Select numerical features for modeling
    numerical_features = ['MW', 'LogP', 'NumHDonors', 'NumHAcceptors', 'pIC50', 'Druglikeness']
    
    # Check for missing values in selected features
    print("Missing values in selected features:")
    print(df[numerical_features].isnull().sum())
    
    # Handle missing values
    df_clean = df[numerical_features + ['class']].dropna()
    print(f"\nShape after removing missing values: {df_clean.shape}")
    
    # Encode target variable
    le = LabelEncoder()
    df_clean['class_encoded'] = le.fit_transform(df_clean['class'])
    
    print("\nClass encoding:")
    for i, class_name in enumerate(le.classes_):
        print(f"{class_name}: {i}")
    
    # Separate features and target
    X = df_clean[numerical_features]
    y = df_clean['class_encoded']
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    return X, y, numerical_features, le

def basic_classification_comparison(X, y, feature_names):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB()
    }
    
    # Train and evaluate models
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        if name in ['SVM', 'KNN']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-Score: {f1:.3f}")
    
    return results, X_test, y_test

def plot_results_comparison(results):
    # Extract metrics for plotting
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    precisions = [results[model]['precision'] for model in models]
    recalls = [results[model]['recall'] for model in models]
    f1_scores = [results[model]['f1'] for model in models]
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Accuracy
    axes[0, 0].bar(models, accuracies, color='skyblue')
    axes[0, 0].set_title('Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Precision
    axes[0, 1].bar(models, precisions, color='lightgreen')
    axes[0, 1].set_title('Precision Comparison')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Recall
    axes[1, 0].bar(models, recalls, color='lightcoral')
    axes[1, 0].set_title('Recall Comparison')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # F1-Score
    axes[1, 1].bar(models, f1_scores, color='gold')
    axes[1, 1].set_title('F1-Score Comparison')
    axes[1, 1].set_ylabel('F1-Score')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    plt.close()

def detailed_model_analysis(X, y, feature_names):
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Use Random Forest for detailed analysis
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    y_proba = rf_model.predict_proba(X_test)
    
    # Print detailed classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
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
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    plt.close()

def hyperparameter_tuning(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
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
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
    
    # Evaluate best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy with best model: {accuracy:.3f}")
    
    return best_model

def cross_validation_analysis(X, y):
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    # Perform cross-validation
    cv_results = {}
    
    for name, model in models.items():
        if name in ['SVM', 'KNN']:
            scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
        else:
            scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        
        cv_results[name] = scores
        print(f"{name}: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
    
    # Plot cross-validation results
    plt.figure(figsize=(10, 6))
    plt.boxplot([cv_results[name] for name in models.keys()], labels=models.keys())
    plt.title('Cross-Validation Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.close()

def practical_insights(X, y, feature_names):
    # Train a Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    
    # Feature importance analysis
    importances = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("Feature Importance Ranking:")
    print(feature_importance_df)
    
    # Analyze feature distributions by class
    X_df = pd.DataFrame(X, columns=feature_names)
    X_df['class'] = y
    
    print("\nFeature Statistics by Class:")
    for feature in feature_names:
        print(f"\n{feature}:")
        print(X_df.groupby('class')[feature].describe())
    
    # Visualize feature distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, feature in enumerate(feature_names):
        if i < 6:  # Limit to 6 features for visualization
            X_df.boxplot(column=feature, by='class', ax=axes[i])
            axes[i].set_title(f'{feature} by Class')
            axes[i].set_xlabel('Class')
    
    plt.tight_layout()
    plt.show()
    plt.close()

if __name__ == "__main__":
    print("Classification Examples with Kinase Dataset")
    print("=" * 50)
    
    # Load and explore data
    df = load_and_explore_data()
    
    # Prepare features
    X, y, feature_names, label_encoder = prepare_features(df)
    
    # Basic classification comparison
    results, X_test, y_test = basic_classification_comparison(X, y, feature_names)
    
    # Plot results comparison
    plot_results_comparison(results)
    
    # Detailed model analysis
    detailed_model_analysis(X, y, feature_names)
    
    # Hyperparameter tuning
    best_model = hyperparameter_tuning(X, y)
    
    # Cross-validation analysis
    cross_validation_analysis(X, y)
    
    # Practical insights
    practical_insights(X, y, feature_names)
    
    print("\n" + "=" * 50)
    print("All classification demonstrations completed!") 