import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# Assume X_train, X_test, y_train, y_test are already defined

# Define models and their hyperparameters
models = {
    'RandomForest': RandomForestClassifier(),
    'SVC': SVC(),
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'KNeighbors': KNeighborsClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
    'AdaBoost': AdaBoostClassifier()
}

# Hyperparameters for Grid Search
param_grid = {
    'RandomForest': {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20]
    },
    'SVC': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    },
    'LogisticRegression': {
        'C': [0.1, 1, 10],
        'solver': ['lbfgs', 'liblinear']
    },
    'KNeighbors': {
        'n_neighbors': [3, 5, 10],
        'weights': ['uniform', 'distance']
    },
    'GradientBoosting': {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'AdaBoost': {
        'n_estimators': [50, 100],
        'learning_rate': [1.0, 0.5, 0.1]
    }
}

# DataFrame to store results
results = pd.DataFrame(columns=['Model', 'Accuracy Train', 'Accuracy Test', 'Best Parameters'])

# Train and evaluate each model
for model_name, model in models.items():
    # Perform Grid Search for hyperparameter tuning
    grid_search = GridSearchCV(model, param_grid[model_name], cv=5)
    grid_search.fit(X_train.toarray(), y_train)
    
    # Best model
    best_model = grid_search.best_estimator_
    
    # Predictions
    y_pred_train = best_model.predict(X_train.toarray())
    y_pred_test = best_model.predict(X_test.toarray())
    
    # Calculate accuracies
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    
    # Store results
    results = results.append({
        'Model': model_name,
        'Accuracy Train': accuracy_train,
        'Accuracy Test': accuracy_test,
        'Best Parameters': grid_search.best_params_
    }, ignore_index=True)

# Display results
print(results)