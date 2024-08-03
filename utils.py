import pickle
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, roc_curve, auc, roc_auc_score, precision_recall_curve
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def clean_text(text):
    """Preprocesses the input text data by performing several cleaning operations."""
    
    text = str(text).lower()
    text = re.sub(r'\(.*', ' ', text)
    text = text.strip()
    text = re.sub(r'[ -]', '_', text)
    
    return text
    

def cap_values(column, feature):
    """
    Cap the values of a numeric column based on predefined lower and upper bounds.
    
    Args:
        column (pd.Series): A pandas Series representing the numeric column to be capped.
        feature (str): The name of the feature for which to apply the bounds.

    Returns:
        pd.Series: The input column with values clipped to fall within the specified bounds.
    """
    
    with open(r'.\dumps\lower_upper_bounds.pkl', 'rb') as f:
        lower_upper_bounds = pickle.load(f)
    
    lower_bound, upper_bound = lower_upper_bounds[feature]

    return column.clip(lower=lower_bound, upper=upper_bound)


def categorize_value(value):
    """
    Categorize a numerical value into one of several predefined categories.
    This function assigns a categorical label to a given numerical value based on specified thresholds.

    Args:
        value (int): A numerical value to be categorized. It is expected to be an integer.

    Returns:
        str: A string representing the category of the value. Possible categories are:
             - "zero" for values equal to 0
             - "low" for values greater than 0 and up to 199
             - "medium" for values greater than 199 and up to 239
             - "high" for values greater than 239
    """
    
    if value == 0:
        return "zero"
    elif 0 < value <= 199:
        return "low"
    elif 199 < value <= 239:
        return "medium"
    elif value > 239:
        return "high"


def calculate_risk_score(row):
    """
    Calculate a risk score based on individual health metrics provided in a dictionary.

    This function computes a risk score by evaluating various health-related factors from the provided
    dictionary and assigning points based on predefined criteria. The total score is calculated by 
    adding points for each condition that is met.

    Args:
        row (dict): A dictionary containing health metrics with the following expected keys:
            - 'Sex' (str): Gender of the individual ('m' for male, 'f' for female).
            - 'ChestPainType' (str): Type of chest pain ('asy' for asymptomatic, etc.).
            - 'FastingBS' (int or str): Fasting blood sugar level ('1' or 1 for positive, otherwise negative).
            - 'ExerciseAngina' (str): Exercise-induced angina ('y' for yes, 'n' for no).
            - 'ST_Slope' (str): Slope of the ST segment ('flat', 'up', or 'down').
            - 'New_Cholesterol_Bin' (str): Binned cholesterol level ('zero', 'low', 'medium', 'high').

    Returns:
        int: The computed risk score based on the input metrics. The score is a sum of points assigned
             according to the specified criteria.
    """
    
    score = 0
    if row['Sex'] == 'm':
        score += 2
    if row['ChestPainType'] == 'asy':
        score += 3
    if row['FastingBS'] == '1' or row['FastingBS'] == 1:
        score += 2
    if row['ExerciseAngina'] == 'y':
        score += 3
    if row['ST_Slope'] == 'flat':
        score += 3
    if row['New_Cholesterol_Bin'] == 'zero':
        score += 3
    
    return score


def preprocess_data(df:pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the input DataFrame by cleaning categorical features, transforming numerical features,
    creating new features, and applying encoding to prepare the data for model input.

    This function performs the following steps:
    1. Cleans categorical features by applying text preprocessing.
    2. Transforms numerical features using predefined value capping.
    3. Creates new features based on existing data.
    4. Maps categorical features to numerical values.
    5. Applies one-hot encoding to specific categorical features.
    6. Concatenates processed features into a final DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the raw data to be preprocessed.

    Returns:
        pd.DataFrame: The preprocessed DataFrame ready for model input, including both the transformed 
                      numerical features and encoded categorical features.
    """
    
    # clean the categorical features if not cleaned
    with open(r'.\dumps\categorical_columns.pkl', 'rb') as f:
        categorical_cols = pickle.load(f)

    for feature in categorical_cols:
        df[feature] = df[feature].apply(clean_text)


    # Numeric variables
    with open(r'.\dumps\numeric_columns.pkl', 'rb') as f:
        numeric_cols = pickle.load(f)
    
    for feature in numeric_cols:
        df[feature] = cap_values(df[feature], feature)

    # FE features
    df['New_Cholesterol_Bin'] = df['Cholesterol'].map(categorize_value)
    df['New_Risk_Score'] = df.apply(calculate_risk_score, axis=1)

    num_FE_features = ['New_Risk_Score']
    cat_FE_features = ['New_Cholesterol_Bin']

    
    # Categorical variables 
    with open(r'.\dumps\one_hot_features.pkl', 'rb') as f:
        one_hot_features = pickle.load(f)

    with open(r'.\dumps\mapping_features.pkl', 'rb') as f:
        mapping_features = pickle.load(f)


    # Mapping features
    with open(r'.\dumps\mapping.pkl', 'rb') as f:
        mapping = pickle.load(f)
    for feature in mapping_features:
        df[feature] = df[feature].map(mapping).astype(int)

    # One-hot features
    with open(r'.\dumps\one_hot_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    ohe_features_encoded = pd.DataFrame(encoder.transform(df[one_hot_features]), columns= encoder.get_feature_names_out())
    ohe_features_encoded = ohe_features_encoded.astype(int)

    df_imb = pd.concat([df[numeric_cols + num_FE_features + mapping_features], ohe_features_encoded], axis=1)
    
    return df_imb


def load_models():
    
    models = { 
        # Logistic Regression model
        "Logistic Regression": LogisticRegression(),
    
        # Naive Bayes model
        "Naive Bayes": GaussianNB(),

        # Support Vector Machine 
        "SVC": SVC(),
       
        # Decision Tree model
        "Decision Tree Classifier": DecisionTreeClassifier(),
    
        # Random Forest model
        "Random Forest": RandomForestClassifier(),
    
        # LightGBM model
        "LightGBM": LGBMClassifier()
    }

    return models
    

def load_model_params():
        
    model_params = {
        'Logistic Regression' : {
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'C': [0.01, 0.1, 1, 10, 100, 500, 1000]},
    
        'Naive Bayes' : {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]},
    
        'SVC' : {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid']},
    
        'Decision Tree Classifier' : {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': range(2, 10)},
        
        'Random Forest' : {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]},
    
        'LightGBM' : {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [31, 50, 100],
            'boosting_type': ['gbdt', 'dart']}
    }

    return model_params


def train_models(train_data:str, test_data:str, method='random', tune=False, models:dict=None) -> dict:

    """
    models to be passed as dictionary with name as key and model as value
    models = { 
        "Logistic Regression": LogisticRegression(), 
        "Naive Bayes": GaussianNB()
    """


    with open(r'.\dumps\train_feature_sets.pkl', 'rb') as f:
        train_feature_sets = pickle.load(f)
    with open(r'.\dumps\test_feature_sets.pkl', 'rb') as f:
        test_feature_sets = pickle.load(f)
    
    X_train, y_train = train_feature_sets[train_data]
    X_test, y_test = test_feature_sets[test_data]

    if models is None:
        models = load_models()
    
    results = []
    for name, model in tqdm(models.items()):

        if tune:
            model_params = load_model_params()
            model = tune_model(method, model, param_grid = model_params[name], X_train=X_train, y_train=y_train, 
                           scoring='accuracy', n_iter=10, cv=5, random_state=42)
        
        # Model training
        model.fit(X_train, y_train)
        y_tr_pred = model.predict(X_train)
        y_te_pred = model.predict(X_test)
        
        # Model Evaluation
        train_f1 = f1_score(y_train, y_tr_pred, average='weighted')
        test_f1 = f1_score(y_test, y_te_pred, average='weighted')

        train_accuracy = accuracy_score(y_train, y_tr_pred)
        test_accuracy = accuracy_score(y_test, y_te_pred)

        # append the scores
        results.append(
            {'Model': name, 
            'train_data': train_data, 'test_data': test_data,
            'train_f1':round(train_f1,4), 'test_f1':test_f1.round(4),
             'train_accuracy' : round(train_accuracy,4), 'test_accuracy': round(test_accuracy,4)
            }
        )

    return results



def evaluate_model(model, X_train, y_train, X_test, y_test):

    model.fit(X_train, y_train)
    y_tr_pred = model.predict(X_train)
    y_te_pred = model.predict(X_test)
        
    # Model Evaluation

    # f1-score
    train_f1 = f1_score(y_train, y_tr_pred, average='micro')
    test_f1 = f1_score(y_test, y_te_pred, average='micro')
    
    print('train_f1:', round(train_f1,4), 'test_f1:', round(test_f1,4), '\n')

    # Classification report
    report = classification_report(y_train, y_tr_pred)
    print('Classification Report on train data:\n', report, '\n')

    report = classification_report(y_test, y_te_pred)
    print('Classification Report on test data:\n', report)
    
    # confusion matrix
    cm_tr = confusion_matrix(y_train, y_tr_pred)
    cm_tr_df = pd.DataFrame(cm_tr) #, index=, columns=iris.target_names)
    
    cm_te = confusion_matrix(y_test, y_te_pred)
    cm_te_df = pd.DataFrame(cm_te) #, index=, columns=iris.target_names)

    fig, axs = plt.subplots(1,2, figsize=(9,3))
    sns.heatmap(cm_tr_df, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axs[0])
    axs[0].set_title('Train Confusion Matrix')
    axs[0].set_xlabel('Predicted Label')
    axs[0].set_ylabel('True Label')
        
    sns.heatmap(cm_te_df, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axs[1])
    axs[1].set_title('Test Confusion Matrix')
    axs[1].set_xlabel('Predicted Label')
    axs[1].set_ylabel('True Label')
    print('\n')

    ## roc_auc curve
    y_tr_proba = model.predict_proba(X_train)[:,1]
    y_te_proba = model.predict_proba(X_test)[:,1]

    # Calculate ROC curve and ROC AUC for train data
    fpr_train, tpr_train, _ = roc_curve(y_train, y_tr_proba)
    roc_auc_train = roc_auc_score(y_train, y_tr_proba)
    
    # Calculate ROC curve and ROC AUC for test data
    fpr_test, tpr_test, _ = roc_curve(y_test, y_te_proba)
    roc_auc_test = roc_auc_score(y_test, y_te_proba)
    
    # Plot ROC Curves
    plt.figure(figsize=(6, 4))
    plt.plot(fpr_train, tpr_train, label=f'Train ROC AUC = {roc_auc_train:.2f}')
    plt.plot(fpr_test, tpr_test, label=f'Test ROC AUC = {roc_auc_test:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    
    ## precision_recall curve
    y_tr_proba = model.predict_proba(X_train)[:,1]
    y_te_proba = model.predict_proba(X_test)[:,1]

    # Calculate Precision-Recall curve for train data
    precision_train, recall_train, _ = precision_recall_curve(y_train, y_tr_proba)
    auc_pr_train = auc(recall_train, precision_train)
    
    # Calculate Precision-Recall curve for test data
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_te_proba)
    auc_pr_test = auc(recall_test, precision_test)
    
    # Plot Precision-Recall Curves
    plt.figure(figsize=(6, 4))
    plt.plot(recall_train, precision_train, label=f'Train AUC = {auc_pr_train:.2f}')
    plt.plot(recall_test, precision_test, label=f'Test AUC = {auc_pr_test:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.show()


def tune_model(method, model, param_grid, X_train, y_train, scoring='accuracy', n_iter=10, cv=5, random_state=42):

    """Tune a model using either GridSearchCV or RandomizedSearchCV and return the best model."""

    # Choose the tuning method based on the 'method' parameter
    if method == 'grid':
        search = GridSearchCV(model, param_grid, scoring=scoring, refit='accuracy', cv=cv, n_jobs=-1, verbose=1)
    elif method == 'random':
        search = RandomizedSearchCV(model, param_distributions=param_grid, scoring=scoring, refit='accuracy', n_iter=n_iter, cv=cv, n_jobs=-1, verbose=1, random_state=random_state)
    else:
        raise ValueError("Invalid method specified. Use 'grid' for GridSearchCV or 'random' for RandomizedSearchCV.")

    # Fit the search to the training data
    search.fit(X_train, y_train)

    # Return the best model
    return search.best_estimator_

