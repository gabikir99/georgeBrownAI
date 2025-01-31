from sklearn.feature_selection import SelectKBest, chi2, RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np
import warnings
import re

def cor_selector(X, y,num_feats):
    cor_list = []
    for i in X.columns:
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(abs(cor))
    cor_list = np.nan_to_num(cor_list)
    feature_indices = np.argsort(cor_list)[-num_feats:]
    cor_support = [i in feature_indices for i in range(len(X.columns))]
    cor_feature = X.columns[feature_indices].tolist()
    return cor_support, cor_feature

def chi_squared_selector(X, y, num_feats):
    X_norm = MinMaxScaler().fit_transform(X)
    chi_selector = SelectKBest(chi2, k=num_feats)
    chi_selector.fit(X_norm, y)
    chi_support = chi_selector.get_support()
    chi_feature = X.loc[:, chi_support].columns.tolist()
    return chi_support, chi_feature

def rfe_selector(X, y, num_feats):
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(solver='lbfgs')
    rfe_selector = RFE(estimator=model, n_features_to_select=num_feats, step=10,verbose=5)
    rfe_selector.fit(X, y)
    rfe_support = rfe_selector.get_support()
    rfe_feature = X.columns[rfe_support].tolist()
    return rfe_support, rfe_feature

def embedded_log_reg_selector(X, y, num_feats):
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)

    embedded_lr_selector = SelectFromModel(LogisticRegression(penalty='l2', solver='liblinear'), max_features=num_feats)
    embedded_lr_selector.fit(X_norm, y)

    embedded_lr_support = embedded_lr_selector.get_support()
    embedded_lr_feature = X.loc[:, embedded_lr_support].columns.tolist()

    return embedded_lr_support, embedded_lr_feature

def embedded_rf_selector(X, y, num_feats):
    embedded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=50), max_features=num_feats) 
    embedded_rf_selector.fit(X, y)

    embedded_rf_support = embedded_rf_selector.get_support()
    embedded_rf_feature = X.loc[:, embedded_rf_support].columns.tolist()
    
    return embedded_rf_support, embedded_rf_feature

def embedded_lgbm_selector(X, y, num_feats):
    lgbc = LGBMClassifier(
        n_estimators=200, 
        learning_rate=0.05, 
        num_leaves=32, 
        colsample_bytree=0.2, 
        reg_alpha=3, 
        reg_lambda=1, 
        min_split_gain=0.01, 
        min_child_weight=40
    )

    embedded_lgbm_selector = SelectFromModel(lgbc, max_features=num_feats)
    embedded_lgbm_selector.fit(X, y)

    embedded_lgbm_support = embedded_lgbm_selector.get_support()
    embedded_lgbm_feature = X.loc[:, embedded_lgbm_support].columns.tolist()
    
    return embedded_lgbm_support, embedded_lgbm_feature

def preprocess_dataset(dataset_path):
    df = pd.read_csv(dataset_path).sample(1000) 
    df = df.drop_duplicates()  

    for col in df.columns:
        if df[col].dtype == 'object':  
            df[col] = df[col].fillna(df[col].mode()[0])  
        else:  
            df[col] = df[col].fillna(df[col].median())  
 
    target_col = df.columns[-1]  
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)  

    # **Use Label Encoding instead of One-Hot Encoding for categorical features**
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        X[col] = LabelEncoder().fit_transform(X[col])   
    
    X.columns = [re.sub(r'[^A-Za-z0-9_]', '_', col) for col in X.columns]
    num_feats = min(30, X.shape[1])  

    print(f"Final dataset shape: {X.shape}") 
    
    return X, y, num_feats

def autoFeatureSelector(dataset_path, methods=[]):
    # Parameters
    # data - dataset to be analyzed (csv file)
    # methods - various feature selection methods we outlined before, use them all here (list)
    
    # preprocessing
    X, y, num_feats = preprocess_dataset(dataset_path)

    print("Starting feature selection...")  # Debugging
    
    feature_dict = {}
    
    # Run every method we outlined above from the methods list and collect returned best features from every method
    if 'pearson' in methods:
        print("Running Pearson Correlation...")
        feature_dict['Pearson'] = cor_selector(X, y, num_feats)[0]
        print("Pearson selection complete.")
    if 'chi-square' in methods:
        print("Running Chi-Square Test...")
        feature_dict['Chi-2'] = chi_squared_selector(X, y, num_feats)[0]
        print("Chi-Square selection complete.")
    if 'rfe' in methods:
        print("Running Recursive Feature Elimination (RFE)...")
        feature_dict['RFE'] = rfe_selector(X, y, num_feats)[0]
        print("RFE selection complete.")
    if 'log-reg' in methods:
        print("Running Logistic Regression Feature Selection...")
        feature_dict['Logistic Regression'] = embedded_log_reg_selector(X, y, num_feats)[0]
        print("Logistic Regression selection complete.")
    if 'rf' in methods:
        print("Running Random Forest Feature Selection...")
        feature_dict['Random Forest'] = embedded_rf_selector(X, y, num_feats)[0]
        print("Random Forest selection complete.")
    if 'lgbm' in methods:
        print("Running LightGBM Feature Selection...")
        feature_dict['LightGBM'] = embedded_lgbm_selector(X, y, num_feats)[0]
        print("LightGBM selection complete.")
    
    
    # Combine all the above feature list and count the maximum set of features that got selected by all methods
    #### Your Code starts here (Multiple lines)
    feature_selection_df = pd.DataFrame(feature_dict, index=X.columns)
    feature_selection_df["Total"] = feature_selection_df.sum(axis=1)
    feature_selection_df = feature_selection_df.sort_values(by="Total", ascending=False)

    best_features = feature_selection_df.index[:num_feats].tolist()

    print("Feature selection complete!")
    #### Your Code ends here
    return best_features
