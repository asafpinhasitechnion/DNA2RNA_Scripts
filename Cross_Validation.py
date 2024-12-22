from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

def cross_validate_with_expression(
    dataframe, expression_df, target_column, sample_column, gene_column, model, cancer_columns = 'Cancer_type', use_expression=True,
    n_splits=5, random_state=42, cv_mode='normal', verbose=0
):
    """
    Perform cross-validation while avoiding expression leakage, supporting both normal and cross-cancer modes.

    Parameters:
    - dataframe (pd.DataFrame): Input data with features, target column, and sample identifiers.
    - expression_df (pd.DataFrame): Gene expression data where rows are genes and columns are samples.
    - target_column (str): The name of the column to be used as the target.
    - sample_column (str): The column that identifies samples (shortened barcodes).
    - gene_column (str): The column in `dataframe` that maps to expression_df's index.
    - model: The machine learning model to use for training.
    - n_splits (int): Number of cross-validation splits (only used in normal mode).
    - random_state (int): Random seed for reproducibility.
    - cv_mode (str): 'normal' for standard CV, 'cross_cancer' for cross-cancer CV.
    - verbose (int): Higher verbosity prints more detailed logs for each fold.

    Returns:
    - dict: Cross-validation metrics including mean accuracy, ROC AUC, fold results, and feature importance (if available).
    """
    # Prepare data
    dataframe[sample_column] = dataframe[sample_column].str.split('-').str[1:3].map('-'.join)
    samples = np.array(dataframe[sample_column].unique())
    
    folds = []

    if cv_mode == 'normal':
        # Standard K-Fold cross-validation
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        folds = list(splitter.split(samples))
        folds = [(None, train_index, test_index) for train_index, test_index in folds] # Dummy cancer type

    elif cv_mode == 'cross_cancer':
        for ct in dataframe[cancer_columns].unique():
            temp_df = dataframe[dataframe[cancer_columns] == ct]
            ct_samples = temp_df[sample_column].unique()
            # Get indices for train and test
            test_sample_idx = np.where(np.isin(samples, ct_samples))[0]
            train_sample_idx = np.where(~np.isin(samples, ct_samples))[0]

            folds.append((ct, train_sample_idx, test_sample_idx))
    else:
        raise ValueError("cv_mode must be 'normal' or 'cross_cancer'")

    fold_results = {}
    all_feature_importances = {}

    for fold, (ct, train_index, test_index) in enumerate(folds):
        
        if verbose:
            print(f"\nFold {fold + 1}/{n_splits}")
            if cv_mode == 'cross_cancer':
                print(f"Test cancer types: {ct}")
            print(f"Train samples: {len(train_samples)}, Test samples: {len(test_samples)}")

        # Train and test samples
        train_samples = samples[train_index]
        test_samples = samples[test_index]
        if use_expression:
            # Exclude test samples from expression data
            temp_expression_df = expression_df.loc[:, ~expression_df.columns.str.split('-').str[1:3].map('-'.join).isin(test_samples)]

            # Drop unnecessary columns
            temp_expression_df = temp_expression_df.drop(sample_column, axis=1)

            # Group by 'Cancer_type' and compute the mean for training genes
            temp_expression_df = temp_expression_df.groupby(cancer_columns).mean()

            # Reset index and melt into long format
            temp_expression_df = temp_expression_df.reset_index().melt(
                id_vars=cancer_columns, var_name=gene_column, value_name='Mean_Expression'
            )

            # Ensure both DataFrames have consistent data types for merging
            temp_expression_df[gene_column] = temp_expression_df[gene_column].astype(str)
            dataframe[gene_column] = dataframe[gene_column].astype(str)

            # Merge the two DataFrames on 'Cancer_type' and gene_column
            merged_train_df = dataframe.merge(
                temp_expression_df,
                on=[cancer_columns, gene_column]
            )
        else:
            merged_train_df = dataframe
        
        merged_train_df = merged_train_df.drop([cancer_columns, gene_column], axis=1)

        # Split into train and test sets
        train_data = merged_train_df[merged_train_df[sample_column].isin(train_samples)].drop(sample_column, axis=1)
        test_data = merged_train_df[merged_train_df[sample_column].isin(test_samples)].drop(sample_column, axis=1)

        X_train, y_train = train_data.drop(target_column, axis=1).values, train_data[target_column].values
        X_test, y_test = test_data.drop(target_column, axis=1).values, test_data[target_column].values

        if verbose > 1:
            print(f"Train label distribution: {np.bincount(y_train.astype(int))}")
            print(f"Test label distribution: {np.bincount(y_test.astype(int))}")

        # Scale the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train the model
        model.fit(X_train, y_train)

        # Extract feature importance with feature names if available
        if hasattr(model, "feature_importances_"):
            feature_importance_dict = dict(zip(train_data.drop(target_column, axis=1).columns, model.feature_importances_))
            all_feature_importances[fold + 1] = feature_importance_dict
            if verbose > 1:
                print(f"Feature importances: {feature_importance_dict}")
        else:
            all_feature_importances = None

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else y_pred

        # Evaluate metrics
        accuracy = accuracy_score(y_test, y_pred)
        try:
            roc_auc = roc_auc_score(y_test, y_pred_prob)
        except ValueError:
            roc_auc = None  # Handle cases where AUC cannot be calculated

        if verbose:
            if cv_mode == 'cross_cancer':
                print(f"Fold {fold + 1} Testing on {ct} Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc if roc_auc is not None else 'N/A'}")
            else:
                print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc if roc_auc is not None else 'N/A'}")

        if cv_mode == 'cross_cancer':
            fold_results[ct] = {'accuracy': accuracy, 'roc_auc': roc_auc}
        else:
            fold_results[fold + 1] = {'accuracy': accuracy, 'roc_auc': roc_auc}

    # Compile results
    results_df = pd.DataFrame(fold_results).T
    mean_accuracy = results_df['accuracy'].mean()
    mean_roc_auc = results_df['roc_auc'].mean()

    if verbose:
        print("\nCross-validation complete.")
        print(f"Mean Accuracy: {mean_accuracy:.4f}")
        print(f"Mean ROC AUC: {mean_roc_auc if mean_roc_auc is not None else 'N/A'}")

    return {
        "fold_results": results_df,
        "mean_accuracy": mean_accuracy,
        "mean_roc_auc": mean_roc_auc,
        "feature_importances": all_feature_importances
    }
