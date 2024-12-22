from collections import defaultdict
import os
import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from Cross_Validation import cross_validate_with_expression
from ML_models import PytorchModelWrapper, SklearnModelWrapper

'''
# Define the argument parser
parser = argparse.ArgumentParser(description="Run predictions with selected model and cancer type.")

# General arguments
parser.add_argument("--model", type=str, required=True, choices=["LogisticRegression", "RandomForest", "SVM", "XGBoost"] + [f"NeuralNet_{i+1}" for i in range(3)],
                    help="Select the model to use for prediction.")
parser.add_argument("--cancer_type", type=str, required=True, help="Specify the cancer type or 'all' for all cancers.")
parser.add_argument("--cv_mode", type=str, choices=["normal", "cross_cancer"], default="normal",
                    help="Specify the cross-validation mode: 'normal' for standard CV or 'cross_cancer' for cross-cancer CV.")
parser.add_argument("--verbosity", type=int, help="Verbosity level during cross-validation.", default=0)

# Model-specific hyperparameters
parser.add_argument("--learning_rate", type=float, help="Learning rate for models that use it (e.g., NeuralNet, XGBoost).")
parser.add_argument("--n_estimators", type=int, help="Number of estimators for ensemble models (e.g., RandomForest, XGBoost).")
parser.add_argument("--max_depth", type=int, help="Max depth for tree-based models (e.g., RandomForest, XGBoost).")
parser.add_argument("--epochs", type=int, help="Number of epochs for neural network models.")

args = parser.parse_args()

# Validate arguments
if args.cancer_type != 'all' and args.cv_mode == 'cross_cancer':
    print("Warning: 'cross_cancer' mode is ignored when a specific cancer type is provided. Switching to 'normal' mode.")
    args.cv_mode = 'normal'
'''


class Arguments ():
    def __init__(self):
        self.description = 'description'
        self.epochs = None
        self.max_depth = None
        self.n_estimators = None
        self.learning_rate = None
        self.verbosity = 0
        self.cancer_type = None


args = Arguments()
args.model = 'XGBoost'
args.cancer_type = 'all'
args.cv_mode = 'cross_cancer'
args.use_expression = False

# Suffix for filenames
suffix_parts = []
if args.learning_rate:
    suffix_parts.append(f"lr{args.learning_rate}")
if args.n_estimators:
    suffix_parts.append(f"nest{args.n_estimators}")
if args.max_depth:
    suffix_parts.append(f"md{args.max_depth}")
if args.epochs:
    suffix_parts.append(f"epochs{args.epochs}")
if args.cv_mode != 'normal':
    suffix_parts.append(f"{args.cv_mode}")

suffix = ''
if suffix_parts:
	suffix = "_" + "_".join(suffix_parts)

# Define models dynamically based on inputs
models = {
    "LogisticRegression": SklearnModelWrapper(LogisticRegression(max_iter=1000)),
    "RandomForest": SklearnModelWrapper(RandomForestClassifier(
        n_estimators=args.n_estimators if args.n_estimators else 100,
        max_depth=args.max_depth,
        random_state=42)),
    "SVM": SklearnModelWrapper(SVC(probability=True, random_state=42)),
    "XGBoost": SklearnModelWrapper(XGBClassifier(
        n_estimators=args.n_estimators if args.n_estimators else 100,
        learning_rate=args.learning_rate if args.learning_rate else 0.1,
        max_depth=args.max_depth if args.max_depth else 6,
        random_state=42))
}

# NeuralNet models
nn_architectures = [
    [64, 32],
    [128, 64, 32],
    [32, 16]
]

# Load data
df = pd.read_csv('../Output/Numerical_input.csv', index_col=0)
expression_df = pd.read_csv('../Output/Expression.csv', index_col=0)

cancer_type = args.cancer_type
cv_mode = args.cv_mode
use_expression = args.use_expression

# Filter data based on cancer type
if cancer_type.lower() == "all":
    filtered_df = df
    cancer_type = "All"
else:
    filtered_df = df[df['Cancer_type'] == cancer_type]
    if filtered_df.empty:
        raise ValueError(f"No data available for cancer type {cancer_type}.")
# (~df.apply(lambda x: (x != 1).sum() == 0)) & (~df.apply(lambda x: (x != 0).sum() == 0))
filtered_df.loc[:,~filtered_df.apply(lambda x: ((x != 0) & (x != False)).sum() == 0)]

# Add NeuralNet models
input_dim = filtered_df.shape[1] - 3

for i, architecture in enumerate(nn_architectures):
    models[f"NeuralNet_{i+1}"] = PytorchModelWrapper(
        input_dim=input_dim,  # This will be set dynamically based on input data
        layers=architecture,
        learning_rate=args.learning_rate if args.learning_rate else 0.005,
        epochs=args.epochs if args.epochs else 25
    )

# Select model
if args.model not in models:
    raise ValueError(f"Model {args.model} is not defined.")

model = models[args.model]

# Prepare results folder
results_folder = "../No_Expression"
os.makedirs(results_folder, exist_ok=True)
fi_folder = os.path.join(results_folder, 'Feature_importances')
os.makedirs(fi_folder, exist_ok=True)
fold_folder = os.path.join(results_folder, 'Per_fold')
os.makedirs(fold_folder, exist_ok=True)
mean_results = os.path.join(results_folder, 'Mean')
os.makedirs(mean_results, exist_ok=True)

# Run cross-validation
results = cross_validate_with_expression(
    dataframe=filtered_df,
    expression_df=expression_df,
    target_column='Appears_in_rna',
    sample_column='Tumor_Sample_Barcode_DNA',
    gene_column='Entrez_Gene_Id_DNA',
    model=model,
    cv_mode=cv_mode,
    use_expression=use_expression,
    n_splits=5,
    random_state=42,
    verbose=args.verbosity
)

# Capture metrics and feature importances
metrics = {
    "Model": args.model,
    "Cancer_Type": cancer_type,
    "Mean_Accuracy": results["mean_accuracy"],
    "Mean_ROC_AUC": results["mean_roc_auc"]
}
feature_importances = results.get("feature_importances", [None])

# Save results
results_file = os.path.join(mean_results, f"results_{args.model}_{cancer_type}{suffix}.csv")
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv(results_file)

# Save per-fold results
fold_results = results["fold_results"]  # Assume this contains per-fold accuracy and ROC AUC
fold_results_file = os.path.join(fold_folder, f"fold_results_{args.model}_{cancer_type}{suffix}.csv")
fold_results_df = pd.DataFrame(fold_results)
fold_results_df.index = 'Fold ' + fold_results_df.index.astype(str)
fold_results_df.loc['Mean'] = fold_results_df.mean()
fold_results_df.to_csv(fold_results_file)
print(f"Per-fold results saved to '{fold_results_file}'.")

# Save feature importances if available
if feature_importances and any(fi is not None for fi in feature_importances):
    fi_file = os.path.join(fi_folder, f"feature_importances_{args.model}_{cancer_type}{suffix}.csv")
    fi_df = pd.DataFrame(feature_importances).T
    fi_df.index = 'Fold ' + fi_df.index.astype(str)
    fi_df.loc['Mean_Feature_Importance'] = fi_df.mean()
    fi_df = fi_df.sort_values('Mean_Feature_Importance', axis = 1, ascending=False)
    fi_df.to_csv(fi_file)
    print(f"Feature importances saved to '{fi_file}'.")

print(f"Job completed. Results saved to '{results_file}'.")
