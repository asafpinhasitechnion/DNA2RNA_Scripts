import os
import pandas as pd


# results_folder = '../Results/Mean'

results_folder = '../Results_All_Features/Mean'
#results_folder = '../Results_Only_Expression/Mean'

files = os.listdir(results_folder)


df_dict = {}
for file_name in files:
	df = pd.read_csv(os.path.join(results_folder, file_name), index_col = 0)
	df_dict[file_name] = df['Mean_ROC_AUC']

df = pd.DataFrame(df_dict)
df = df.T
df.sort_values(by = 0).to_csv('../Output/mean_roc_auc.csv')

'''
results_folder = '../Results_All_Features/Feature_importances'

files = os.listdir(results_folder)


df_dict = {}
for file_name in files:
	df = pd.read_csv(os.path.join(results_folder, file_name), index_col = 0)
	df_dict[file_name] = df.loc['Mean_Feature_Importance']

df = pd.DataFrame(df_dict)
df = df.T
df.to_csv('../Output/mean_feature_importance.csv')
'''