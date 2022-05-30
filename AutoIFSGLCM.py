# -*- coding: utf-8 -*-
print("------------------------------------------------------")
print("---------------- Metadata Information ----------------")
print("------------------------------------------------------")
print("")

print("In the name of God")
print("Project: AutoIFSGLCM: Automatic Important Features Selection Based on GLCM for Machine Learning Model")
print("Creator: Mohammad Reza Saraei")
print("Contact: mrsaraei@yahoo.com")
print("Supervisor: Dr. Saman Rajebi")
print("Created Date: May 29, 2022")
print("") 

print("----------------------------------------------------")
print("------------------ Import Libraries ----------------")
print("----------------------------------------------------")
print("")

# Import Libraries for Python
import os
import cv2
import glob
import numpy as np
import pandas as pd
from pandas import set_option
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops
from skimage import color, img_as_ubyte
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# print("----------------------------------------------------")
# print("----------------- Set Option -----------------------")
# print("----------------------------------------------------")
# print("")

set_option('display.max_rows', 500)
set_option('display.max_columns', 500)
set_option('display.width', 1000)

print("------------------------------------------------------")
print("---------------- Pixel Data Ingestion ----------------")
print("------------------------------------------------------")
print("")

# Import Images From Folders 
ImagePath = "Images4/"

print(os.listdir(ImagePath))
print("")

print("------------------------------------------------------")
print("---------------- Image Preprocessing -----------------")
print("------------------------------------------------------")
print("")

# Creating Empty List
images = []
target = []

for target_path in glob.glob(ImagePath + '//**/*', recursive = True):
    lable = target_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(target_path, "*.jpg")):
        img = cv2.imread(img_path, 0)
        img = cv2.resize(img, (128, 128)) 
        gray = color.rgb2gray(img)
        img = img_as_ubyte(gray)
        # plt.figure()
        # plt.imshow(img, cmap = 'gray')
        images.append(img)
        target.append(lable)
        print(img_path)

# Convert List to Array
images = np.array(images)
target = np.array(target)

print("")
print('Resized Images Shape:', images.shape)
print("")

print("------------------------------------------------------")
print("------------------- GLCM Function --------------------")
print("------------------------------------------------------")
print("")

# Creating GLCM Matrix
def FE(dataset):
    ImageDF = pd.DataFrame()
    for image in range(dataset.shape[0]):
        df = pd.DataFrame()
        img = dataset[image, :, :]
        
        GLCM = greycomatrix(img, [1], [0])        
        GLCM_Energy = greycoprops(GLCM, 'energy')[0]
        df['Energy'] = GLCM_Energy
        GLCM_corr = greycoprops(GLCM, 'correlation')[0]
        df['Corr'] = GLCM_corr       
        GLCM_diss = greycoprops(GLCM, 'dissimilarity')[0]
        df['Diss_sim'] = GLCM_diss       
        GLCM_hom = greycoprops(GLCM, 'homogeneity')[0]
        df['Homogen'] = GLCM_hom       
        GLCM_contr = greycoprops(GLCM, 'contrast')[0]
        df['Contrast'] = GLCM_contr
        GLCM_asm = greycoprops(GLCM, 'contrast')[0]
        df['ASM'] = GLCM_asm
        
        GLCM2 = greycomatrix(img, [3], [0])        
        GLCM_Energy2 = greycoprops(GLCM2, 'energy')[0]
        df['Energy2'] = GLCM_Energy2
        GLCM_corr2 = greycoprops(GLCM2, 'correlation')[0]
        df['Corr2'] = GLCM_corr2       
        GLCM_diss2 = greycoprops(GLCM2, 'dissimilarity')[0]
        df['Diss_sim2'] = GLCM_diss2       
        GLCM_hom2 = greycoprops(GLCM2, 'homogeneity')[0]
        df['Homogen2'] = GLCM_hom2       
        GLCM_contr2 = greycoprops(GLCM2, 'contrast')[0]
        df['Contrast2'] = GLCM_contr2
        GLCM_asm2 = greycoprops(GLCM2, 'contrast')[0]
        df['ASM2'] = GLCM_asm2
        
        GLCM3 = greycomatrix(img, [5], [0])        
        GLCM_Energy3 = greycoprops(GLCM3, 'energy')[0]
        df['Energy3'] = GLCM_Energy3
        GLCM_corr3 = greycoprops(GLCM3, 'correlation')[0]
        df['Corr3'] = GLCM_corr3       
        GLCM_diss3 = greycoprops(GLCM3, 'dissimilarity')[0]
        df['Diss_sim3'] = GLCM_diss3       
        GLCM_hom3 = greycoprops(GLCM3, 'homogeneity')[0]
        df['Homogen3'] = GLCM_hom3       
        GLCM_contr3 = greycoprops(GLCM3, 'contrast')[0]
        df['Contrast3'] = GLCM_contr3
        GLCM_asm3 = greycoprops(GLCM3, 'contrast')[0]
        df['ASM3'] = GLCM_asm3
        
        GLCM4 = greycomatrix(img, [0], [np.pi/2])        
        GLCM_Energy4 = greycoprops(GLCM4, 'energy')[0]
        df['Energy4'] = GLCM_Energy4
        GLCM_corr4 = greycoprops(GLCM4, 'correlation')[0]
        df['Corr4'] = GLCM_corr4     
        GLCM_diss4 = greycoprops(GLCM4, 'dissimilarity')[0]
        df['Diss_sim4'] = GLCM_diss4       
        GLCM_hom4 = greycoprops(GLCM4, 'homogeneity')[0]
        df['Homogen4'] = GLCM_hom4       
        GLCM_contr4 = greycoprops(GLCM4, 'contrast')[0]
        df['Contrast4'] = GLCM_contr4
        GLCM_asm4 = greycoprops(GLCM4, 'contrast')[0]
        df['ASM4'] = GLCM_asm4
            
        GLCM5 = greycomatrix(img, [0], [np.pi/4])        
        GLCM_Energy5 = greycoprops(GLCM5, 'energy')[0]
        df['Energy5'] = GLCM_Energy5
        GLCM_corr5 = greycoprops(GLCM5, 'correlation')[0]
        df['Corr5'] = GLCM_corr5     
        GLCM_diss5 = greycoprops(GLCM5, 'dissimilarity')[0]
        df['Diss_sim5'] = GLCM_diss5       
        GLCM_hom5 = greycoprops(GLCM5, 'homogeneity')[0]
        df['Homogen5'] = GLCM_hom5      
        GLCM_contr5 = greycoprops(GLCM5, 'contrast')[0]
        df['Contrast5'] = GLCM_contr5
        GLCM_asm5 = greycoprops(GLCM5, 'contrast')[0]
        df['ASM5'] = GLCM_asm5
        
        ImageDF = ImageDF.append(df)
    return ImageDF

print("------------------------------------------------------")
print("------------------- GLCM Propertis -------------------")
print("------------------------------------------------------")
print("")

# Extracting Features from All Images        
ImageFeatures = FE(images)
ImageFeatures['Diagnosis'] = target
print(ImageFeatures)        
print("")

print("------------------------------------------------------")
print("-------------------- Save Output ---------------------")
print("------------------------------------------------------")
print("")

# Save DataFrame After Encoding
pd.DataFrame(ImageFeatures).to_csv('AutoGLCM.csv', index = False)

print("------------------------------------------------------")
print("--------------- Pixel Data Ingestion -----------------")
print("------------------------------------------------------")
print("")

# Import DataFrame (.csv) by Pandas Library
df = pd.read_csv('AutoGLCM.csv')

# print("----------------------------------------------------")
# print("----------------- Set Option -----------------------")
# print("----------------------------------------------------")
# print("")

set_option('display.max_rows', 500)
set_option('display.max_columns', 500)
set_option('display.width', 1000)

print("------------------------------------------------------")
print("------------ Initial Data Understanding --------------")
print("------------------------------------------------------")
print("")

print("Initial General Information:")
print("****************************")
print(df.info())
print("")

print("------------------------------------------------------")
print("---------------- Data Label Encoding -----------------")
print("------------------------------------------------------")
print("")

# Encoding Coulmns Having Objects by LabelEncoder
obj = df.select_dtypes(include = ['object'])
LE = preprocessing.LabelEncoder()
col = obj.apply(LE.fit_transform)

print("Columns Having Objects:")
print("***********************")
print(obj.head(10))
print("")

print("Encoding Columns Having Object:")
print("*******************************")
print(col.head(10))
print("")
print('Shape of Encoded Columns:', col.shape)
print("")

print("------------------------------------------------------")
print("------------- Save Encoded Objects Data --------------")
print("------------------------------------------------------")
print("")

# Save DataFrame After Encoding
pd.DataFrame(col).to_csv('EncodedData.csv', index = False)

print("------------------------------------------------------")
print("------- Creating Main DataFrame by Combination -------")
print("------------------------------------------------------")
print("")

# Import Encoded Objects DataFrame (.csv) by Pandas Library
df_col = pd.read_csv('EncodedData.csv')

# Combinating Encoded Data with Main DataFrame
df_obj = df.drop(df.select_dtypes(include = ['object']), axis = 1)

print("Columns' Name that needs to encoding:", obj.columns)
print("")

print("The Target Column Name:", df.columns[-1])
print("")

if df.columns[-1] in obj.columns:
    df = pd.concat([df_obj, df_col], axis = 1)
else:
    df = pd.concat([df_col, df_obj], axis = 1)

print("An overview of Encoded Data:")
print("****************************")
print("")
print(df.head(1))
print("")

print("------------------------------------------------------")
print("--------- Data Understanding After Encoding ----------")
print("------------------------------------------------------")
print("")

print("General Information After Encoding:")
print("***********************************")
print(df.info())
print("")

print("------------------------------------------------------")
print("------------------ Data Spiliting --------------------")
print("------------------------------------------------------")
print("")

# Select Features (as "f") and Target (as "t") Data
f = df.iloc[:, 0: -1].values
t = df.iloc[:, -1].values     

print("------------------------------------------------------")
print("---------------- Data Normalization ------------------")
print("------------------------------------------------------")
print("")

# Normalization [0, 1] of Data
scaler = MinMaxScaler(feature_range = (0, 1))
f = scaler.fit_transform(f)
print(f)
print("")

print("------------------------------------------------------")
print("----------- Save Features and Target Data ------------")
print("------------------------------------------------------")
print("")

# Save DataFrame (f, t) After Munging
pd.DataFrame(f).to_csv('FeaturesData.csv', index = False)
pd.DataFrame(t).to_csv('TargetData.csv', index = False)

print("------------------------------------------------------")
print("-------- Features and Target Data Combination --------")
print("------------------------------------------------------")
print("")

# Import Again DataFrames (f, t) by Pandas Library
df_f = pd.read_csv('FeaturesData.csv')
df_t = pd.read_csv('TargetData.csv')

# Rename t Column
df_t.rename(columns = {'0': 'Diagnosis'}, inplace = True)

# Combination of DataFrames
df = pd.concat([df_f, df_t], axis = 1)

# Save Combination f and t DataFrames After Munging
pd.DataFrame(df).to_csv('MainDataFrame.csv', index = False)

# print("----------------------------------------------------")
# print("------------------ Data Ingestion ------------------")
# print("----------------------------------------------------")
# print("")

# Import DataFrame (.csv) by Pandas Library
df = pd.read_csv('MainDataFrame.csv')

print("------------------------------------------------------")
print("---------------- Data Preprocessing ------------------")
print("------------------------------------------------------")
print("")

# Replace Question Mark to NaN:
df.replace("?", np.nan, inplace = True)

# Remove Duplicate Samples
df = df.drop_duplicates()
print("Duplicate Records After Removal:", df.duplicated().sum())
print("")

# Replace Mean instead of Missing Values
imp = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imp.fit(df)
df = imp.transform(df)
print("Mean Value For NaN Value:", "{:.3f}".format(df.mean()))
print("")

# Reordering Records / Samples / Rows
print("Reordering Records:")
print("*******************")
df = pd.DataFrame(df).reset_index(drop = True)
print(df)
print("")

print("------------------------------------------------------")
print("------------------ Data Respiliting ------------------")
print("------------------------------------------------------")
print("")

# Select Features (as "f") and Target (as "t") Data
f = df.iloc[:, 0: -1].values
t = df.iloc[:, -1].values     

print("------------------------------------------------------")
print("----------------- Outliers Detection -----------------")
print("------------------------------------------------------")
print("")

# Identify Outliers in the Training Data
ISF = IsolationForest(n_estimators = 100, contamination = 0.1, bootstrap = True, n_jobs = -1)

# Fitting Outliers Algorithms on the Training Data
ISF = ISF.fit_predict(f, t)

# Select All Samples that are not Outliers
Mask = ISF != -1
f, t = f[Mask, :], t[Mask]

print('nFeature:', f.shape)
print('nTarget:', t.shape)
print("")

print("------------------------------------------------------")
print("------------- Data Balancing By SMOTE ----------------")
print("------------------------------------------------------")
print("")

# Summarize Targets Distribution
print('Targets Distribution Before SMOTE:', sorted(Counter(t).items()))

# OverSampling (OS) Fit and Transform the DataFrame
OS = SMOTE()
f, t = OS.fit_resample(f, t)

# Summarize the New Targets Distribution
print('Targets Distribution After SMOTE:', sorted(Counter(t).items()))
print("")

print('nFeature:', f.shape)
print('nTarget:', t.shape)
print("")

print("------------------------------------------------------")
print("----------- Save Features and Target Data ------------")
print("------------------------------------------------------")
print("")

# Save DataFrame (f, t) After Munging
pd.DataFrame(f).to_csv('FeaturesData.csv', index = False)
pd.DataFrame(t).to_csv('TargetData.csv', index = False)

print("------------------------------------------------------")
print("-------- Features and Target Data Combination --------")
print("------------------------------------------------------")
print("")

# Import Again DataFrames (f, t) by Pandas Library
df_f = pd.read_csv('FeaturesData.csv')
df_t = pd.read_csv('TargetData.csv')

# Rename t Column
df_t.rename(columns = {'0': 'Diagnosis'}, inplace = True)

# Combination of DataFrames
df = pd.concat([df_f, df_t], axis = 1)

# Save Combination f and t DataFrames After Munging
pd.DataFrame(df).to_csv('MainDataFrame.csv', index = False)

print("------------------------------------------------------")
print("----------------- Data Understanding -----------------")
print("------------------------------------------------------")
print("")

print("Dataset Overview:")
print("*****************")
print(df.head(10))
print("")

print("General Information:")
print("********************")
print(df.info())
print("")

print("Statistics Information:")
print("***********************")
print(df.describe(include="all"))
print("")

print("nSample & (nFeature + Target):", df.shape)
print("")

print("Samples Range:", df.index)
print("")

print(df.columns)
print("")

print("Missing Values (NaN):")
print("*********************")
print(df.isnull().sum())                                         
print("")

print("Duplicate Records:", df.duplicated().sum())
print("")   

print("Features Correlations:")
print("**********************")
print(df.corr(method='pearson'))
print("")

print("------------------------------------------------------")
print("--------------- Data Distribution --------------------")
print("------------------------------------------------------")
print("")

print("nSample & (nFeature + Target):", df.shape)
print("")

print("Skewed Distribution of Features:")
print("********************************")
print(df.skew())
print("")
print(df.dtypes)
print("")

print("Target Distribution:")
print("********************")
print(df.groupby(df.iloc[:, -1].values).size())
print("")

print("------------------------------------------------------")
print("----------- Plotting Distribution of Data ------------")
print("------------------------------------------------------")
print("")

# Plot the Scores by Descending
plt.hist(df)
plt.xlabel('Data Value', fontsize = 11)
plt.ylabel('Data Frequency', fontsize = 11)
plt.title('GLCM-Based Pixel Data Distribution After Preparation')
plt.savefig('AutoPDPGLCM_DataDistribution.png', dpi = 600)
plt.savefig('AutoPDPGLCM_DataDistribution.tif', dpi = 600)
plt.show()
plt.close()

# print("----------------------------------------------------")
# print("------------------ Data Ingestion ------------------")
# print("----------------------------------------------------")
# print("")

# Import DataFrame (.csv) by Pandas Library
df = pd.read_csv('MainDataFrame.csv')

# Select Features (as "f") and Target (as "t") Data
f = df.iloc[:, 0: -1]
t = df.iloc[:, -1]

# Computing the Number of Features in Dataset
nFeature = len(f.columns)
print('The Number of Features:', nFeature)
print("")

print("----------------------------------------------------")
print("------------ Select K Best (ANOVA F) ---------------")
print("----------------------------------------------------")
print("")

# Determine the most Important Features
BF = SelectKBest(score_func = f_classif, k = 'all')    
fit_ANOVAF = BF.fit(f, t)

# Create DataFrames
df_columns = pd.DataFrame(f.columns)
df_ANOVAF = pd.DataFrame(fit_ANOVAF.scores_)

# Concatenate DataFrames
feature_ANOVAF_scores = pd.concat([df_columns, df_ANOVAF], axis = 1)
feature_ANOVAF_scores.columns = ['Features', 'Score']

print(feature_ANOVAF_scores.nlargest(nFeature, 'Score'))
print("")

print("----------------------------------------------------")
print("-------------- Select K Best (Chi2) ----------------")
print("----------------------------------------------------")
print("")

# Determine the most Important Features
BF = SelectKBest(score_func = chi2, k = 'all')
fit_Chi2 = BF.fit(f, t)

# Create DataFrames
df_columns = pd.DataFrame(f.columns)
df_Chi2 = pd.DataFrame(fit_Chi2.scores_)

# Concatenate DataFrames
feature_Chi2_scores = pd.concat([df_columns, df_Chi2], axis = 1)
feature_Chi2_scores.columns = ['Feature', 'Score']  

print(feature_Chi2_scores.nlargest(nFeature, 'Score'))
print("")

print("----------------------------------------------------")
print("------- Select K Best (Mutual Info Classif) --------")
print("----------------------------------------------------")
print("")

# Determine the most Important Features
BF = SelectKBest(score_func = mutual_info_classif, k = 'all')    
fit_MICIF = BF.fit(f, t)

# Create DataFrames
df_columns = pd.DataFrame(f.columns)
df_MICIF = pd.DataFrame(fit_MICIF.scores_)

# Concatenate DataFrames
feature_MICIF_scores = pd.concat([df_columns, df_MICIF], axis = 1)
feature_MICIF_scores.columns = ['Features', 'Score']

print(feature_MICIF_scores.nlargest(nFeature, 'Score'))
print("")

print("----------------------------------------------------")
print("------------ PI KNeighbors Classifier --------------")
print("----------------------------------------------------")
print("")

# Determine the most Important Features
KNN = KNeighborsClassifier()
KNN.fit(f,t)

# Perform Permutation Importance
results = permutation_importance(KNN, f, t, scoring = 'accuracy')

# Create DataFrames
df_columns = pd.DataFrame(f.columns)
df_PIKNN = pd.DataFrame(results.importances_mean)

# Concatenate DataFrames
feature_PIKNN_scores = pd.concat([df_columns, df_PIKNN], axis = 1)
feature_PIKNN_scores.columns = ['Features', 'Score']

print(feature_PIKNN_scores.nlargest(nFeature, 'Score'))
print("")

print("----------------------------------------------------")
print("---------------------- LASSO -----------------------")
print("----------------------------------------------------")
print("")

# Determine the most Important Features
reg = LassoCV()
reg.fit(f, t)

# Create DataFrames
df_columns = pd.DataFrame(f.columns)
df_LASSO = pd.DataFrame(reg.coef_)

# Concatenate DataFrames
feature_LASSO_scores = pd.concat([df_columns, df_LASSO], axis = 1)
feature_LASSO_scores.columns = ['Features', 'Score']

print(feature_LASSO_scores.nlargest(nFeature, 'Score'))
print("")

print("----------------------------------------------------")
print("------------- Decision Trees Classifier ------------")
print("----------------------------------------------------")
print("")

# Determine the most Important Features
DTC = DecisionTreeClassifier()
DTC.fit(f,t)

# Create DataFrames
df_columns = pd.DataFrame(f.columns)
df_DTC = pd.DataFrame(DTC.feature_importances_)

# Concatenate DataFrames
feature_DTC_scores = pd.concat([df_columns, df_DTC], axis = 1)
feature_DTC_scores.columns = ['Features', 'Score']

print(feature_DTC_scores.nlargest(nFeature, 'Score'))
print("")

print("----------------------------------------------------")
print("------------- Extra Trees Classifier ---------------")
print("----------------------------------------------------")
print("")

# Determine the most Important Features
ETC = ExtraTreesClassifier()
ETC.fit(f,t)

# Create DataFrames
df_columns = pd.DataFrame(f.columns)
df_ETC = pd.DataFrame(ETC.feature_importances_)

# Concatenate DataFrames
feature_ETC_scores = pd.concat([df_columns, df_ETC], axis = 1)
feature_ETC_scores.columns = ['Features', 'Score']

print(feature_ETC_scores.nlargest(nFeature, 'Score'))
print("")

print("----------------------------------------------------")
print("------------ Random Forest Classifier --------------")
print("----------------------------------------------------")
print("")

# Determine the most Important Features
RFC = RandomForestClassifier()
RFC.fit(f,t)

# Create DataFrames
df_columns = pd.DataFrame(f.columns)
df_RFC = pd.DataFrame(RFC.feature_importances_)

# Concatenate DataFrames
feature_RFC_scores = pd.concat([df_columns, df_RFC], axis = 1)
feature_RFC_scores.columns = ['Features', 'Score']

print(feature_RFC_scores.nlargest(nFeature, 'Score'))
print("")

print("----------------------------------------------------")
print("--------------- XGBoost Classifier -----------------")
print("----------------------------------------------------")
print("")

# Determine the most Important Features
XGB = XGBClassifier()
XGB.fit(f,t)

# Create DataFrames
df_columns = pd.DataFrame(f.columns)
df_XGB = pd.DataFrame(XGB.feature_importances_)

# Concatenate DataFrames
feature_XGB_scores = pd.concat([df_columns, df_XGB], axis = 1)
feature_XGB_scores.columns = ['Features', 'Score']

print(feature_XGB_scores.nlargest(nFeature, 'Score'))
print("")

print("----------------------------------------------------")
print("------------------- Make DataFrame -----------------")
print("----------------------------------------------------")
print("")

# Create DataFrames
df_columns = pd.DataFrame(f.columns)
df_ANOVAF = pd.DataFrame(fit_ANOVAF.scores_)
df_Chi2 = pd.DataFrame(fit_Chi2.scores_)
df_MICIF = pd.DataFrame(fit_MICIF.scores_)
df_PIKNN = pd.DataFrame(results.importances_mean)
df_LASSO = pd.DataFrame(reg.coef_)
df_DTC = pd.DataFrame(DTC.feature_importances_)
df_ETC = pd.DataFrame(ETC.feature_importances_)
# df_LRG = pd.DataFrame(np.transpose(abs(LRG.coef_)))
df_RFC = pd.DataFrame(RFC.feature_importances_)
df_XGB = pd.DataFrame(XGB.feature_importances_)

# Concatenate DataFrames
feature_scores = pd.concat([df_columns, df_ANOVAF, df_Chi2, df_MICIF, df_PIKNN, df_LASSO, df_DTC, df_ETC, df_RFC, df_XGB], axis = 1)
feature_scores.columns = ['Features', 'ANOVAF', 'SKB-Chi2', 'SKB-MICIF', 'PIKNN', 'LASSO', 'DTC', 'ETC', 'RFC', 'XGB']

print(feature_scores)
print("")

# Adding 'Mean Scores' Column to Feature Scores DataFrame
df_Mean = pd.DataFrame(feature_scores.iloc[:, 1: -1].mean(axis = 1))
feature_scores['Mean Scores'] = df_Mean

# Prioritized Features
prioritized_features = feature_scores.nlargest(nFeature, 'Mean Scores')
print(prioritized_features)
print("")

print("----------------------------------------------------")
print("---------------- Plotting Outputs ------------------")
print("----------------------------------------------------")
print("")

# Plot the Scores by Descending
plt.bar(prioritized_features['Features'], prioritized_features['Mean Scores'], color = 'black')
plt.xlabel('Feature', fontsize = 12)
plt.ylabel('Score', fontsize = 12)
plt.legend(['Importance Bar'])
plt.title('Prioritized Features based on AutoIFSGLCM')
plt.savefig('AutoIFSGLCM.png', dpi = 600)
plt.savefig('AutoIFSGLCM.tif', dpi = 600)
plt.show()
plt.close()

print("----------------------------------------------------")
print("----------------- Saving Outputs -------------------")
print("----------------------------------------------------")
print("")

# Export Selected Features to .CSV
df_feat = feature_scores.nlargest(nFeature, 'Mean Scores')
df_feat.to_csv('ImpoertantFeatures.csv', index = False)

print("----------------------------------------------------")
print("------------ Important Features Selection ----------")
print("----------------------------------------------------")
print("")

# Display Top 50% Important Features
top_feat = feature_scores.nlargest(int(0.5*nFeature), 'Mean Scores')
important_feat = (top_feat.index).values
featureList = list(important_feat)
print(featureList)
print("")

print("------------------------------------------------------")
print("---------- Thank you for waiting, Good Luck ----------")
print("---------- Signature: Mohammad Reza Saraei -----------")
print("------------------------------------------------------")

