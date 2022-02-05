### Part 1: Import packages and modules ###
## ======================================================================= ##
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost
import tensorflow as tf
import keras
import shap
import lime
# ---------------------------------------------------------------------------
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, plot_roc_curve, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split 
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.naive_bayes import GaussianNB as gnb
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
# ---------------------------------------------------------------------------
from xgboost import XGBClassifier
# ---------------------------------------------------------------------------
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.utils import to_categorical
from keras.regularizers import l2

### Part 2: Data preparation ###
## ======================================================================= ##
## Set random seed.
seed = 888
# ---------------------------------------------------------------------------
## Import data from Excel file.
all_data = pd.read_excel('./data.xlsx', sheet_name='main').set_index('sample')
# ---------------------------------------------------------------------------
## Single out grades and peaks, separately
X = all_data.iloc[:,13:78]
y = all_data['grade']
# ---------------------------------------------------------------------------
## Split the test and train datasets according to 85:15 ratio
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.15, random_state=seed, stratify=y, shuffle=True)
# ---------------------------------------------------------------------------
## Define stratified cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# ---------------------------------------------------------------------------
## One-hot encoding
cat_y = to_categorical(y, num_classes=3)
cat_y_train = to_categorical(y_train, num_classes=3)
cat_y_test = to_categorical(y_test, num_classes=3)

### Part 3: PCA-LDA method ###
## ======================================================================= ##
## Initialise PCA instance
pca = PCA(n_components=5)
# ---------------------------------------------------------------------------
## Normalise the original dataset into 0-1 scale
X_train_norm = pd.DataFrame(MinMaxScaler().fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_norm = pd.DataFrame(MinMaxScaler().fit_transform(X_test), columns=X_test.columns, index=X_test.index)
# ---------------------------------------------------------------------------
## Perform PCA
X_train_pca = pca.fit_transform(X_train_norm, y_train)
X_test_pca = pca.fit_transform(X_test_norm, y_test)
pca.explained_variance_ratio_ # Confirm the explained variance by each PC
# ---------------------------------------------------------------------------
## Perform LDA using PCA results
lda_model = LDA()
X_train_lda = lda_model.fit(X_train_pca, y_train)
# ---------------------------------------------------------------------------
# Calculate LDA classification predictions, probabilities, and accuracy
lda_y_pred = lda_model.predict(X_test_pca)
lda_y_pred_prob = lda_model.predict_proba(X_test_pca)
lda_predictions = [round(value) for value in lda_y_pred]
accuracy = accuracy_score(y_test, lda_predictions)
print('PCA-LDA_Accuracy: %.2f%%' % (accuracy * 100.0))
# ---------------------------------------------------------------------------
## Calculate the ROC AUC for LDA
lda_roc = roc_auc_score(y_test, lda_y_pred_prob, multi_class='ovr')
print(lda_roc)
# ---------------------------------------------------------------------------
## Calculate and visualise SHAP values of the PCA-LDA model
lda_explainer = shap.KernelExplainer(lda.predict, X_train_pca)
lda_shap_values = lda_explainer.shap_values(X_test_pca)
lda_summary_plot = shap.summary_plot(lda_shap_values, X_test_pca, plot_type='dot') # Print the SHAP summary plot.
# ---------------------------------------------------------------------------
## Calculate the ROC AUC for LDA in cross validation
aucroclda = []
for train, test in kfold.split(X, y):
    # normalise the data
    X_train_normed = pd.DataFrame(MinMaxScaler().fit_transform(X.iloc[train]), columns=X.columns)
    X_test_normed = pd.DataFrame(MinMaxScaler().fit_transform(X.iloc[test]), columns=X.columns)
    # append PCA before LDA
    X_train_crossval_pca = pca.fit_transform(X_train_normed, y[train])
    X_test_crossval_pca = pca.fit_transform(X_test_normed, y[test])
    # fit the LDA model
    lda_model.fit(X_train_crossval_pca, y[train])
    # test the LDA model
    lda_y_pred_prob = lda_model.predict_proba(X_test_crossval_pca) 
    # calculate auc of roc
    roc_auc_lda = roc_auc_score(y[test], lda_y_pred_prob, multi_class='ovr')
    aucroclda.append(roc_auc_lda)
print(np.mean(aucroclda), np.std(aucroclda))
# ---------------------------------------------------------------------------
## Calculate cross validation accuracy
X_norm = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=X.columns)
pca_crossval = X_norm = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=X.columns, index=X.index)
lda_crossval = cross_val_score(lda_model, pca_crossval, y, cv=kfold)
print(lda_crossval.mean()*100, lda_crossval.std()*100)

### Part 4: k Nearest Neighbour (kNN) method ###
## ======================================================================= ##
## Define kNN parameters (set k) and train the kNN model
knn_model = knn(n_neighbors=5)
knn_model.fit(X_train, y_train)
# ---------------------------------------------------------------------------
## Calculate kNN classification predictions, probabilities, and accuracy
knn_y_pred = knn_model.predict(X_test)
knn_y_pred_prob = knn_model.predict_proba(X_test)
knn_accuracy = accuracy_score(y_test, knn_y_pred)
print("kNN_Accuracy: %.2f%%" % (knn_accuracy * 100.0)) ## Print prediction accuracy.
knn_roc = roc_auc_score(y_test, knn_y_pred_prob, multi_class='ovr')
print(knn_roc)
# ---------------------------------------------------------------------------
## Calculate the ROC AUC for kNN in cross validation
aucrocknn = []
for train, test in kfold.split(X, y):
    # fit the kNN model
    knn_model.fit(X.iloc[train], y[train])
    # test the kNN model
    knn_y_pred_prob = knn_model.predict_proba(X.iloc[test]) 
    # calculate auc of roc
    roc_auc_knn = roc_auc_score(y[test], knn_y_pred_prob, multi_class='ovr')
    aucrocknn.append(roc_auc_knn)
print(np.mean(aucrocknn), np.std(aucrocknn))
# ---------------------------------------------------------------------------
## Calculate cross validation accuracy
knncrossval = cross_val_score(knn_model, X, y, cv=kfold)
print(knncrossval.mean()*100, knncrossval.std()*100)

### Part 5: Support vector machine (SVM) method ###
## ======================================================================= ##
## Train the SVM model 
svm_model = svm.SVC(kernel='poly', probability=True)
svm_model = svm.fit(X_train,y_train)
# ---------------------------------------------------------------------------
## Calculate SVM classification predictions, probabilities, and accuracy
svm_y_pred_prob = svm_model.predict_proba(X_test)
svm_y_pred = svm_model.predict(X_test)
print("Accuracy:",accuracy_score(y_test, svm_y_pred))
# ---------------------------------------------------------------------------
## Calculate the ROC AUC for the SVM model
svm_roc = roc_auc_score(y_test, svm_y_pred_prob, multi_class='ovr')
print(svm_roc)
# ---------------------------------------------------------------------------
## Calculate the ROC AUC for SVM in cross validation
aucrocsvm = []
for train, test in kfold.split(X, y):
    # fit the SVM model
    svm_model.fit(X.iloc[train], y[train])
    # test the SVM model
    svm_y_pred_prob = svm_model.predict_proba(X.iloc[test]) 
    # calculate auc of roc
    roc_auc_svm = roc_auc_score(y[test], svm_y_pred_prob, multi_class='ovr')
    aucrocsvm.append(roc_auc_svm)
print(np.mean(aucrocsvm), np.std(aucrocsvm))
# ---------------------------------------------------------------------------
## Calculate cross validation accuracy
svmcrossval = cross_val_score(svm_model, X, y, cv=kfold)
print(svmcrossval.mean()*100, svmcrossval.std()*100)

### Part 6: eXtreme Gradient Boosting (XGBoost) method ###
## ======================================================================= ##
## Train the XGBoost model and inspect the trained model
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
print(xgb_model)
# ---------------------------------------------------------------------------
## Calculate XGBoost classification predictions, probabilities, and accuracy
xgb_y_pred = xgb_model.predict(X_test) 
xgb_y_pred_prob = xgb_model.predict_proba(X_test) 
xgb_predictions = [round(value) for value in xgb_y_pred]
xgb_accuracy = accuracy_score(y_test, xgb_predictions)
print("XGBoost_Accuracy: %.2f%%" % (xgb_accuracy * 100.0)) 
# ---------------------------------------------------------------------------
## Calculate and visualise SHAP values of the XGBoost model
xgb_explainer = shap.TreeExplainer(xgb_model)
xgb_shap_values = xgb_explainer.shap_values(X_train.values)
xgb_summary_plot = shap.summary_plot(xgb_shap_values[0], X_train, plot_type='dot') # Print the SHAP summary plot.
# ---------------------------------------------------------------------------
## Calculate the ROC AUC for XGBoost
xgb_roc = roc_auc_score(y_test, xgb_y_pred_prob, multi_class='ovr')
print(xgb_roc)
# ---------------------------------------------------------------------------
## Calculate the ROC AUC for XGBoost in cross validation
aucrocxgb = []
for train, test in kfold.split(X, y):
    # fit the xgboost model
    xgb_model.fit(X.iloc[train], y[train])
    # test the xgboost model
    xgb_y_pred_prob = xgb_model.predict_proba(X.iloc[test]) 
    # calculate auc of roc
    roc_auc_xgb = roc_auc_score(y[test], xgb_y_pred_prob, multi_class='ovr')
    aucrocxgb.append(roc_auc_xgb)
print(np.mean(aucrocxgb), np.std(aucrocxgb))
# ---------------------------------------------------------------------------
## Calculate cross validation accuracy
xgbcrossval = cross_val_score(xgb_model, X, y, cv=kfold)
print(xgbcrossval.mean()*100, xgbcrossval.std()*100)

### Part 7: Artificial Neural Network (ANN) method ###
'''
!!!WARNING!!!
## ======================================================================= ##
DUE TO THE STOCHIASTIC NATURE OF THE ANN TRAINING PROCESS, THE VALUES CALCULATED FOR ANN MAY NOT BE ENTIRELY REPRODUCIBLE SHOULD THE SAME CODE BE EXECUTED FOR ANOTHER TIME.
'''
## ======================================================================= ##
## Define the ANN model
dl_model = Sequential()
dl_model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01), input_dim=X.shape[1]))
dl_model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
dl_model.add(Dense(3, activation='softmax'))
adamoptimise = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# ---------------------------------------------------------------------------
## Compile the ANN model
dl_model.compile(loss='categorical_crossentropy', optimizer=adamoptimise, metrics=['accuracy'])
# ---------------------------------------------------------------------------
## Train the ANN model
dl_history = dl_model.fit(X_train.values, cat_y_train, epochs=7500, batch_size=32)
# ---------------------------------------------------------------------------
## Calculate ANN classification predictions, probabilities, and accuracy
dl_y_pred = np.argmax(dl_model.predict(X_test), axis=-1)
dl_y_pred_prob = dl_model.predict(X_test) 
dl_predictions = [round(value) for value in dl_y_pred]
accuracy = accuracy_score(y_test, dl_predictions)
print("ANN_Accuracy: %.2f%%" % (accuracy * 100.0))
# ---------------------------------------------------------------------------
## Calculate cross validation accuracy
dlcvscores = []
for train, test in kfold.split(X, y):
    # One-hot encode gradings
    cat_y_train = to_categorical(y[train], num_classes=3)
    cat_y_test = to_categorical(y[test], num_classes=3)
    # Silently fit the ANN model (without epoch outputs)
    dl_model.fit(X.iloc[train], cat_y_train, epochs=7500, batch_size=32, verbose=0)
    # Evaluate the model
    dl_accuracy_scores = dl_model.evaluate(X.iloc[test], cat_y_test, verbose=0)
    # Print accuracy scores to list
    print("%s: %.2f%%" % (dl_model.metrics_names[1], dl_accuracy_scores[1]*100))
    dlcvscores.append(dl_accuracy_scores[1] * 100)
# Print mean and standard deviation for accuracy scores
print("%.2f%% (+/- %.2f%%)" % (np.mean(dlcvscores), np.std(dlcvscores)))
# ---------------------------------------------------------------------------
## Calculate ROC curve and ROC area for each class
n_classes = 3
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(cat_y_test[:, i], dl_model.predict(X_test)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
## Calculate micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(cat_y_test.ravel(), dl_model.predict(X_test).ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
print(roc_auc["micro"])
# ---------------------------------------------------------------------------
## Calculate and visualise SHAP values of the ANN model
dl_explainer = shap.DeepExplainer(dl_model, X_train.values)
dl_shap_values = dl_explainer.shap_values(X_train.values)
# Print SHAP summary plots
def show_shap_plots():
    for i in [0,1,2]:
        print(shap.summary_plot(dl_shap_values[i], X_train, plot_type='bar'))
    print('All three SHAP plots exported.')
show_shap_plots()
