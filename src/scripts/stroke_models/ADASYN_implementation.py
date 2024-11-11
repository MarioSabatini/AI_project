import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import ADASYN

# Read the dataset using pandas
data_standard = pd.read_csv('../../../data/stroke_datasets/stroke_dataset_one_hot.csv')
data_adasyn = pd.read_csv('../../../data/stroke_datasets/ADASYN_one_hot.csv')

def plot_stroke_pie_chart(dataframe):
    #Pie Plot
    stroke_0 = dataframe.loc[dataframe["stroke"] == 0, :].shape[0]
    stroke_1 = dataframe.loc[dataframe["stroke"] == 1, :].shape[0]

    labels = ["With stroke", "Without stroke"]
    sizes = [stroke_1, stroke_0]
    colors = ["#F57163", "#B2F57E"]
    plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", textprops={'fontsize': 18})
    plt.title("Number of record per 'stroke'")
    plt.plot()
    
plt.figure(figsize=(20, 8), dpi=80)  # Adjust width, height, and dpi as needed
plt.subplot(1, 2, 1)
plot_stroke_pie_chart(data_standard)
plt.subplot(1, 2, 2)
plot_stroke_pie_chart(data_adasyn)
plt.tight_layout()
plt.show()

X_standard = data_standard.drop(['stroke','id'], axis=1)
y_standard = data_standard['stroke']

start = 42
maxIter = 30
step = 100
results_random_forest = []
results_k = []
results_log_reg = []
i = 0
while i < maxIter:
    seed = np.random.seed(42 + step)
    # Apply ADASYN to oversample the minority class
    adasyn = ADASYN(random_state=seed)
    X_train_sta, X_test_sta, y_train_sta, y_test_sta = train_test_split(X_standard, y_standard, random_state=seed)
   
    X_test_ada, y_test_ada = adasyn.fit_resample(X_test_sta, y_test_sta)

    # Fit models on X_train and y_train without oversampling
    model_random_forest =  RandomForestClassifier().fit(X_train_sta, y_train_sta)
    model_k = KNeighborsClassifier().fit(X_train_sta, y_train_sta)
    model_log_reg = LogisticRegression().fit(X_train_sta, y_train_sta)

    # Prediction on oversampled X_test_sta (X_test_ada)
    pred_random_forest = model_random_forest.predict(X_test_ada)
    pred_k = model_k.predict(X_test_ada)
    pred_log_reg = model_log_reg.predict(X_test_ada)

    # Accuracy for the models
    accuracy_random_forest = accuracy_score(y_test_ada, pred_random_forest)
    accuracy_k = accuracy_score(y_test_ada, pred_k)
    accuracy_log_reg = accuracy_score(y_test_ada, pred_log_reg)

    # Append the result of each iteration for the accuracy of the models in a list 
    results_random_forest.append(accuracy_random_forest)
    results_k.append(accuracy_k)
    results_log_reg.append(accuracy_log_reg)
    #print("iteration number: " + str(i) + " With seed number: " + str(42+step))
    i = i + 1
    step = step + 100

# Compute the variance for the models
variance_random_forest = np.var(results_random_forest)
variance_k = np.var(results_k)
variance_log_reg = np.var(results_log_reg)

print("Random Forest Variance: ", variance_random_forest)
print("K Neighbour Variance: ", variance_k)
print("Logistic Regression Variance: ", variance_log_reg)

# Apply ADASYN to oversample the minority class
adasyn = ADASYN(random_state=start)
X_train_sta, X_test_sta, y_train_sta, y_test_sta = train_test_split(X_standard, y_standard, random_state=start)

X_test_ada, y_test_ada = adasyn.fit_resample(X_test_sta, y_test_sta)

#Check frequency stroke
y_test_sta_df = pd.DataFrame({'stroke': y_test_sta})
plot_stroke_pie_chart(y_test_sta_df)
plt.show()

#------------ KNeighborsClassifier ------------
model_k =  KNeighborsClassifier().fit(X_train_sta, y_train_sta)
pred_k = model_k.predict(X_test_ada)

#------------ RandomForestClassifier ------------
model_random_forest =  RandomForestClassifier().fit(X_train_sta, y_train_sta)
pred_random_forest = model_random_forest.predict(X_test_ada)

#------------ LogisticRegression ------------
model_logistic =  LogisticRegression().fit(X_train_sta, y_train_sta)
pred_logistic = model_logistic.predict(X_test_ada)

# Calculate the confusion matrix for the model KNeighborsClassifier
confusion_matrix_k = confusion_matrix(y_test_ada, pred_k)
accuracy_k = accuracy_score(y_test_ada, pred_k)*100
recall_k = recall_score(y_test_ada, pred_k)

# Calculate the confusion matrix for the model RandomForestClassifier
confusion_matrix_random_forest = confusion_matrix(y_test_ada, pred_random_forest)
accuracy_random_forest = accuracy_score(y_test_ada, pred_random_forest)*100
recall_random_forest = recall_score(y_test_ada, pred_random_forest)

# Calculate the confusion matrix for the model LogisticRegression
confusion_matrix_logistic = confusion_matrix(y_test_ada, pred_logistic)
accuracy_logistic = accuracy_score(y_test_ada, pred_logistic)*100
recall_logistic = recall_score(y_test_ada, pred_logistic)

def plot_matrix_model(model,accuracy,recall,confusion_matrix):
    #plot matrix
    #plt.ion()
    group_names = ['True Neg','False Pos','False Neg','True Pos']

    group_counts = ["{0:0.0f}".format(value) for value in
                    confusion_matrix.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in
                        confusion_matrix.flatten()/np.sum(confusion_matrix)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
            zip(group_names,group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(2,2)

    ax = sns.heatmap(confusion_matrix, annot=labels, fmt='', cmap='Blues')

    ax.set_title("Confusion Matrix {:s} \nAccuracy={:.2f}  Recall={:.2f}".format(model,accuracy, recall));
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])

    ## Display the visualization of the Confusion Matrix.
    plt.plot()

plt.subplot(1, 3, 1)
plot_matrix_model("KNeighborsClassifier",accuracy_k,recall_k,confusion_matrix_k)
plt.subplot(1, 3, 2)
plot_matrix_model("RandomForestClassifier",accuracy_random_forest,recall_random_forest,confusion_matrix_random_forest)
plt.subplot(1, 3, 3)
plot_matrix_model("LogisticRegression",accuracy_logistic,recall_logistic,confusion_matrix_logistic)
plt.show()
print(model_random_forest.feature_importances_)

#Compute the explainer for shap for the models
explainer_random_forest = shap.Explainer(model_random_forest.predict,X_test_ada)
explainer_k = shap.Explainer(model_random_forest.predict,X_test_ada)
explainer_linear = shap.Explainer(model_random_forest.predict,X_test_ada)

#Compute the shap values for the models
shap_values_random_forest = explainer_random_forest(X_standard[0:100])
shap_values_k = explainer_k(X_standard[0:100])
shap_values_linear = explainer_linear(X_standard[0:100])

#waterfall plot for the models
shap.plots.waterfall(shap_values_random_forest[0], show=False)
plt.title("Waterfall plot for Random Forest")
plt.show()

shap.plots.waterfall(shap_values_k[0], show=False)
plt.title("Waterfall plot for KNeighbors")
plt.show()

shap.plots.waterfall(shap_values_linear[0], show=False)
plt.title("Waterfall plot for Logistic Regression")
plt.show()

#bar plot for models
shap.plots.bar(shap_values_random_forest, show=False)
plt.title("bar plot for Random Forest")
plt.show()

shap.plots.bar(shap_values_k, show=False)
plt.title("Waterfall plot for KNeighbors")
plt.show()

shap.plots.bar(shap_values_linear, show=False)
plt.title("bar plot for Logistic Regression")
plt.show()

#beeswarm plot for models
shap.plots.beeswarm(shap_values_random_forest, show=False)
plt.title("Beeswarm plot for Random Forest")
plt.show()

shap.plots.beeswarm(shap_values_k, show=False)
plt.title("Beeswarm plot for KNeighbors")
plt.show()

shap.plots.beeswarm(shap_values_linear, show=False)
plt.title("Beeswarm plot for Logistic Regression")
plt.show()