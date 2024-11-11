import pandas as pd
from imblearn.over_sampling import ADASYN

# Read the dataset using pandas
data = pd.read_csv('stroke_dataset_one_hot.csv')

# Separate the features (X) and the labels (y)
X = data.drop('stroke', axis=1)  # Replace 'target_column' with the actual column name
y = data['stroke']  # Replace 'target_column' with the actual column name

# Apply ADASYN to oversample the minority class
adasyn = ADASYN(random_state=43)
X_resampled, y_resampled = adasyn.fit_resample(X, y)

# Create a new DataFrame with the balanced data
balanced_data = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled)], axis=1)

balanced_data.drop(balanced_data.columns[0], axis=1)

#unamed columns
#unamed_column = [col for col in data.columns if col.startswith('Unnamed:') or col == '']
#balanced_data = data.drop(columns = unamed_column)
# Save the balanced data to a new CSV file
balanced_data.to_csv('ADASYN_one_hot.csv', index=False)
adasynData = pd.read_csv('ADASYN_one_hot.csv')
unamed_column = [col for col in adasynData.columns if col.startswith('Unnamed:') or col == '']
adasynData = adasynData.drop(columns = unamed_column)

adasynData.drop(adasynData.columns[1], axis=1)

adasynData['bmi'] = adasynData['bmi'].round(2)
adasynData['age'] = adasynData['age'].astype(int)
adasynData.to_csv('ADASYN_one_hot.csv')
