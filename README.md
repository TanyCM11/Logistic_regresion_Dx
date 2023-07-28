# Logistic_regresion_Dx
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Read the data from the aud1 library
data = pd.read_sas(XXXXXX)

# Convert Sex variable from character to numeric
data['Sex_num'] = np.where(data['Sex'] == "Female", 0,
                           np.where(data['Sex'] == "Male", 1,
                                    np.where(data['Sex'] == "Unknown", 2, np.nan)))

# Combine Sexnum=2 into either SexNum=0 or SexNum=1
data['Sex_num'] = np.where(data['Sex_num'] == 2, 0, data['Sex_num'])

# Convert FirstRace variable from character to numeric
data['FirstRace_num'] = np.where(data['FirstRace'] == "Asian", 1,
                                np.where(data['FirstRace'] == "Black or African American", 2,
                                         np.where(data['FirstRace'] == "White or Caucasian", 3,
                                                  np.where(data['FirstRace'].isin(["Asian", "Black or African American", "White or Caucasian"]), 4, np.nan))))

# Recode individuals with multiple races
data.loc[(data['SecondRace'].notna()) | (data['ThirdRace'].notna()), 'FirstRace_num'] = 5

# Convert Ethnicity variable from character to numeric
data['Ethnicity_num'] = np.where(data['Ethnicity'].isin(["Hispanic, Latinx, or Spanish Origin", "Mexican, Mexican American, or Chicano/a"]), 1,
                                 np.where(data['Ethnicity'] == "Not Hispanic, Latinx, or Spanish Origin", 2,
                                          np.where(data['Ethnicity'].isin(["Hispanic, Latinx, or Spanish Origin", "Mexican, Mexican American, or Chicano/a", "Not Hispanic, Latinx, or Spanish Origin"]), 3, np.nan)))

# Create CombinedRaceEthnicity variable
data['CombinedRaceEthnicity'] = np.where((data['FirstRace_num'] == 1) & (data['Ethnicity_num'] == 2), 0,
                                         np.where((data['FirstRace_num'] == 2) & (data['Ethnicity_num'] == 2), 1,
                                                  np.where((data['FirstRace_num'] == 3) & (data['Ethnicity_num'] == 2), 2,
                                                           np.where(data['Ethnicity_num'] == 1, 3,
                                                                    np.where(((data['FirstRace_num'] == 4) | (data['FirstRace_num'] == 5)) & (data['Ethnicity_num'] == 2)) | (data['Ethnicity_num'] == 3), 4,
                                                                             np.where((data['FirstRace_num'].isna()) | (data['Ethnicity_num'].isna()), 5, np.nan))))))

# Convert SCREEN_RESULT variable from character to numeric
data['screen_result_num'] = np.where(data['screen_result'] == "Negative", 0, 1)

# Bin the Age variable into age groups starting at 12 with a group size of 20
age_bins = [12, 31, 51, 71, 98]
age_labels = [0, 1, 2, 3]
data['Age_group'] = pd.cut(data['Age'], bins=age_bins, labels=age_labels, include_lowest=True)

# Convert PatientDxSeverity variable from character to numeric
data['NumDx'] = np.where(data['PatientDxSeverity2021'].isin(["Dependence", "Abuse", "Use", "Alcohol-related health problems"]), 0, 1)

# Filter the dataset to include only individuals between ages 12 and 21
filtered_data = data[data['Age_group'].isin([0, 1, 2])]

# Run the frequency analysis on the filtered dataset
frequency_table = pd.crosstab(filtered_data['NumDx'], filtered_data['Age_group'], dropna=False)

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.3, random_state=12345)

# Fit a logistic regression model on the training data
features = ['Age_group', 'Sex_num', 'auditCalcTotal', 'CombinedRaceEthnicity']
target = 'NumDx'
logreg = LogisticRegression()
logreg.fit(train_data[features], train_data[target])

# Predict on the testing data
test_data['predicted_probability'] = logreg.predict_proba(test_data[features])[:, 1]

# Calculate the area under the ROC curve
fpr, tpr, thresholds = metrics.roc_curve(test_data[target], test_data['predicted_probability'])
roc_auc = metrics.auc(fpr, tpr)

# Print the area under the ROC curve
print("Area under the ROC curve:", roc_auc)

# Calculate the number of observations in the filtered dataset
nobs = filtered_data.shape[0]
print("Number of observations:", nobs)
