# {
#  "cells": [
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "# Titanic Survival Prediction - Model Development\n",
#     "# Part A: Model Development\n",
#     "\n",
#     "import pandas as pd\n",
#     "import numpy as np\n",
#     "from sklearn.model_selection import train_test_split\n",
#     "from sklearn.ensemble import RandomForestClassifier\n",
#     "from sklearn.preprocessing import LabelEncoder, StandardScaler\n",
#     "from sklearn.metrics import classification_report, accuracy_score\n",
#     "import joblib\n",
#     "import warnings\n",
#     "warnings.filterwarnings('ignore')\n",
#     "\n",
#     "# 1. Load the Titanic dataset\n",
#     "print(\"Loading Titanic dataset...\")\n",
#     "url = \"https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv\"\n",
#     "df = pd.read_csv(url)\n",
#     "\n",
#     "print(f\"Dataset shape: {df.shape}\")\n",
#     "print(f\"\\nFirst few rows:\\n{df.head()}\")\n",
#     "print(f\"\\nDataset info:\")\n",
#     "print(df.info())\n",
#     "\n",
#     "# 2. Data Preprocessing\n",
#     "\n",
#     "# Select 5 features: Pclass, Sex, Age, Fare, Embarked\n",
#     "# Target: Survived\n",
#     "selected_features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Survived']\n",
#     "df = df[selected_features].copy()\n",
#     "\n",
#     "print(f\"\\n--- Data Preprocessing ---\")\n",
#     "print(f\"Selected features: {selected_features[:-1]}\")\n",
#     "print(f\"Target variable: Survived\")\n",
#     "\n",
#     "# a. Handling missing values\n",
#     "print(f\"\\nMissing values before handling:\")\n",
#     "print(df.isnull().sum())\n",
#     "\n",
#     "# Fill Age with median\n",
#     "df['Age'].fillna(df['Age'].median(), inplace=True)\n",
#     "\n",
#     "# Fill Embarked with mode\n",
#     "df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)\n",
#     "\n",
#     "# Drop any remaining rows with missing values\n",
#     "df.dropna(inplace=True)\n",
#     "\n",
#     "print(f\"\\nMissing values after handling:\")\n",
#     "print(df.isnull().sum())\n",
#     "print(f\"Dataset shape after cleaning: {df.shape}\")\n",
#     "\n",
#     "# c. Encoding categorical variables\n",
#     "print(f\"\\n--- Encoding Categorical Variables ---\")\n",
#     "\n",
#     "# Encode Sex (Male=1, Female=0)\n",
#     "le_sex = LabelEncoder()\n",
#     "df['Sex'] = le_sex.fit_transform(df['Sex'])\n",
#     "print(f\"Sex encoding: {dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_)))}\")\n",
#     "\n",
#     "# Encode Embarked\n",
#     "le_embarked = LabelEncoder()\n",
#     "df['Embarked'] = le_embarked.fit_transform(df['Embarked'])\n",
#     "print(f\"Embarked encoding: {dict(zip(le_embarked.classes_, le_embarked.transform(le_embarked.classes_)))}\")\n",
#     "\n",
#     "# b. Feature selection - separate X and y\n",
#     "X = df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]\n",
#     "y = df['Survived']\n",
#     "\n",
#     "print(f\"\\nFeature matrix shape: {X.shape}\")\n",
#     "print(f\"Target vector shape: {y.shape}\")\n",
#     "\n",
#     "# d. Feature scaling\n",
#     "scaler = StandardScaler()\n",
#     "X_scaled = scaler.fit_transform(X)\n",
#     "\n",
#     "print(f\"\\n--- Feature Scaling Applied ---\")\n",
#     "print(f\"Scaled features (first 5 rows):\\n{X_scaled[:5]}\")\n",
#     "\n",
#     "# Split data into training and testing sets\n",
#     "X_train, X_test, y_train, y_test = train_test_split(\n",
#     "    X_scaled, y, test_size=0.2, random_state=42, stratify=y\n",
#     ")\n",
#     "\n",
#     "print(f\"\\nTrain set size: {X_train.shape[0]}\")\n",
#     "print(f\"Test set size: {X_test.shape[0]}\")\n",
#     "\n",
#     "# 3. Implement Random Forest Classifier\n",
#     "print(f\"\\n--- Model Training ---\")\n",
#     "print(f\"Algorithm: Random Forest Classifier\")\n",
#     "\n",
#     "model = RandomForestClassifier(\n",
#     "    n_estimators=100,\n",
#     "    max_depth=10,\n",
#     "    random_state=42,\n",
#     "    min_samples_split=5,\n",
#     "    min_samples_leaf=2\n",
#     ")\n",
#     "\n",
#     "# 4. Train the model\n",
#     "model.fit(X_train, y_train)\n",
#     "print(\"Model training completed!\")\n",
#     "\n",
#     "# Make predictions\n",
#     "y_pred = model.predict(X_test)\n",
#     "\n",
#     "# 5. Evaluate the model\n",
#     "print(f\"\\n--- Model Evaluation ---\")\n",
#     "print(f\"\\nAccuracy Score: {accuracy_score(y_test, y_pred):.4f}\")\n",
#     "print(f\"\\nClassification Report:\")\n",
#     "print(classification_report(y_test, y_pred, target_names=['Did Not Survive', 'Survived']))\n",
#     "\n",
#     "# Feature importance\n",
#     "feature_names = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']\n",
#     "feature_importance = pd.DataFrame({\n",
#     "    'Feature': feature_names,\n",
#     "    'Importance': model.feature_importances_\n",
#     "}).sort_values('Importance', ascending=False)\n",
#     "\n",
#     "print(f\"\\nFeature Importance:\")\n",
#     "print(feature_importance)\n",
#     "\n",
#     "# 6. Save the trained model and preprocessors\n",
#     "print(f\"\\n--- Saving Model ---\")\n",
#     "\n",
#     "# Save model, scaler, and encoders\n",
#     "joblib.dump(model, 'titanic_survival_model.pkl')\n",
#     "joblib.dump(scaler, 'scaler.pkl')\n",
#     "joblib.dump(le_sex, 'le_sex.pkl')\n",
#     "joblib.dump(le_embarked, 'le_embarked.pkl')\n",
#     "\n",
#     "print(\"Model saved as: titanic_survival_model.pkl\")\n",
#     "print(\"Scaler saved as: scaler.pkl\")\n",
#     "print(\"Encoders saved as: le_sex.pkl, le_embarked.pkl\")\n",
#     "\n",
#     "# 7. Demonstrate model reloading and prediction\n",
#     "print(f\"\\n--- Testing Model Reload ---\")\n",
#     "\n",
#     "# Load the saved model\n",
#     "loaded_model = joblib.load('titanic_survival_model.pkl')\n",
#     "loaded_scaler = joblib.load('scaler.pkl')\n",
#     "loaded_le_sex = joblib.load('le_sex.pkl')\n",
#     "loaded_le_embarked = joblib.load('le_embarked.pkl')\n",
#     "\n",
#     "print(\"Model successfully reloaded!\")\n",
#     "\n",
#     "# Test prediction with sample data\n",
#     "sample_data = pd.DataFrame({\n",
#     "    'Pclass': [3],\n",
#     "    'Sex': ['male'],\n",
#     "    'Age': [22],\n",
#     "    'Fare': [7.25],\n",
#     "    'Embarked': ['S']\n",
#     "})\n",
#     "\n",
#     "print(f\"\\nSample passenger data:\")\n",
#     "print(sample_data)\n",
#     "\n",
#     "# Preprocess sample data\n",
#     "sample_data['Sex'] = loaded_le_sex.transform(sample_data['Sex'])\n",
#     "sample_data['Embarked'] = loaded_le_embarked.transform(sample_data['Embarked'])\n",
#     "sample_scaled = loaded_scaler.transform(sample_data)\n",
#     "\n",
#     "# Make prediction\n",
#     "prediction = loaded_model.predict(sample_scaled)\n",
#     "probability = loaded_model.predict_proba(sample_scaled)\n",
#     "\n",
#     "print(f\"\\nPrediction: {'Survived' if prediction[0] == 1 else 'Did Not Survive'}\")\n",
#     "print(f\"Survival Probability: {probability[0][1]:.2%}\")\n",
#     "\n",
#     "print(f\"\\n{'='*50}\")\n",
#     "print(\"Model development completed successfully!\")\n",
#     "print(f\"{'='*50}\")"
#    ]
#   }
#  ],
#  "metadata": {
#   "language_info": {
#    "name": "python"
#   },
#   "orig_nbformat": 4
#  },
#  "nbformat": 4,
#  "nbformat_minor": 2
# }




# Titanic Survival Prediction - Model Development
# Part A: Model Development

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import warnings

warnings.filterwarnings("ignore")

# 1. Load the Titanic dataset
print("Loading Titanic dataset...")
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print(f"Dataset shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

# 2. Data Preprocessing
selected_features = ["Pclass", "Sex", "Age", "Fare", "Embarked", "Survived"]
df = df[selected_features].copy()

print("\n--- Data Preprocessing ---")
print("Missing values before handling:")
print(df.isnull().sum())

# Handle missing values
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
df.dropna(inplace=True)

print("\nMissing values after handling:")
print(df.isnull().sum())

# Encode categorical variables
print("\n--- Encoding Categorical Variables ---")
le_sex = LabelEncoder()
df["Sex"] = le_sex.fit_transform(df["Sex"])

le_embarked = LabelEncoder()
df["Embarked"] = le_embarked.fit_transform(df["Embarked"])

# Split features and target
X = df[["Pclass", "Sex", "Age", "Fare", "Embarked"]]
y = df["Survived"]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Model training
print("\n--- Model Training ---")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    min_samples_split=5,
    min_samples_leaf=2
)

model.fit(X_train, y_train)
print("Model training completed!")

# Evaluation
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_names = ["Pclass", "Sex", "Age", "Fare", "Embarked"]
feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Save model and preprocessors
joblib.dump(model, "titanic_survival_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le_sex, "le_sex.pkl")
joblib.dump(le_embarked, "le_embarked.pkl")

print("\nModel and preprocessors saved successfully!")

# Test reload
loaded_model = joblib.load("titanic_survival_model.pkl")
loaded_scaler = joblib.load("scaler.pkl")
loaded_le_sex = joblib.load("le_sex.pkl")
loaded_le_embarked = joblib.load("le_embarked.pkl")

sample_data = pd.DataFrame({
    "Pclass": [3],
    "Sex": ["male"],
    "Age": [22],
    "Fare": [7.25],
    "Embarked": ["S"]
})

sample_data["Sex"] = loaded_le_sex.transform(sample_data["Sex"])
sample_data["Embarked"] = loaded_le_embarked.transform(sample_data["Embarked"])
sample_scaled = loaded_scaler.transform(sample_data)

prediction = loaded_model.predict(sample_scaled)
probability = loaded_model.predict_proba(sample_scaled)

print("\nPrediction:", "Survived" if prediction[0] == 1 else "Did Not Survive")
print(f"Survival Probability: {probability[0][1]:.2%}")

print("\n" + "=" * 50)
print("Model development completed successfully!")
print("=" * 50)
