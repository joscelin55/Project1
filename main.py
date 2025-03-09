import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# STEP 1: Data Collection (Simulated)
np.random.seed(42)

data = {
    "Job_Title": np.random.choice(["Software Engineer", "Data Scientist", "AI Engineer", "Machine Learning Engineer", "DevOps Engineer"], 3000),
    "Company": np.random.choice(["Google", "Amazon", "Facebook", "Microsoft", "Tesla"], 3000),
    "Location": np.random.choice(["New York, NY", "San Francisco, CA", "Austin, TX", "Seattle, WA", "Boston, MA"], 3000),
    "Skills": np.random.choice(["Python, SQL, Machine Learning", "Java, AWS, Docker", "C++, Deep Learning, TensorFlow", "Python, Pandas, Data Science", "JavaScript, React, Node.js"], 3000),
    "Experience_Level": np.random.choice(["Entry", "Mid", "Senior"], 3000),
    "Salary": np.random.randint(60000, 200000, 3000)
}

df = pd.DataFrame(data)
df.to_csv("job_data.csv", index=False)
print("Dataset created and saved as job_data.csv")

# STEP 2: Data Preprocessing
df = pd.read_csv("job_data.csv")

# Handle missing values (if any)
df.dropna(inplace=True)

# Convert categorical data to numerical labels
encoder = LabelEncoder()
df["Job_Title"] = encoder.fit_transform(df["Job_Title"])
df["Company"] = encoder.fit_transform(df["Company"])
df["Location"] = encoder.fit_transform(df["Location"])
df["Experience_Level"] = encoder.fit_transform(df["Experience_Level"])

# Convert skills into numerical features (One-Hot Encoding)
df = df.join(df["Skills"].str.get_dummies(sep=", "))
df.drop(columns=["Skills"], inplace=True)

# Save preprocessed data
df.to_csv("cleaned_job_data.csv", index=False)
print("Data preprocessing completed and saved as cleaned_job_data.csv")

# STEP 3: Feature Engineering

# Split features and target variable
X = df.drop(columns=["Salary"])
y = df["Salary"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 4: Salary Prediction Models

# Model 1: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Evaluate Model 1
mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
print(f"Linear Regression - MAE: {mae_lr}, RMSE: {rmse_lr}")

# Model 2: Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluate Model 2
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print(f"Random Forest Regressor - MAE: {mae_rf}, RMSE: {rmse_rf}")

# STEP 5: Skill Importance Analysis
feature_importance = rf_model.feature_importances_
importance_df = pd.DataFrame({"Feature": X.columns, "Importance": feature_importance}).sort_values(by="Importance", ascending=False)

# Plot Feature Importance
plt.figure(figsize=(12, 6))
sns.barplot(x=importance_df["Importance"], y=importance_df["Feature"])
plt.title("Skill Importance for Salary Prediction")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# STEP 6: Visualizations

# Load the original dataset for visualization
df_plot = pd.read_csv("job_data.csv")

# Salary Distribution by Job Role
fig = px.box(df_plot, x="Job_Title", y="Salary", color="Job_Title", title="Salary Distribution by Job Role")
fig.show()

# Average Salary by Location (Heatmap)
salary_by_location = df_plot.groupby("Location")["Salary"].mean().reset_index()

fig = go.Figure(data=go.Heatmap(
    z=salary_by_location["Salary"],
    x=salary_by_location["Location"],
    y=["Salary"] * len(salary_by_location),
    colorscale="Blues"))

fig.update_layout(title="Average Salary by Location")
fig.show()
