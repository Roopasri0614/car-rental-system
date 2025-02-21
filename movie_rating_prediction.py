import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
file_path = r"C:\Users\roopa\Downloads\movie\IMDb Movies India.csv"
df = pd.read_csv(file_path, encoding="ISO-8859-1")

# Rename columns for consistency
df.rename(columns={
    'Genre': 'genre', 
    'Director': 'director', 
    'Actor 1': 'actor_1', 
    'Actor 2': 'actor_2', 
    'Actor 3': 'actor_3', 
    'Rating': 'rating',
    'Votes': 'votes',
    'Year': 'year',
    'Duration': 'duration'
}, inplace=True)

# Selecting relevant columns
df = df[['genre', 'director', 'actor_1', 'actor_2', 'actor_3', 'votes', 'year', 'duration', 'rating']]

# Handling missing values
df.dropna(inplace=True)

# Encoding categorical data
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
df[['genre', 'director', 'actor_1', 'actor_2', 'actor_3']] = encoder.fit_transform(df[['genre', 'director', 'actor_1', 'actor_2', 'actor_3']])

# Save encoder
joblib.dump(encoder, "encoder.pkl")

# Train-test split
X = df.drop(columns=['rating'])  # Features
y = df['rating']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
lr_model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Evaluate models
y_pred_lr = lr_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

print("Linear Regression Performance:")
print("MSE:", mean_squared_error(y_test, y_pred_lr))
print("R2 Score:", r2_score(y_test, y_pred_lr))

print("\nRandom Forest Performance:")
print("MSE:", mean_squared_error(y_test, y_pred_rf))
print("R2 Score:", r2_score(y_test, y_pred_rf))

# Save model
joblib.dump(rf_model, "movie_rating_model.pkl")

# Load and predict
loaded_model = joblib.load("movie_rating_model.pkl")
encoder = joblib.load("encoder.pkl")

# Prepare a sample input
sample_input = pd.DataFrame([X_train.iloc[0]], columns=X.columns)  # Retains feature names
predicted_rating = loaded_model.predict(sample_input)
print("\nPredicted IMDb Rating:", predicted_rating[0])
