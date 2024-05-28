import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder

# Load the transformed match data
transformed_file = 'transformed_match_data.csv'
data = pd.read_csv(transformed_file)

# Preprocess the data
data['ball_faced'] = data['ball_faced'].astype(int)
data['run_scored'] = data['run_scored'].astype(int)

# One-hot encode the against_team column
encoder = OneHotEncoder(drop='first')
against_team_encoded = encoder.fit_transform(data[['against_team']])
against_team_encoded_df = pd.DataFrame(against_team_encoded.toarray(), columns=encoder.get_feature_names_out(['against_team']))

# Reset indices of both dataframes
data.reset_index(drop=True, inplace=True)
against_team_encoded_df.reset_index(drop=True, inplace=True)

# Concatenate the encoded against_team columns with the original dataframe
data = pd.concat([data, against_team_encoded_df], axis=1)

# Select relevant features and target variable
features = data[['ball_faced'] + list(against_team_encoded_df.columns)]
target = data['run_scored']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Function to predict runs against a specific team
def predict_runs(player_name, against_team, model, data, encoder):
    player_data = data[data['player'] == player_name]
    if not player_data.empty:
        # One-hot encode the against_team column for the new input
        against_team_encoded = encoder.transform([[against_team]])
        against_team_encoded_df = pd.DataFrame(against_team_encoded.toarray(), columns=encoder.get_feature_names_out(['against_team']))
        
        # Use the average balls faced by the player
        avg_balls_faced = player_data['ball_faced'].mean()
        
        # Create a dataframe with the features for prediction
        input_data = pd.DataFrame([[avg_balls_faced] + list(against_team_encoded_df.iloc[0])], columns=features.columns)
        
        predicted_runs = model.predict(input_data)
        return predicted_runs[0]
    else:
        return None

# Input player name and team
player_name = input("Enter player name: ")
against_team = input("Enter against team: ")

# Predict runs against the specified team
predicted_runs = predict_runs(player_name, against_team, model, data, encoder)

if predicted_runs is not None:
    print(f"Predicted runs scored by {player_name} against {against_team}: {predicted_runs:.2f}")
else:
    print(f"No data available for {player_name} against {against_team}")
