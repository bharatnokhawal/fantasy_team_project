import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.preprocessing import OneHotEncoder

# Load the transformed match data
transformed_file = 'transformed_match_data.csv'
data = pd.read_csv(transformed_file)

# Preprocess the data
data['ball_faced'] = data['ball_faced'].astype(int)
data['run_scored'] = data['run_scored'].astype(int)
data['ball_delivered'] = data['ball_delivered'].astype(int)
data['run_given'] = data['run_given'].astype(int)
data['wicket'] = data['wicket'].astype(int)
data['4s'] = data['4s'].astype(int)
data['6s'] = data['6s'].astype(int)
data['50s'] = data['50s'].astype(int)
data['100s'] = data['100s'].astype(int)
data['catch'] = data['catch'].astype(int)
data['stump'] = data['stump'].astype(int)
data['run_out'] = data['run_out'].astype(int)

# One-hot encode the against_team column
encoder = OneHotEncoder(drop='first')
against_team_encoded = encoder.fit_transform(data[['against_team']])
against_team_encoded_df = pd.DataFrame(against_team_encoded.toarray(), columns=encoder.get_feature_names_out(['against_team']))

# Reset indices of both dataframes
data.reset_index(drop=True, inplace=True)
against_team_encoded_df.reset_index(drop=True, inplace=True)

# Concatenate the encoded against_team columns with the original dataframe
data = pd.concat([data, against_team_encoded_df], axis=1)

# Select relevant features for runs and wickets
features_runs = data[['ball_faced', '4s', '6s', '50s', '100s'] + list(against_team_encoded_df.columns)]
target_runs = data['run_scored']
features_wickets = data[['ball_delivered', 'run_given', 'catch', 'stump', 'run_out'] + list(against_team_encoded_df.columns)]
target_wickets = data['wicket']

# Perform cluster analysis
kmeans_runs = KMeans(n_clusters=5, random_state=42)
data['cluster_runs'] = kmeans_runs.fit_predict(features_runs)
kmeans_wickets = KMeans(n_clusters=5, random_state=42)
data['cluster_wickets'] = kmeans_wickets.fit_predict(features_wickets)

# Split the data into training and testing sets for runs
X_train_runs, X_test_runs, y_train_runs, y_test_runs = train_test_split(features_runs, target_runs, test_size=0.2, random_state=42)

# Train a Random Forest Regressor model for each cluster (runs)
models_runs = {}
for cluster_id in range(kmeans_runs.n_clusters):
    cluster_data = data[data['cluster_runs'] == cluster_id]
    cluster_features = cluster_data[['ball_faced', '4s', '6s', '50s', '100s'] + list(against_team_encoded_df.columns)]
    cluster_target = cluster_data['run_scored']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(cluster_features, cluster_target)
    models_runs[cluster_id] = model

# Split the data into training and testing sets for wickets
X_train_wickets, X_test_wickets, y_train_wickets, y_test_wickets = train_test_split(features_wickets, target_wickets, test_size=0.2, random_state=42)

# Train a Random Forest Classifier model for each cluster (wickets)
models_wickets = {}
for cluster_id in range(kmeans_wickets.n_clusters):
    cluster_data = data[data['cluster_wickets'] == cluster_id]
    cluster_features = cluster_data[['ball_delivered', 'run_given', 'catch', 'stump', 'run_out'] + list(against_team_encoded_df.columns)]
    cluster_target = cluster_data['wicket']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(cluster_features, cluster_target)
    models_wickets[cluster_id] = model

# Evaluate the models (runs)
mae_cluster_runs = {}
for cluster_id, model in models_runs.items():
    cluster_test_data = data[data['cluster_runs'] == cluster_id]
    X_cluster_test = cluster_test_data[['ball_faced', '4s', '6s', '50s', '100s'] + list(against_team_encoded_df.columns)]
    y_cluster_test = cluster_test_data['run_scored']
    
    y_cluster_pred = model.predict(X_cluster_test)
    mae_cluster_runs[cluster_id] = mean_absolute_error(y_cluster_test, y_cluster_pred)

# Evaluate the models (wickets)
accuracy_cluster_wickets = {}
for cluster_id, model in models_wickets.items():
    cluster_test_data = data[data['cluster_wickets'] == cluster_id]
    X_cluster_test = cluster_test_data[['ball_delivered', 'run_given', 'catch', 'stump', 'run_out'] + list(against_team_encoded_df.columns)]
    y_cluster_test = cluster_test_data['wicket']
    
    y_cluster_pred = model.predict(X_cluster_test)
    accuracy_cluster_wickets[cluster_id] = accuracy_score(y_cluster_test, y_cluster_pred)

# Define the point system
points_system = {
    'Run': 1, '4s': 1, '6s': 2, '50s': 4, '100s': 8,
    'Wicket': 25, 'catch': 8, 'stump': 12, 'run_out': 6,
    'Economy_rate': {
        '5rpo': 6, '6rpo': 4, '7rpo': 2, '10rpo': -2, '11rpo': -4, '12rpo': -6
    }
}

# Function to calculate fantasy points
def calculate_fantasy_points(predictions, data):
    fantasy_points = {}
    for player_name, prediction in predictions.items():
        if prediction['predicted_runs'] is not None and prediction['predicted_wickets'] is not None:
            player_data = data[data['player'] == player_name]
            total_points = 0

            # Calculate batting points
            predicted_runs = prediction['predicted_runs']
            total_points += predicted_runs * points_system['Run']

            # Calculate boundary points
            total_points += player_data['4s'].mean() * points_system['4s']
            total_points += player_data['6s'].mean() * points_system['6s']

            # Calculate milestone points
            if predicted_runs >= 50:
                total_points += points_system['50s']
            if predicted_runs >= 100:
                total_points += points_system['100s']

            # Calculate bowling points
            predicted_wickets = prediction['predicted_wickets']
            total_points += predicted_wickets * points_system['Wicket']

            # Calculate fielding points
            total_points += player_data['catch'].mean() * points_system['catch']
            total_points += player_data['stump'].mean() * points_system['stump']
            total_points += player_data['run_out'].mean() * points_system['run_out']

            fantasy_points[player_name] = total_points
        else:
            fantasy_points[player_name] = 0
    return fantasy_points

# Function to predict runs and wickets against a specific team using clustering
def predict_runs_and_wickets(player_names, against_team, models_runs, models_wickets, data, encoder, kmeans_runs, kmeans_wickets):
    predictions = {}
    for player_name in player_names:
        player_data = data[data['player'] == player_name]
        if not player_data.empty:
            # One-hot encode the against_team column for the new input
            against_team_encoded = encoder.transform([[against_team]])
            against_team_encoded_df = pd.DataFrame(against_team_encoded.toarray(), columns=encoder.get_feature_names_out(['against_team']))
            
            # Use the average balls faced by the player for runs prediction
            avg_balls_faced = player_data['ball_faced'].mean()
            # Use the average ball delivered and run given by the player for wickets prediction
            avg_ball_delivered = player_data['ball_delivered'].mean()
            avg_run_given = player_data['run_given'].mean()
            
            # Create dataframes with the features for prediction
            input_data_runs = pd.DataFrame([[avg_balls_faced, player_data['4s'].mean(), player_data['6s'].mean(), player_data['50s'].mean(), player_data['100s'].mean()] + list(against_team_encoded_df.iloc[0])], columns=features_runs.columns)
            input_data_wickets = pd.DataFrame([[avg_ball_delivered, avg_run_given, player_data['catch'].mean(), player_data['stump'].mean(), player_data['run_out'].mean()] + list(against_team_encoded_df.iloc[0])], columns=features_wickets.columns)
            
            # Predict cluster
            cluster_runs = kmeans_runs.predict(input_data_runs)[0]
            cluster_wickets = kmeans_wickets.predict(input_data_wickets)[0]
            
            # Predict runs and wickets using the models
            predicted_runs = models_runs[cluster_runs].predict(input_data_runs)[0]
            predicted_wickets = models_wickets[cluster_wickets].predict(input_data_wickets)[0]
            
            predictions[player_name] = {'predicted_runs': predicted_runs, 'predicted_wickets': predicted_wickets}
        else:
            predictions[player_name] = {'predicted_runs': None, 'predicted_wickets': None}
    
    return predictions

# Example player names and the team they are playing against
player_names =  ['SV Samson', 'Avesh Khan', 'YBK Jaiswal', 'TA Boult', 'R Parag', 'SO Hetmyer', 'R Ashwin', 'Dhruv Jurel', 'YS Chahal', 'R Powell', 'Sandeep Sharma']
against_team = 'Royal Challengers Bangalore'

# Predict runs and wickets for the players against the specified team
predictions = predict_runs_and_wickets(player_names, against_team, models_runs, models_wickets, data, encoder, kmeans_runs, kmeans_wickets)

# Calculate fantasy points based on the predictions
fantasy_points = calculate_fantasy_points(predictions, data)

print("Fantasy Points for Players:")
for player, points in fantasy_points.items():
    print(f"{player}: {points} points")
