import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
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
features_runs = data[['ball_faced'] + list(against_team_encoded_df.columns)]
target_runs = data['run_scored']
features_wickets = data[['ball_delivered', 'run_given'] + list(against_team_encoded_df.columns)]
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
    cluster_features = cluster_data[['ball_faced'] + list(against_team_encoded_df.columns)]
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
    cluster_features = cluster_data[['ball_delivered', 'run_given'] + list(against_team_encoded_df.columns)]
    cluster_target = cluster_data['wicket']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(cluster_features, cluster_target)
    models_wickets[cluster_id] = model

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
            input_data_runs = pd.DataFrame([[avg_balls_faced] + list(against_team_encoded_df.iloc[0])], columns=features_runs.columns)
            input_data_wickets = pd.DataFrame([[avg_ball_delivered, avg_run_given] + list(against_team_encoded_df.iloc[0])], columns=features_wickets.columns)
            
            # Predict the cluster for the input data
            cluster_id_runs = kmeans_runs.predict(input_data_runs)[0]
            cluster_id_wickets = kmeans_wickets.predict(input_data_wickets)[0]
            
            # Use the corresponding model for prediction
            predicted_runs = models_runs[cluster_id_runs].predict(input_data_runs)[0]
            predicted_wickets = models_wickets[cluster_id_wickets].predict(input_data_wickets)[0]
            
            predictions[player_name] = {
                'predicted_runs': predicted_runs,
                'predicted_wickets': predicted_wickets
            }
        else:
            predictions[player_name] = {
                'predicted_runs': None,
                'predicted_wickets': None
            }
    return predictions

# Define the points system
points_system = {
    'Run': 1,
    '4s': 1,
    '6s': 2,
    '50s': 4,
    '100s': 8,
    'Wicket': 25,
    'catch': 8,
    'stump': 12,
    'run_out': 6,
    'Economy_rate': {
        '5rpo': 6, '6rpo': 4, '7rpo': 2, '10rpo': -2, '11rpo': -4, '12rpo': -6
    }
}

# Function to calculate fantasy points
def calculate_fantasy_points(predictions, data, against_team):
    fantasy_points = {}
    for player_name, prediction in predictions.items():
        player_data = data[data['player'] == player_name]
        if not player_data.empty:
            total_points = 0
            predicted_runs = prediction['predicted_runs']
            predicted_wickets = prediction['predicted_wickets']
            
            if predicted_runs is not None:
                total_points += predicted_runs * points_system['Run']
                
                player_data_team = player_data[player_data['against_team'] == against_team]
                if not player_data_team.empty:
                    total_4s = player_data_team['4s'].sum()
                    total_6s = player_data_team['6s'].sum()
                    total_50s = player_data_team['50s'].sum()
                    total_100s = player_data_team['100s'].sum()
                    
                    total_points += total_4s * points_system['4s']
                    total_points += total_6s * points_system['6s']
                    total_points += total_50s * points_system['50s']
                    total_points += total_100s * points_system['100s']
            
            if predicted_wickets is not None:
                total_points += predicted_wickets * points_system['Wicket']
                
                total_catches = player_data_team['catch'].sum()
                total_stumps = player_data_team['stump'].sum()
                total_runouts = player_data_team['run_out'].sum()
                
                total_points += total_catches * points_system['catch']
                total_points += total_stumps * points_system['stump']
                total_points += total_runouts * points_system['run_out']
            
            fantasy_points[player_name] = total_points
        else:
            fantasy_points[player_name] = 0
    return fantasy_points

# Define the two teams of players
team1_players = ['Abdul Samad','Abhishek Sharma','RA Tripathi','H Klaasen','TM Head','B Kumar', 'T Natarajan','AK Markram','Shahbaz Ahmed','PJ Cummins','Nithish Kumar Reddy']
team2_players = ['VG Arora','AD Russell','Ramandeep Singh', 'SP Narine','Rahmanullah Gurbaz','SS Iyer', 'RK Singh', 'MA Starc', 'Harshit Rana','VR Iyer', 'Varun Chakravarthy']

# Define the opponent teams for each team
team1_against_team = 'Kolkata Knight Riders'
team2_against_team = 'Sunrisers Hyderabad'

# Predict runs and wickets for both teams
team1_predictions = predict_runs_and_wickets(team1_players, team1_against_team, models_runs, models_wickets, data, encoder, kmeans_runs, kmeans_wickets)
team2_predictions = predict_runs_and_wickets(team2_players, team2_against_team, models_runs, models_wickets, data, encoder, kmeans_runs, kmeans_wickets)

# Calculate fantasy points for both teams
team1_fantasy_points = calculate_fantasy_points(team1_predictions, data, team1_against_team)
team2_fantasy_points = calculate_fantasy_points(team2_predictions, data, team2_against_team)

# Combine fantasy points from both teams
all_fantasy_points = {**team1_fantasy_points, **team2_fantasy_points}

# Rank players based on their fantasy points
ranked_players = sorted(all_fantasy_points.items(), key=lambda x: x[1], reverse=True)

# Print the top 11 players based on fantasy points
print("Top 11 players based on fantasy points:")
for player, points in ranked_players[:11]:
    print(f'{player}: {points:.2f} points')
