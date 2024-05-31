import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings("ignore")

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

# Evaluate the models (runs)
mae_cluster_runs = {}
for cluster_id, model in models_runs.items():
    cluster_test_data = data[data['cluster_runs'] == cluster_id]
    X_cluster_test = cluster_test_data[['ball_faced'] + list(against_team_encoded_df.columns)]
    y_cluster_test = cluster_test_data['run_scored']
    
    y_cluster_pred = model.predict(X_cluster_test)
    mae_cluster_runs[cluster_id] = mean_absolute_error(y_cluster_test, y_cluster_pred)

# Evaluate the models (wickets)
accuracy_cluster_wickets = {}
for cluster_id, model in models_wickets.items():
    cluster_test_data = data[data['cluster_wickets'] == cluster_id]
    X_cluster_test = cluster_test_data[['ball_delivered', 'run_given'] + list(against_team_encoded_df.columns)]
    y_cluster_test = cluster_test_data['wicket']
    
    y_cluster_pred = model.predict(X_cluster_test)
    accuracy_cluster_wickets[cluster_id] = accuracy_score(y_cluster_test, y_cluster_pred)

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

# Function to calculate impact score based on predicted runs and wickets
def calculate_impact_score(predictions):
    impact_scores = {}
    for player_name, prediction in predictions.items():
        if prediction['predicted_runs'] is not None and prediction['predicted_wickets'] is not None:
            # Assuming impact score is a combination of predicted runs and predicted wickets with weights
            impact_score = (prediction['predicted_runs'] * 1.4) + (prediction['predicted_wickets'] * 25)
            impact_scores[player_name] = impact_score
        else:
            impact_scores[player_name] = None
    return impact_scores

# Input team names and against teams
teams = [['MS Dhoni', 'Shaik Rasheed', 'Shivam Dube', 'RD Gaikwad', 'DL Chahar', 'RA Jadeja', 'AM Rahane', 'M Theekshana', 'TU Deshpande', 'Simarjeet Singh', 'MM Ali'], ['Rashid Khan', 'Shubman Gill', 'Mohammed Shami', 'WP Saha', 'DA Miller', 'V Shankar', 'MS Wade', 'J Yadav', 'KS Williamson', 'R Sai Kishore', 'MM Sharma']]
against_teams = ['Gujarat Titans', 'Chennai Super Kings']

# Predict runs, wickets, and calculate impact scores for each team
all_impact_scores = {}
for i, team in enumerate(teams):
    against_team = against_teams[i]
    predictions = predict_runs_and_wickets(team, against_team, models_runs, models_wickets, data, encoder, kmeans_runs, kmeans_wickets)
    impact_scores = calculate_impact_score(predictions)
    all_impact_scores.update(impact_scores)

# Load data into DataFrames for fantasy points calculation
df = pd.read_csv('transformed_match_data.csv')
cricket_data = pd.read_csv('IPl Ball-by-Ball 2008-2023.csv')

# Define functions to calculate performance against a specific team and player
def performance_against_team(player_name, team_name, df):
    player_team_data = df[(df['player'] == player_name) & (df['against_team'] == team_name)]
    matches_played = player_team_data['match_id'].nunique()

    total_runs = player_team_data['run_scored'].sum()
    total_wickets = player_team_data['wicket'].sum()
    total_4s = player_team_data['4s'].sum()
    total_6s = player_team_data['6s'].sum()
    total_50s = len(player_team_data[player_team_data['run_scored'] >= 50])
    total_100s = len(player_team_data[player_team_data['run_scored'] >= 100])

    return {
        'matches_played': matches_played,
        'total_runs': total_runs,
        'total_wickets': total_wickets,
        'total_4s': total_4s,
        'total_6s': total_6s,
        'total_50s': total_50s,
        'total_100s': total_100s
    }

def calculate_fantasy_points(player_stats):
    points = 0

    # Add runs points
    points += player_stats['total_runs'] * 1.4

    # Add boundaries points
    points += player_stats['total_4s'] * 1
    points += player_stats['total_6s'] * 2

    # Add wickets points
    points += player_stats['total_wickets'] * 25

    # Add milestone points
    points += player_stats['total_50s'] * 8
    points += player_stats['total_100s'] * 16

    return points

# Calculate fantasy points and impact scores for each player
fantasy_points = {}
for team in teams:
    for player_name in team:
        player_stats = performance_against_team(player_name, against_team, df)
        fantasy_points[player_name] = calculate_fantasy_points(player_stats)

# Combine fantasy points and impact scores
# Combine fantasy points and impact scores
combined_scores = {}
for player_name in fantasy_points.keys():
    impact_score = all_impact_scores.get(player_name, 0)
    if impact_score is None:
        impact_score = 0
    combined_score = fantasy_points[player_name] + impact_score
    combined_scores[player_name] = combined_score

# Rank players based on combined scores
ranked_players = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

# Print top 11 players based on combined scores
top_11_players = ranked_players[:11]
print("Top 11 Players Based on Combined Scores:")
for player_name, score in top_11_players:
    print(f"{player_name}: {score}")

