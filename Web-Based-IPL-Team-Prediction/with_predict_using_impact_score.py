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

# Preprocess the data (assuming preprocessing steps)
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
for i, team in enumerate(teams):
    against_team = against_teams[i]
    predictions = predict_runs_and_wickets(team, against_team, models_runs, models_wickets, data, encoder, kmeans_runs, kmeans_wickets)
    impact_scores = calculate_impact_score(predictions)
    print(f"Team {i+1} vs {against_team}:")
    for player_name in team:
        if predictions[player_name]['predicted_runs'] is not None and predictions[player_name]['predicted_wickets'] is not None:
            print(f"  Impact score for {player_name}: {impact_scores[player_name]:.2f}")
        else:
            print(f"  No data available for {player_name} against {against_team}")

# Define points system for fantasy points calculation (assuming already defined)

# Aggregate performance metrics for each player in one team against the other team

# Load data into DataFrames
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
    total_50s = player_team_data['50s'].sum()
    total_100s = player_team_data['100s'].sum()
    
    return {
        'matches_played': matches_played,
        'total_runs': total_runs,
        'total_wickets': total_wickets,
        'total_4s': total_4s,
        'total_6s': total_6s,
        'total_50s': total_50s,
        'total_100s': total_100s
    }

def performance_against_player(batsman, bowler, cricket_data):
    filtered_data = cricket_data[(cricket_data['batsman'] == batsman) & (cricket_data['bowler'] == bowler)]
    total_runs = filtered_data['batsman_runs'].sum()
    total_wickets = filtered_data[filtered_data['is_wicket'] == 1].shape[0]
    
    return {
        'total_runs': total_runs,
        'total_wickets': total_wickets
    }



def calculate_fantasy_points(player_performance, Batsman_points, Bowling_points, Fielding_points, Impact_points={'Impact': 0}):
    fantasy_points = (
        player_performance['total_runs'] * Batsman_points['Run'] +
        player_performance['total_4s'] * Batsman_points['bFour'] +
        player_performance['total_6s'] * Batsman_points['bSix'] +
        player_performance['total_50s'] * Batsman_points['Half_century'] +
        player_performance['total_100s'] * Batsman_points['Century'] +
        player_performance['total_wickets'] * Bowling_points['Wicket']
    )
    
    # Assuming some dummy values for fielding points
    total_catches = player_performance.get('catches', 0)
    total_stumpings = player_performance.get('stumpings', 0)
    total_runouts = player_performance.get('runouts', 0)

    fantasy_points += (
        total_catches * Fielding_points['Catch'] +
        total_stumpings * Fielding_points['Stumping'] +
        total_runouts * Fielding_points['RunOutD']  # Assuming direct runouts
    )
    
    # Adding impact score to fantasy points
    impact_score = Impact_points['Impact']
    fantasy_points += impact_score
    
    return fantasy_points


# Define the teams and their names (assuming already defined)
# Define points system
Batsman_points = {
    'Run': 1, 'bFour': 1, 'bSix': 2, '30Runs': 4,
    'Half_century': 8, 'Century': 16, 'Duck': -2, '170sr': 6,
    '150sr': 4, '130sr': 2, '70sr': -2, '60sr': -4, '50sr': -6
}

Bowling_points = {
    'Wicket': 25, 'LBW_Bowled': 8, '3W': 4, '4W': 8,
    '5W': 16, 'Maiden': 12, '5rpo': 6, '6rpo': 4, '7rpo': 2, '10rpo': -2,
    '11rpo': -4, '12rpo': -6
}

Fielding_points = {
    'Catch': 8, '3Cath': 4, 'Stumping': 12, 'RunOutD': 12,
    'RunOutInd': 6
}

# Aggregate performance metrics for each player in one team against the other team
def aggregate_performance_metrics(team, opponent_team, team_name, opponent_team_name, df, cricket_data):
    player_performances = []

    for player in team:
        team_performance = performance_against_team(player, opponent_team_name, df)
        
        # Performance against each player in the opponent team
        total_opponent_performance = {'total_runs': 0, 'total_wickets': 0}
        for opponent in opponent_team:
            player_vs_opponent = performance_against_player(player, opponent, cricket_data)
            total_opponent_performance['total_runs'] += player_vs_opponent['total_runs']
            total_opponent_performance['total_wickets'] += player_vs_opponent['total_wickets']

        # Aggregate performance
        aggregated_performance = {
            'total_runs': team_performance['total_runs'] + total_opponent_performance['total_runs'],
            'total_wickets': team_performance['total_wickets'] + total_opponent_performance['total_wickets'],
            'total_4s': team_performance['total_4s'],
            'total_6s': team_performance['total_6s'],
            'total_50s': team_performance['total_50s'],
            'total_100s': team_performance['total_100s']
        }
        
        fantasy_points = calculate_fantasy_points(aggregated_performance, Batsman_points, Bowling_points, Fielding_points)
        player_performances.append((fantasy_points, player))
        
    return player_performances

# Define the teams and their names
csk = ['MS Dhoni', 'Shaik Rasheed', 'Shivam Dube', 'RD Gaikwad', 'DL Chahar', 'RA Jadeja', 'AM Rahane', 'M Theekshana', 'TU Deshpande', 'Simarjeet Singh', 'MM Ali']
csk_name = 'Chennai Super Kings'
gt = ['Rashid Khan', 'Shubman Gill', 'Mohammed Shami', 'WP Saha', 'DA Miller', 'V Shankar', 'MS Wade', 'J Yadav', 'KS Williamson', 'R Sai Kishore', 'MM Sharma']
gt_name = 'Gujarat Titans'
# Aggregate performance metrics for CSK players against GT players
csk_performances = aggregate_performance_metrics(csk, gt, csk_name, gt_name, df, cricket_data)

# Aggregate performance metrics for GT players against CSK players
gt_performances = aggregate_performance_metrics(gt, csk, gt_name, csk_name, df, cricket_data)

# Combine and sort performances
combined_performances = csk_performances + gt_performances
combined_performances.sort(reverse=True, key=lambda x: x[0])

# Get the top 11 players
top_11_players = combined_performances[:11]

# Print top 11 players based on fantasy points
print("Top 11 players based on fantasy points:")
for points, player in top_11_players:
    print(f"Player name: {player} - Fantasy Points: {points:.2f}")

