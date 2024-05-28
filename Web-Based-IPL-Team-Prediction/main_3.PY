import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split
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

# Function to predict runs and wickets against specific teams using clustering
def predict_runs_and_wickets(teams, against_teams, models_runs, models_wickets, data, encoder, kmeans_runs, kmeans_wickets):
    ranked_players_runs = {}
    ranked_players_wickets = {}

    for i, team in enumerate(teams):
        against_team = against_teams[i]

        predictions_runs = {}
        predictions_wickets = {}

        for player_name in team:
            player_data = data[data['player'] == player_name]
            if not player_data.empty:
                against_team_encoded = encoder.transform([[against_team]])
                against_team_encoded_df = pd.DataFrame(against_team_encoded.toarray(), columns=encoder.get_feature_names_out(['against_team']))

                avg_balls_faced = player_data['ball_faced'].mean()
                avg_ball_delivered = player_data['ball_delivered'].mean()
                avg_run_given = player_data['run_given'].mean()

                input_data_runs = pd.DataFrame([[avg_balls_faced] + list(against_team_encoded_df.iloc[0])], columns=features_runs.columns)
                input_data_wickets = pd.DataFrame([[avg_ball_delivered, avg_run_given] + list(against_team_encoded_df.iloc[0])], columns=features_wickets.columns)

                cluster_id_runs = kmeans_runs.predict(input_data_runs)[0]
                cluster_id_wickets = kmeans_wickets.predict(input_data_wickets)[0]

                predicted_runs = models_runs[cluster_id_runs].predict(input_data_runs)[0]
                predicted_wickets = models_wickets[cluster_id_wickets].predict(input_data_wickets)[0]

                predictions_runs[player_name] = predicted_runs
                predictions_wickets[player_name] = predicted_wickets
            else:
                predictions_runs[player_name] = None
                predictions_wickets[player_name] = None

        ranked_players_runs[f'Team {i + 1}'] = sorted(predictions_runs.items(), key=lambda x: x[1], reverse=True)
        ranked_players_wickets[f'Team {i + 1}'] = sorted(predictions_wickets.items(), key=lambda x: x[1], reverse=True)

    return ranked_players_runs, ranked_players_wickets

# Input team names and against teams
teams = [['Abdul Samad','Abhishek Sharma','RA Tripathi','H Klaasen','TM Head','B Kumar', 'T Natarajan','Sanvir Singh','Shahbaz Ahmed','PJ Cummins','Nithish Kumar Reddy'],
         ['SV Samson', 'Avesh Khan', 'YBK Jaiswal', 'TA Boult', 'R Parag', 'SO Hetmyer', 'R Ashwin', 'Dhruv Jurel', 'YS Chahal', 'R Powell', 'Sandeep Sharma']]
against_teams = ['Rajasthan Royals', 'Sunrisers Hyderabad']  # Specify against teams for each team

# Predict runs and wickets against the specified teams using clustering and rank players
ranked_runs, ranked_wickets = predict_runs_and_wickets(teams, against_teams, models_runs, models_wickets, data, encoder, kmeans_runs, kmeans_wickets)

# Display the ranked players based on runs
print("Ranking based on Runs:")
for team, ranked_players in ranked_runs.items():
    print(f"\nTeam: {team}")
    for rank, (player_name, predicted_runs) in enumerate(ranked_players, 1):
        if predicted_runs is not None:
            print(f"{rank}. {player_name}: {predicted_runs:.2f} runs")
        else:
            print(f"{rank}. {player_name}: No data available")

# Display the ranked players based on wickets
print("\nRanking based on Wickets:")
for team, ranked_players in ranked_wickets.items():
    print(f"\nTeam: {team}")
    for rank, (player_name, predicted_wickets) in enumerate(ranked_players, 1):
        if predicted_wickets is not None:
            print(f"{rank}. {player_name}: {predicted_wickets:.2f} wickets")
        else:
            print(f"{rank}. {player_name}: No data available")
