import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, accuracy_score

# Load the transformed match data
transformed_file = 'transformed_match_data.csv'
data = pd.read_csv(transformed_file)

# Preprocess the data
data['ball_faced'] = data['ball_faced'].astype(int)
data['run_scored'] = data['run_scored'].astype(int)
data['ball_delivered'] = data['ball_delivered'].astype(int)
data['run_given'] = data['run_given'].astype(int)
data['wicket'] = data['wicket'].astype(int)

# Select relevant features for runs and wickets
features_runs = data[['ball_faced']]
target_runs = data['run_scored']
features_wickets = data[['ball_delivered', 'run_given']]
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
    cluster_features = cluster_data[['ball_faced']]
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
    cluster_features = cluster_data[['ball_delivered', 'run_given']]
    cluster_target = cluster_data['wicket']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(cluster_features, cluster_target)
    models_wickets[cluster_id] = model

# Evaluate the models (runs)
mae_cluster_runs = {}
for cluster_id, model in models_runs.items():
    cluster_test_data = data[data['cluster_runs'] == cluster_id]
    X_cluster_test = cluster_test_data[['ball_faced']]
    y_cluster_test = cluster_test_data['run_scored']
    
    y_cluster_pred = model.predict(X_cluster_test)
    mae_cluster_runs[cluster_id] = mean_absolute_error(y_cluster_test, y_cluster_pred)

# Evaluate the models (wickets)
accuracy_cluster_wickets = {}
for cluster_id, model in models_wickets.items():
    cluster_test_data = data[data['cluster_wickets'] == cluster_id]
    X_cluster_test = cluster_test_data[['ball_delivered', 'run_given']]
    y_cluster_test = cluster_test_data['wicket']
    
    y_cluster_pred = model.predict(X_cluster_test)
    accuracy_cluster_wickets[cluster_id] = accuracy_score(y_cluster_test, y_cluster_pred)

# Function to predict runs and wickets based on overall performance using clustering
def predict_runs_and_wickets(player_names, models_runs, models_wickets, data, kmeans_runs, kmeans_wickets):
    predictions = {}
    for player_name in player_names:
        player_data = data[data['player'] == player_name]
        if not player_data.empty:
            # Calculate the average balls faced, runs scored, ball delivered, and runs given by the player
            avg_balls_faced = player_data['ball_faced'].mean()
            avg_runs_scored = player_data['run_scored'].mean()
            avg_ball_delivered = player_data['ball_delivered'].mean()
            avg_run_given = player_data['run_given'].mean()
            
            # Create dataframes with the features for prediction
            input_data_runs = pd.DataFrame([[avg_balls_faced]], columns=features_runs.columns)
            input_data_wickets = pd.DataFrame([[avg_ball_delivered, avg_run_given]], columns=features_wickets.columns)
            
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

# Input player names
player_names = ['V Kohli', 'GJ Maxwell', 'Mohammed Siraj', 'C Green', 'RM Patidar', 'MK Lomror', 'Yash Dayal', 'KD Karthik', 'KV Sharma', 'F du Plessis', 'LH Ferguson']

# Predict runs and wickets based on overall performance using clustering
predictions = predict_runs_and_wickets(player_names, models_runs, models_wickets, data, kmeans_runs, kmeans_wickets)

# Display the predictions
for player_name, prediction in predictions.items():
    if prediction['predicted_runs'] is not None and prediction['predicted_wickets'] is not None:
        print(f"Predicted runs scored by {player_name}: {prediction['predicted_runs']:.2f}")
        print(f"Predicted wickets taken by {player_name}: {prediction['predicted_wickets']:.2f}")
    else:
        print(f"No data available for {player_name}")
