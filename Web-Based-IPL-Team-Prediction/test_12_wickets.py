import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

# Load the transformed match data
transformed_file = 'transformed_match_data.csv'
data = pd.read_csv(transformed_file)

# Preprocess the data
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

# Select relevant features and target variable
features = data[['ball_delivered', 'run_given'] + list(against_team_encoded_df.columns)]
target = data['wicket']

# Perform cluster analysis
kmeans = KMeans(n_clusters=5, random_state=42)
data['cluster'] = kmeans.fit_predict(features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train a Random Forest Classifier model for each cluster
models = {}
for cluster_id in range(kmeans.n_clusters):
    cluster_data = data[data['cluster'] == cluster_id]
    cluster_features = cluster_data[['ball_delivered', 'run_given'] + list(against_team_encoded_df.columns)]
    cluster_target = cluster_data['wicket']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(cluster_features, cluster_target)
    models[cluster_id] = model

# Evaluate the models
accuracy_cluster = {}
for cluster_id, model in models.items():
    cluster_test_data = data[data['cluster'] == cluster_id]
    X_cluster_test = cluster_test_data[['ball_delivered', 'run_given'] + list(against_team_encoded_df.columns)]
    y_cluster_test = cluster_test_data['wicket']
    
    y_cluster_pred = model.predict(X_cluster_test)
    accuracy_cluster[cluster_id] = accuracy_score(y_cluster_test, y_cluster_pred)

# Function to predict wickets against a specific team using clustering
def predict_wickets(player_name, against_team, model, data, encoder, kmeans):
    player_data = data[data['player'] == player_name]
    if not player_data.empty:
        # One-hot encode the against_team column for the new input
        against_team_encoded = encoder.transform([[against_team]])
        against_team_encoded_df = pd.DataFrame(against_team_encoded.toarray(), columns=encoder.get_feature_names_out(['against_team']))
        
        # Use the average ball delivered and run given by the player
        avg_ball_delivered = player_data['ball_delivered'].mean()
        avg_run_given = player_data['run_given'].mean()
        
        # Create a dataframe with the features for prediction
        input_data = pd.DataFrame([[avg_ball_delivered, avg_run_given] + list(against_team_encoded_df.iloc[0])], columns=features.columns)
        
        # Predict the cluster for the input data
        cluster_id = kmeans.predict(input_data)[0]
        
        # Use the corresponding model for prediction
        predicted_wickets = model[cluster_id].predict(input_data)
        return predicted_wickets[0]
    else:
        return None

# Input player name and team
player_name = input("Enter player name: ")
against_team = input("Enter against team: ")

# Predict wickets against the specified team using clustering
predicted_wickets = predict_wickets(player_name, against_team, models, data, encoder, kmeans)

if predicted_wickets is not None:
    print(f"Predicted wickets taken by {player_name} against {against_team}: {predicted_wickets}")
else:
    print(f"No data available for {player_name} against {against_team}")
