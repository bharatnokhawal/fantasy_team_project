import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Read CSV file
df = pd.read_csv('player_performance_combined4.csv')


# Define Fantasy Points System
Batsman_points = {'Run': 1, 'bFour': 1, 'bSix': 2, '30Runs': 4, 'Half_century': 8, 'Century': 16, 'Duck': -2, '170sr': 6, '150sr': 4, '130sr': 2, '70sr': -2, '60sr': -4, '50sr': -6}
Bowling_points = {'Wicket': 25, 'LBW_Bowled': 8, '3W': 4, '4W': 8, '5W': 16, 'Maiden': 12, '5rpo': 6, '6rpo': 4, '7rpo': 2, '10rpo': -2, '11rpo': -4, '12rpo': -6}
Fielding_points = {'Catch': 8, '3Catch': 4, 'Stumping': 12, 'RunOutD': 12, 'RunOutInd': 6}

# Example teams
team1_players = ['Abdul Samad','Abhishek Sharma','Shahbaz Ahmed','H Klaasen','TM Head','B Kumar', 'T Natarajan','Nithish Kumar Reddy','vijayakanth viyaskanth','PJ Cummins','Sanvir Singh']

team2_players =  ['HV Patel','Atharva Taide', 'Arshdeep Singh','Ashutosh Sharma','Shashank Singh','Jitesh Sharma','Prabhsimran Singh','R Dhawan','RD Chahar','Harpreet Brar', 'RR Rossouw']

# Combine the players from both teams
selected_players = team1_players + team2_players

# Filter the dataset for the selected players
df_selected = df[df['player'].isin(selected_players)]

# Points calculation functions (same as above)
def calculate_batting_points(row):
    points = row['runs_scored'] * Batsman_points['Run'] + row['fours'] * Batsman_points['bFour'] + row['sixes'] * Batsman_points['bSix']
    if row['runs_scored'] >= 30:
        points += Batsman_points['30Runs']
    if row['runs_scored'] >= 50:
        points += Batsman_points['Half_century']
    if row['runs_scored'] >= 100:
        points += Batsman_points['Century']
    if row['runs_scored'] == 0:
        points += Batsman_points['Duck']
    if 'balls_faced' in row:
        sr = (row['runs_scored'] / row['balls_faced']) * 100 if row['balls_faced'] > 0 else 0
        if sr >= 170:
            points += Batsman_points['170sr']
        elif sr >= 150:
            points += Batsman_points['150sr']
        elif sr >= 130:
            points += Batsman_points['130sr']
        elif sr <= 70:
            points += Batsman_points['70sr']
        elif sr <= 60:
            points += Batsman_points['60sr']
        elif sr <= 50:
            points += Batsman_points['50sr']
    return points

def calculate_bowling_points(row):
    points = row['wickets_taken'] * Bowling_points['Wicket']
    if 'lbw_bowled' in row:
        points += row['lbw_bowled'] * Bowling_points['LBW_Bowled']
    if 'runs_conceded' in row and 'overs_bowled' in row:
        rpo = (row['runs_conceded'] / row['overs_bowled']) if row['overs_bowled'] > 0 else 0
        if rpo <= 5:
            points += Bowling_points['5rpo']
        elif rpo <= 6:
            points += Bowling_points['6rpo']
        elif rpo <= 7:
            points += Bowling_points['7rpo']
        elif rpo >= 10:
            points -= Bowling_points['10rpo']
        elif rpo >= 11:
            points -= Bowling_points['11rpo']
        elif rpo >= 12:
            points -= Bowling_points['12rpo']
    return points

def calculate_fielding_points(row):
    points = row['catches'] * Fielding_points['Catch'] + row['runouts'] * Fielding_points['RunOutD']
    return points

# Calculate total points for each player
df_selected['batting_points'] = df_selected.apply(calculate_batting_points, axis=1)
df_selected['bowling_points'] = df_selected.apply(calculate_bowling_points, axis=1)
df_selected['fielding_points'] = df_selected.apply(calculate_fielding_points, axis=1)
df_selected['total_points'] = df_selected['batting_points'] + df_selected['bowling_points'] + df_selected['fielding_points']

# Aggregate points for each player
player_stats = df_selected.groupby('player').agg({
    'total_points': 'sum',
    'runs_scored': 'sum',
    'fours': 'sum',
    'sixes': 'sum',
    'wickets_taken': 'sum',
    'runouts': 'sum',
    'catches': 'sum'
}).reset_index()

# Define features and target
X = player_stats[['runs_scored', 'fours', 'sixes', 'wickets_taken', 'runouts', 'catches']]
y = player_stats['total_points']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')

# Predict the points for the selected 22 players
player_stats['predicted_points'] = model.predict(X)
best_11_players = player_stats.sort_values(by='predicted_points', ascending=False).head(11)
print(best_11_players[['player', 'predicted_points']])
