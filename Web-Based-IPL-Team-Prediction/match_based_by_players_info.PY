import pandas as pd

# Assuming the data is already loaded into a DataFrame called 'df'
df = pd.read_csv('transformed_match_data.csv')

# Filter the data for the player and the specific team
player_name = 'KS Williamson'
team_name = 'Chennai Super Kings'
player_team_data = df[(df['player'] == player_name) & (df['against_team'] == team_name)]

# Count the number of unique matches played against the specific team
matches_played = player_team_data['match_id'].nunique()

# Calculate total runs, wickets, boundaries, 50s, 100s
total_runs = player_team_data['run_scored'].sum()
total_wickets = player_team_data['wicket'].sum()
total_4s = player_team_data['4s'].sum()
total_6s = player_team_data['6s'].sum()
total_50s = player_team_data['50s'].sum()
total_100s = player_team_data['100s'].sum()

# Calculate averages per match
average_runs_per_match = total_runs / matches_played if matches_played else 0
average_wickets_per_match = total_wickets / matches_played if matches_played else 0

# Calculate strike rate
total_balls_faced = player_team_data['ball_faced'].sum()
strike_rate = (total_runs / total_balls_faced) * 100 if total_balls_faced else 0

# Calculate bowling statistics
total_balls_delivered = player_team_data['ball_delivered'].sum()
total_runs_given = player_team_data['run_given'].sum()
bowling_average = total_runs_given / total_wickets if total_wickets else 0

# Fantasy Points calculation
Batsman_points = {'Run': 1, 'bFour': 1, 'bSix': 2, '30Runs': 4,
                  'Half_century': 8, 'Century': 16, 'Duck': -2, '170sr': 6,
                  '150sr': 4, '130sr': 2, '70sr': -2, '60sr': -4, '50sr': -6}

Bowling_points = {'Wicket': 25, 'LBW_Bowled': 8, '3W': 4, '4W': 8,
                  '5W': 16, 'Maiden': 12, '5rpo': 6, '6rpo': 4, '7rpo': 2, '10rpo': -2,
                  '11rpo': -4, '12rpo': -6}

Fielding_points = {'Catch': 8, '3Cath': 4, 'Stumping': 12, 'RunOutD': 12,
                   'RunOutInd': 6}

# Calculate Fantasy Points
# Batting points
fantasy_points = (
    total_runs * Batsman_points['Run'] +
    total_4s * Batsman_points['bFour'] +
    total_6s * Batsman_points['bSix'] +
    total_50s * Batsman_points['Half_century'] +
    total_100s * Batsman_points['Century']
)

# Bowling points
fantasy_points += (
    total_wickets * Bowling_points['Wicket']
)

# Fielding points (assuming some dummy values for catches, stumpings, runouts)
total_catches = player_team_data['catches'].sum() if 'catches' in player_team_data else 0
total_stumpings = player_team_data['stumpings'].sum() if 'stumpings' in player_team_data else 0
total_runouts = player_team_data['runouts'].sum() if 'runouts' in player_team_data else 0

fantasy_points += (
    total_catches * Fielding_points['Catch'] +
    total_stumpings * Fielding_points['Stumping'] +
    total_runouts * Fielding_points['RunOutD']  # Assuming direct runouts
)

print(f"{player_name} has played {matches_played} matches against {team_name} or run score {total_runs}.")
print(f"Total Fantasy Points: {fantasy_points:.2f}")
print(f'4s:{total_4s} , 6s: {total_6s} , 50s: {total_50s} , 100s: {total_100s}')




import pandas as pd

# Assuming the provided data is stored in a DataFrame named 'cricket_data'
# You can load the data into a DataFrame like this:
cricket_data = pd.read_csv('IPl Ball-by-Ball 2008-2023.csv')

# Filter data for a specific batsman and bowler
batsman = 'KS Williamson'
bowler = 'RA Jadeja'
team ='Chennai Super Kings'
filtered_data = cricket_data[(cricket_data['batsman'] == batsman) & (cricket_data['bowler'] == bowler)]
filtered_data_for = cricket_data[(cricket_data['batsman'] == batsman) & (cricket_data['bowling_team'] == team)]
# Calculate total runs and wickets
total_runs = filtered_data['batsman_runs'].sum()
total_wickets = filtered_data[filtered_data['is_wicket'] == 1].shape[0]

total_runs_for = filtered_data_for['batsman_runs'].sum()
total_wickets_for = filtered_data_for[filtered_data_for['is_wicket'] == 1].shape[0]

print(f"Total runs scored by {batsman} against {bowler}: {total_runs}")
print(f"Total times {batsman} got out against {bowler}: {total_wickets}")

print(total_runs_for)
print(total_wickets_for)



