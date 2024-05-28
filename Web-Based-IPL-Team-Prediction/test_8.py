import pandas as pd

# Load CSV data into a pandas DataFrame
df = pd.read_csv('IPl Ball-by-Ball 2008-2023.csv')

# Create a dictionary to store player performance against teams
player_performance_team = {}

# Create a dictionary to store player performance against bowlers
player_performance_bowler = {}

# Process each row in the DataFrame
for index, row in df.iterrows():
    batsman = row['batsman']
    bowler = row['bowler']
    bowling_team = row['bowling_team']
    runs = row['batsman_runs']
    fours = 1 if runs == 4 else 0
    sixes = 1 if runs == 6 else 0
    is_wicket = row['is_wicket']
    dismissal_kind = row['dismissal_kind']
    
    # Update player performance against teams
    if batsman not in player_performance_team:
        player_performance_team[batsman] = {
            'teams': {},
            'total_runs': 0,
            'fours': 0,
            'sixes': 0,
            'balls_faced': 0,  # Added for calculating strike rate
            'wickets': 0,
            'catches': 0,
            'runouts': 0,
            'stumped': 0
        }
    if bowling_team not in player_performance_team[batsman]['teams']:
        player_performance_team[batsman]['teams'][bowling_team] = {
            'runs': 0,
            'balls_faced': 0,  # Added for calculating strike rate
            '50_runs': 0,
            '30_runs': 0,
            '100_runs': 0,
            '3w': 0,
            '5w': 0,
            'economy': 0
        }
    
    player_performance_team[batsman]['teams'][bowling_team]['runs'] += runs
    player_performance_team[batsman]['teams'][bowling_team]['balls_faced'] += 1  # Increment balls faced
    player_performance_team[batsman]['total_runs'] += runs
    player_performance_team[batsman]['fours'] += fours
    player_performance_team[batsman]['sixes'] += sixes
    player_performance_team[batsman]['balls_faced'] += 1  # Increment balls faced for the player

    # Update player performance against bowlers
    if batsman not in player_performance_bowler:
        player_performance_bowler[batsman] = {}
    if bowler not in player_performance_bowler[batsman]:
        player_performance_bowler[batsman][bowler] = {
            'runs': 0,
            'wickets': 0,
            'catches': 0,
            'runouts': 0,
            'stumped': 0
        }
    
    player_performance_bowler[batsman][bowler]['runs'] += runs

    # Update wickets, catches, runouts, stumped
    if is_wicket:
        player_performance_team[batsman]['wickets'] += 1
        if dismissal_kind == 'caught':
            player_performance_team[batsman]['catches'] += 1
        elif dismissal_kind == 'run out':
            player_performance_team[batsman]['runouts'] += 1
        elif dismissal_kind == 'stumped':
            player_performance_team[batsman]['stumped'] += 1

        if bowler in player_performance_bowler[batsman]:
            player_performance_bowler[batsman][bowler]['wickets'] += 1
            if dismissal_kind == 'caught':
                player_performance_bowler[batsman][bowler]['catches'] += 1
            elif dismissal_kind == 'run out':
                player_performance_bowler[batsman][bowler]['runouts'] += 1
            elif dismissal_kind == 'stumped':
                player_performance_bowler[batsman][bowler]['stumped'] += 1

# Prepare data for player performance against teams
data_team = []
for player, stats in player_performance_team.items():
    for team, team_stats in stats['teams'].items():
        balls_faced = team_stats['balls_faced']
        strike_rate = 100 * team_stats['runs'] / balls_faced if balls_faced > 0 else 0
        economy = team_stats['runs'] / (team_stats['balls_faced'] / 6) if team_stats['balls_faced'] > 0 else 0
        data_team.append({
            'player': player,
            'team': team,
            'runs_against_team': team_stats['runs'],
            'fours': stats['fours'],
            'sixes': stats['sixes'],
            'strike_rate': strike_rate,
            '3w': 1 if stats['wickets'] >= 3 else 0,
            '5w': 1 if stats['wickets'] >= 5 else 0,
            '50_runs': 1 if team_stats['runs'] >= 50 else 0,
            '30_runs': 1 if team_stats['runs'] >= 30 else 0,
            '100_runs': 1 if team_stats['runs'] >= 100 else 0,
            'wickets': stats['wickets'],
            'catches': stats['catches'],
            'runouts': stats['runouts'],
            'stumped': stats['stumped']
        })

# Prepare data for player performance against bowlers
data_bowler = []
for player, bowler_stats in player_performance_bowler.items():
    for bowler, stats in bowler_stats.items():
        data_bowler.append({
            'player': player,
            'bowler': bowler,
            'runs_against_bowler': stats['runs'],
            'wickets': stats['wickets'],
            'catches': stats['catches'],
            'runouts': stats['runouts'],
            'stumped': stats['stumped']
        })

# Create DataFrames for player performance against teams and bowlers
df_team = pd.DataFrame(data_team)
df_bowler = pd.DataFrame(data_bowler)

# Save DataFrames to CSV files
df_team.to_csv('player_performance_team.csv', index=False)
df_bowler.to_csv('player_performance_bowler.csv', index=False)

print(f"Player performance against teams saved to 'player_performance_team.csv'")
print(f"Player performance against bowlers saved to 'player_performance_bowler.csv'")
