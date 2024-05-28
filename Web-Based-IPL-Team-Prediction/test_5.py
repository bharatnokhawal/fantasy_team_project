import pandas as pd

# Sample data
# data = {
#     'id': [335982, 335982, 335982, 335982],
#     'inning': [1, 1, 1, 1],
#     'over': [6, 6, 7, 7],
#     'ball': [5, 6, 1, 2],
#     'batsman': ['SC Ganguly', 'BB McCullum', 'BB McCullum', 'BB McCullum'],
#     'non_striker': ['BB McCullum', 'SC Ganguly', 'SC Ganguly', 'SC Ganguly'],
#     'bowler': ['P Kumar', 'P Kumar', 'P Kumar', 'P Kumar'],
#     'batsman_runs': [0, 0, 0, 0],
#     'extra_runs': [1, 0, 1, 0],
#     'total_runs': [1, 1, 0, 1],
#     'non_boundary': [0, 0, 0, 0],
#     'is_wicket': [0, 0, 0, 0],
#     'dismissal_kind': [None, None, None, None],
#     'player_dismissed': [None, None, None, None],
#     'fielder': [None, None, None, None],
#     'extras_type': [None, None, None, None],
#     'batting_team': ['Kolkata Knight Riders', 'Kolkata Knight Riders', 'Kolkata Knight Riders', 'Kolkata Knight Riders'],
#     'bowling_team': ['Royal Challengers Bangalore', 'Royal Challengers Bangalore', 'Royal Challengers Bangalore', 'Royal Challengers Bangalore']
# }

# Convert data to DataFrame
df =  pd.read_csv('IPl Ball-by-Ball 2008-2023.csv')

# Initialize player points
player_points = {}

# Batsman, Bowling, Fielding points system
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
    'Catch': 8, '3Catch': 4, 'Stumping': 12, 'RunOutD': 12,
    'RunOutInd': 6
}

# Function to calculate fantasy points
def calculate_fantasy_points(df, team1, team2, team1_players, team2_players):
    # Filter for matches between the two teams
    df_filtered = df[((df['batting_team'] == team1) & (df['bowling_team'] == team2)) | 
                     ((df['batting_team'] == team2) & (df['bowling_team'] == team1))]

    # Track stats
    batting_stats = {}
    bowling_stats = {}
    fielding_stats = {}

    # Process each delivery
    for index, row in df_filtered.iterrows():
        batsman = row['batsman']
        bowler = row['bowler']
        batsman_runs = row['batsman_runs']
        is_wicket = row['is_wicket']
        dismissal_kind = row['dismissal_kind']
        player_dismissed = row['player_dismissed']
        fielder = row['fielder']
        extra_runs = row['extra_runs']
        
        # Update batsman stats
        if batsman in team1_players or batsman in team2_players:
            if batsman not in batting_stats:
                batting_stats[batsman] = {'runs': 0, 'balls': 0, 'fours': 0, 'sixes': 0}
            batting_stats[batsman]['runs'] += batsman_runs
            batting_stats[batsman]['balls'] += 1
            if batsman_runs == 4:
                batting_stats[batsman]['fours'] += 1
            if batsman_runs == 6:
                batting_stats[batsman]['sixes'] += 1

        # Update bowler stats
        if bowler in team1_players or bowler in team2_players:
            if bowler not in bowling_stats:
                bowling_stats[bowler] = {'wickets': 0, 'balls': 0, 'runs': 0, 'maidens': 0}
            bowling_stats[bowler]['balls'] += 1
            if extra_runs == 0:
                bowling_stats[bowler]['runs'] += row['total_runs']
        
        # Update fielding stats
        if fielder in team1_players or fielder in team2_players:
            if is_wicket and player_dismissed and dismissal_kind != 'run out':
                if fielder not in fielding_stats:
                    fielding_stats[fielder] = {'catches': 0, 'stumpings': 0, 'runouts_direct': 0, 'runouts_indirect': 0}
                if dismissal_kind == 'caught':
                    fielding_stats[fielder]['catches'] += 1
                if dismissal_kind == 'stumped':
                    fielding_stats[fielder]['stumpings'] += 1
        
        # Check for wickets taken by the bowler
        if bowler in team1_players or bowler in team2_players:
            if is_wicket and dismissal_kind in ['bowled', 'lbw', 'caught', 'stumped']:
                bowling_stats[bowler]['wickets'] += 1

    # Calculate fantasy points for batsmen
    for batsman, stats in batting_stats.items():
        runs = stats['runs']
        balls = stats['balls']
        fours = stats['fours']
        sixes = stats['sixes']
        
        points = runs * Batsman_points['Run'] + fours * Batsman_points['bFour'] + sixes * Batsman_points['bSix']
        
        if runs >= 30:
            points += Batsman_points['30Runs']
        if runs >= 50:
            points += Batsman_points['Half_century']
        if runs >= 100:
            points += Batsman_points['Century']
        if runs == 0 and balls > 0:
            points += Batsman_points['Duck']
        
        strike_rate = (runs / balls) * 100 if balls > 0 else 0
        if strike_rate >= 170:
            points += Batsman_points['170sr']
        elif strike_rate >= 150:
            points += Batsman_points['150sr']
        elif strike_rate >= 130:
            points += Batsman_points['130sr']
        elif strike_rate <= 70:
            points += Batsman_points['70sr']
        elif strike_rate <= 60:
            points += Batsman_points['60sr']
        elif strike_rate <= 50:
            points += Batsman_points['50sr']
        
        player_points[batsman] = player_points.get(batsman, 0) + points

    # Calculate fantasy points for bowlers
    for bowler, stats in bowling_stats.items():
        wickets = stats['wickets']
        balls = stats['balls']
        runs = stats['runs']
        
        points = wickets * Bowling_points['Wicket']
        if wickets >= 3:
            points += Bowling_points['3W']
        if wickets >= 4:
            points += Bowling_points['4W']
        if wickets >= 5:
            points += Bowling_points['5W']
        
        overs = balls // 6
        if overs > 0:
            economy_rate = runs / overs
            if economy_rate <= 5:
                points += Bowling_points['5rpo']
            elif economy_rate <= 6:
                points += Bowling_points['6rpo']
            elif economy_rate <= 7:
                points += Bowling_points['7rpo']
            elif economy_rate >= 10:
                points += Bowling_points['10rpo']
            elif economy_rate >= 11:
                points += Bowling_points['11rpo']
            elif economy_rate >= 12:
                points += Bowling_points['12rpo']
        
        player_points[bowler] = player_points.get(bowler, 0) + points

    # Calculate fantasy points for fielders
    for fielder, stats in fielding_stats.items():
        catches = stats['catches']
        stumpings = stats['stumpings']
        runouts_direct = stats['runouts_direct']
        runouts_indirect = stats['runouts_indirect']
        
        points = catches * Fielding_points['Catch'] + stumpings * Fielding_points['Stumping'] + runouts_direct * Fielding_points['RunOutD'] + runouts_indirect * Fielding_points['RunOutInd']
        
        if catches >= 3:
            points += Fielding_points['3Catch']
        
        player_points[fielder] = player_points.get(fielder, 0) + points

    return player_points

# Teams and players to compare
team1 = 'Sunrisers Hyderabad'
team2 = 'Kings XI Punjab'

team1_players =  ['Abdul Samad','Abhishek Sharma','Shahbaz Ahmed','H Klaasen','TM Head','B Kumar', 'T Natarajan','Nithish Kumar Reddy','vijayakanth viyaskanth','PJ Cummins','Sanvir Singh']
team2_players =   ['HV Patel','Atharva Taide', 'Arshdeep Singh','Ashutosh Sharma','Shashank Singh','Jitesh Sharma','Prabhsimran Singh','R Dhawan','RD Chahar','Harpreet Brar', 'RR Rossouw']
# Calculate and rank players based on fantasy points
points = calculate_fantasy_points(df, team1, team2, team1_players, team2_players)
sorted_points = sorted(points.items(), key=lambda x: x[1], reverse=True)

# Output the ranked players
for player, points in sorted_points:
    print(f"{player}: {points} points")
