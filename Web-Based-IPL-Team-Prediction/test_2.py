import pandas as pd
import numpy as np

# Load datasets
byb = pd.read_csv('IPl Ball-by-Ball 2008-2023.csv')
match = pd.read_csv('IPL Mathces 2008-2023.csv')

# Define fantasy points
Batsman_points = {'Run': 1, 'bFour': 1, 'bSix': 2, '30Runs': 4, 'Half_century': 8, 'Century': 16, 'Duck': -2, '170sr': 6, '150sr': 4, '130sr': 2, '70sr': -2, '60sr': -4, '50sr': -6}
Bowling_points = {'Wicket': 25, 'LBW_Bowled': 8, '3W': 4, '4W': 8, '5W': 16, 'Maiden': 12, '5rpo': 6, '6rpo': 4, '7rpo': 2, '10rpo': -2, '11rpo': -4, '12rpo': -6}
Fielding_points = {'Catch': 8, '3Catch': 4, 'Stumping': 12, 'RunOutD': 12, 'RunOutInd': 6}

# Define teams and player points
ind =  ['Abdul Samad','Abhishek Sharma','Shahbaz Ahmed','H Klaasen','TM Head','B Kumar', 'T Natarajan','Nithish Kumar Reddy','vijayakanth viyaskanth','PJ Cummins','Sanvir Singh']
aus =   ['HV Patel','Atharva Taide', 'Arshdeep Singh','Ashutosh Sharma','Shashank Singh','Jitesh Sharma','Prabhsimran Singh','R Dhawan','RD Chahar','Harpreet Brar', 'RR Rossouw']

ind_fp = {player: 0 for player in ind}
aus_fp = {player: 0 for player in aus}

def calculate_individual_points(player, team, opponent_team, player_points):
    # Get unique match IDs
    unq_ids = byb[byb["batsman"] == player]['id'].unique()
    matches_played = len(unq_ids)
    
    # Batting performance
    bbr = []
    for match_id in unq_ids:
        runs = byb[(byb["batsman"] == player) & (byb['id'] == match_id)]['batsman_runs'].sum()
        bbr.append(runs)

    r30, r50, r100 = sum(run >= 30 for run in bbr), sum(run >= 50 for run in bbr), sum(run >= 100 for run in bbr)
    catches = byb[(byb['fielder'] == player) & (byb['dismissal_kind'] == 'caught')].shape[0]
    run_outs = byb[(byb['fielder'] == player) & (byb['dismissal_kind'] == 'run out')].shape[0]
    extra_points = r30 * Batsman_points['30Runs'] + r50 * Batsman_points['Half_century'] + r100 * Batsman_points['Century'] + catches * Fielding_points['Catch'] + run_outs * Fielding_points['RunOutInd']

    # Bowling performance
    wickets_taken = byb[(byb["bowler"] == player) & (byb['id'].isin(unq_ids))]['is_wicket'].sum()
    lbws = byb[(byb['bowler'] == player) & (byb['dismissal_kind'] == 'lbw')].shape[0]
    bowled = byb[(byb['bowler'] == player) & (byb['dismissal_kind'] == 'bowled')].shape[0]
    
    # Calculate additional points for wickets
    w3 = Bowling_points['3W'] if wickets_taken >= 3 else 0
    w4 = Bowling_points['4W'] if wickets_taken >= 4 else 0
    w5 = Bowling_points['5W'] if wickets_taken >= 5 else 0
    wexp = w3 + w4 + w5 + (lbws + bowled) * Bowling_points['LBW_Bowled']

    # Calculate performance against the opponent team as a whole
    team_matches = byb[(byb["batsman"] == player) & (byb["bowling_team"].isin(opponent_team))]['id'].unique()
    runs_against_team = byb[(byb["batsman"] == player) & (byb["bowling_team"].isin(opponent_team))]['batsman_runs'].sum()
    wickets_against_team = byb[(byb["bowler"] == player) & (byb["batting_team"].isin(opponent_team))]['is_wicket'].sum()
    
    # Calculate performance against individual players from the opponent team
    ffp = []
    for opponent in team:
        bat_vs_bowl = byb[(byb["batsman"] == player) & (byb["bowler"] == opponent)]
        runs_scored = bat_vs_bowl['batsman_runs'].sum()
        fours = bat_vs_bowl[bat_vs_bowl['batsman_runs'] == 4].shape[0]
        sixes = bat_vs_bowl[bat_vs_bowl['batsman_runs'] == 6].shape[0]
        wickets = bat_vs_bowl['is_wicket'].sum()

        penalty = 0
        if len(bat_vs_bowl) <= 6 * 10 and wickets >= 5:
            penalty = -16
        elif len(bat_vs_bowl) <= 6 * 8 and wickets >= 4:
            penalty = -8
        elif len(bat_vs_bowl) <= 6 * 6 and wickets >= 3:
            penalty = -4

        strike_rate = (runs_scored / len(bat_vs_bowl) * 100) if len(bat_vs_bowl) > 0 else 0

        fantasy_points1 = runs_scored + fours * Batsman_points['bFour'] + sixes * Batsman_points['bSix'] - wickets * Bowling_points['Wicket'] + penalty
        ffp.append(fantasy_points1)

    sum_ffp = sum(ffp)
    recent_performance_points = player_points[player] / 3
    weight1, weight2 = 0.5, 0.5
    final_fantasy_point = (sum_ffp + extra_points + wexp + runs_against_team - wickets_against_team * Bowling_points['Wicket']) * weight1 + recent_performance_points * weight2
    final_fantasy_point = round(final_fantasy_point, 2)
    
    return final_fantasy_point

def get_fantasy_points(team1, team2, team1_fp, opponent_team):
    fantasy_team_players = []

    for player in team1:
        final_fantasy_point = calculate_individual_points(player, team2, opponent_team, team1_fp)
        fantasy_team_players.append((final_fantasy_point, player))

    fantasy_team_players.sort(reverse=True, key=lambda x: x[0])
    return fantasy_team_players

# Get top players
team_aus = get_fantasy_points(aus, ind, aus_fp, ['India'])
team_ind = get_fantasy_points(ind, aus, ind_fp, ['Australia'])

# Combine and rank
combined_team = team_aus + team_ind
combined_team.sort(reverse=True, key=lambda x: x[0])
final_team = combined_team[:11]

# Convert to DataFrame and display
result = pd.DataFrame(final_team, columns=['Fantasy Points', 'Player'])
print('\nFinal Predicted Team:\n', result)
