import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Reading files
byb = pd.read_csv('IPl Ball-by-Ball 2008-2023.csv')
match = pd.read_csv('IPL Mathces 2008-2023.csv')
byb.head()

# Fantasy Points
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

# Team squads
csk = ['MS Dhoni', 'Shaik Rasheed', 'Shivam Dube', 'RD Gaikwad', 'DL Chahar', 'RA Jadeja', 'AM Rahane', 'M Theekshana', 'TU Deshpande', 'Simarjeet Singh', 'MM Ali']
gt =  ['Rashid Khan', 'Shubman Gill', 'Mohammed Shami', 'WP Saha', 'DA Miller', 'V Shankar', 'MS Wade', 'J Yadav', 'KS Williamson', 'R Sai Kishore', 'MM Sharma']

# Initial fantasy points for players
csk_fp = {player: 0 for player in csk}
gt_fp = {player: 0 for player in gt}

def calculate_fantasy_points(team1, team2, team1_fp):
    fantasy_team_players = []

    for player in team1:
        unq_ids = byb[byb["batsman"] == player]['id'].unique()
        matches_played = len(unq_ids)
        print("Number of matches played", matches_played, player)

        bbr = []
        for x in unq_ids:
            bat_run = byb[(byb["batsman"] == player) & (byb['id'] == x)]['batsman_runs'].sum()
            bbr.append(bat_run)

        r30, r50, r100 = 0, 0, 0
        for m in bbr:
            if m >= 100:
                r100 += 1
            elif m >= 50:
                r50 += 1
            elif m >= 30:
                r30 += 1

        catches = len(byb[(byb['fielder'] == player) & (byb['dismissal_kind'] == 'caught')]) / matches_played if matches_played else 0
        run_outs = len(byb[(byb['fielder'] == player) & (byb['dismissal_kind'] == 'run out')]) / matches_played if matches_played else 0
        extra_points = (r30 * Batsman_points['30Runs'] + r50 * Batsman_points['Half_century'] + r100 * Batsman_points['Century'] + catches * Fielding_points['Catch'] + run_outs * Fielding_points['RunOutInd'])

        # Extra Points for bowlers
        wickets_taken = []
        for x in unq_ids:
            twx = byb[(byb["bowler"] == player) & (byb['id'] == x)]['is_wicket'].sum()
            wickets_taken.append(twx)

        w3, w4, w5 = 0, 0, 0
        for z in wickets_taken:
            if z >= 5:
                w5 += 1
            elif z >= 4:
                w4 += 1
            elif z >= 3:
                w3 += 1

        lbws = len(byb[(byb['bowler'] == player) & (byb['dismissal_kind'] == 'lbw')]) / matches_played if matches_played else 0
        bowled = len(byb[(byb['bowler'] == player) & (byb['dismissal_kind'] == 'bowled')]) / matches_played if matches_played else 0
        wexp = (w3 * Bowling_points['3W'] + w4 * Bowling_points['4W'] + w5 * Bowling_points['5W'] + lbws * Bowling_points['LBW_Bowled'] + bowled * Bowling_points['LBW_Bowled'])

        ffp = []
        for opponent in team2:
            bat_vs_bowl = byb[(byb["batsman"] == player) & (byb["bowler"] == opponent)]
            bowls_played = len(bat_vs_bowl.batsman_runs)
            runs_scored = bat_vs_bowl.batsman_runs.sum()
            fours = len(bat_vs_bowl[bat_vs_bowl['batsman_runs'] == 4])
            sixes = len(bat_vs_bowl[bat_vs_bowl['batsman_runs'] == 6])
            wicket = bat_vs_bowl.is_wicket.sum()

            penalty = 0
            if bowls_played <= 60 and wicket >= 5:
                penalty = -16
                print(f"{player} has been dismissed {wicket} times by {opponent}")
            elif bowls_played <= 48 and wicket >= 4:
                penalty = -8
                print(f"{player} has been dismissed {wicket} times by {opponent}")
            elif bowls_played <= 36 and wicket >= 3:
                penalty = -4
                print(f"{player} has been dismissed {wicket} times by {opponent}")

            strike_rate = int(runs_scored / bowls_played * 100) if bowls_played else 'NA'
            if bowls_played >= 10 and strike_rate != 'NA':
                if strike_rate >= 170:
                    print(f"{player} has scored {runs_scored} runs in {bowls_played} balls against {opponent}, strike rate: {strike_rate}")
                elif strike_rate >= 150:
                    print(f"{player} has scored {runs_scored} runs in {bowls_played} balls against {opponent}, strike rate: {strike_rate}")

            bowl_vs_bat = byb[(byb["bowler"] == player) & (byb["batsman"] == opponent)]
            wicket_took = bowl_vs_bat.is_wicket.sum()
            fantasy_points1 = runs_scored + fours * Batsman_points['bFour'] + sixes * Batsman_points['bSix'] - wicket * Bowling_points['Wicket'] + wicket_took * Bowling_points['Wicket'] + penalty
            ffp.append(fantasy_points1)
            print(f"{player} against {opponent}: Runs {runs_scored}, Balls {bowls_played}, Strike Rate {strike_rate}, Wickets {wicket}, Fours {fours}, Sixes {sixes}, Fantasy Points {fantasy_points1}")

        sum_ffp = sum(ffp)
        recent_performance_points = np.log(team1_fp[player]) if team1_fp[player] > 0 else (-np.log(abs(team1_fp[player])) if team1_fp[player] < 0 else 0)
        recent_performance_points = team1_fp[player] / 3  # New method for recent performance points

        weight1 = 0.5
        weight2 = 1 - weight1
        final_fantasy_point = (sum_ffp + extra_points + wexp) * weight1 + recent_performance_points * weight2
        final_fantasy_point = round(final_fantasy_point, 2)
        fantasy_team_players.append((final_fantasy_point, player))
        fantasy_team_players.sort(reverse=True)
        print(f"Fantasy points of {player}: {final_fantasy_point}")
    
    return fantasy_team_players

def performance_against_team(player, against_team):
    performance = {
        'total_runs': 0, 'total_balls': 0, 'total_wickets': 0, 'total_fours': 0, 'total_sixes': 0,
        'dismissals': 0
    }
    
    for opponent in against_team:
        player_vs_opponent = byb[(byb["batsman"] == player) & (byb["bowler"] == opponent)]
        performance['total_runs'] += player_vs_opponent['batsman_runs'].sum()
        performance['total_balls'] += len(player_vs_opponent)
        performance['total_fours'] += len(player_vs_opponent[player_vs_opponent['batsman_runs'] == 4])
        performance['total_sixes'] += len(player_vs_opponent[player_vs_opponent['batsman_runs'] == 6])
        performance['dismissals'] += player_vs_opponent['is_wicket'].sum()
    
    return performance

def update_team_performance_with_opponent(team1, team2, team1_fp):
    for player in team1:
        performance = performance_against_team(player, team2)
        team1_fp[player] += (performance['total_runs'] + performance['total_fours']*Batsman_points['bFour'] + performance['total_sixes']*Batsman_points['bSix'] - performance['dismissals']*Bowling_points['Wicket'])
    return team1_fp

# Updating team performance against specific opponents
csk_fp = update_team_performance_with_opponent(csk, gt, csk_fp)
gt_fp = update_team_performance_with_opponent(gt, csk, gt_fp)

# Calculating fantasy points considering performance against the opponent team
fantasy_team_players = calculate_fantasy_points(csk, gt, csk_fp)
print("Top 11 players based on fantasy points:", fantasy_team_players[:11])
