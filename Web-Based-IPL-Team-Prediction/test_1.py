from flask import Flask, request, render_template,redirect,url_for
import matplotlib.pyplot as plt
#import seborn as sns
import pandas as pd
import numpy as np

app = Flask(__name__)

Team_1=  []
Team_2 = []
Team1_Squad = {}
Team2_Squad = {}

# Importing basic libraries
# Reading files
byb=pd.read_csv('IPl Ball-by-Ball 2008-2023.csv')
match= pd.read_csv('IPL Mathces 2008-2023.csv')
byb.head()

# Fantasy Points

Batsman_points = {'Run':1, 'bFour':1, 'bSix':2, '30Runs':4,
        'Half_century':8, 'Century':16, 'Duck':-2, '170sr':6,
                 '150sr':4, '130sr':2, '70sr':-2, '60sr':-4, '50sr':-6}

Bowling_points = {'Wicket':25, 'LBW_Bowled':8, '3W':4, '4W':8, 
                  '5W':16, 'Maiden':12, '5rpo':6, '6rpo':4, '7rpo':2, '10rpo':-2,
                 '11rpo':-4, '12rpo':-6}

Fielding_points = {'Catch':8, '3Cath':4, 'Stumping':12, 'RunOutD':12,
                  'RunOutInd':6}

# Storing team players
# Here I have to do manual work... choosing the players after the toss and putting them here


#TEAM 3
ind = ['RG Sharma', 'YBK Jaiswal', 'V Kohli','SA Yadav', 'RR Pant' ,'RA Jadeja','HH Pandya','YS Chahal','JJ Bumrah','Mohammed Siraj','Arshdeep Singh']
aus =  ['MA Starc', 'TM Head', 'JP Inglis', 'DA Warner', 'MR Marsh', 'GJ Maxwell','C Green','PJ Cummins','MP Stoinis','A Zampa','JR Hazlewood']

ind_fp = {
    "RG Sharma": 0,
    "YBK Jaiswal": 0,
    "V Kohli": 0,
    "SA Yadav": 0,
    "RA Jadeja": 0,
    "RR Pant": 0,
    "HH Pandya": 0,
    "YS Chahal": 0,
    "JJ Bumrah": 0,
    "Mohammed Siraj": 0,
    "Arshdeep Singh": 0,
    "AR Patel": 0,
    "KK Ahmed": 0,
    "Kuldeep Yadav": 0,
    "Avesh Khan": 0,
    
}

aus_fp = {
    "MA Starc": 0,
    "MS Wade": 0,
    "TM Head": 0,
    "JP Inglis": 0,
    "DA Warner": 0,
    "MR Marsh": 0,
    "GJ Maxwell": 0,
    "C Green": 0,
    "PJ Cummins": 0,
    "MP Stoinis": 0,
    "A Zampa": 0,
    "JR Hazlewood": 0,
    "AC Agar": 0,
    "NT Ellis": 0,
    "TH David":0,
}


# ruturaj, A rahane , 
ind = [
    "RG Sharma",
    "YBK Jaiswal",
    "V Kohli",
    "SA Yadav",
    "RR Pant",
    "HH Pandya",
    "RA Jadeja",
    "YS Chahal",
    "JJ Bumrah",
    "Arshdeep Singh",
    "Mohammed Siraj",
]

aus = [
    "DA Warner",
    "TM Head",
    "MR Marsh",
    "GJ Maxwell",
    "JP Inglis",
    "MP Stoinis",
    "C Green",
    "A Zampa",
    "PJ Cummins",
    "MA Starc",
    "JR Hazlewood",
]


#TEAM 9



def get_players(team1,team2,team1_fp):
    fantasy_team_players = []

    for i in range(len(team1)):
        unq_ids = byb[byb["batsman"]==team1[i]]['id'].unique()
        mathces_played = len(unq_ids)
        print ( "Number of matches played" , len(unq_ids),team1[i])
        bbr = []
        for x in unq_ids:
            bat_run = sum(byb[(byb["batsman"]==team1[i])&(byb['id']==x)]['batsman_runs'])
            bbr.append(bat_run)

        r30,r50,r100 =0,0,0
        for m in bbr:
            if m>=100:
                r100+=1
            elif m>=50:
                r50+=1
            elif m>=30:
                r30+=1
        try:
            catches = len(byb[(byb['fielder']==team1[i]) & (byb['dismissal_kind']=='caught')])/mathces_played
            run_outs = len(byb[(byb['fielder']==team1[i]) & (byb['dismissal_kind']=='run out')])/mathces_played
            extra_points = r30/mathces_played*Batsman_points['30Runs'] +r50/mathces_played*Batsman_points['Half_century'] +r100/mathces_played*Batsman_points['Century'] +catches*Fielding_points['Catch']+run_outs*Fielding_points['RunOutInd']
        except:
            catches, run_outs, extra_points = 0,0,0
        
        # Extra Points for bowlers to be estimated here
        wickets_taken = []
        for x in unq_ids:
            twx = sum(byb[(byb["bowler"]==team1[i]) & (byb['id']==x)]['is_wicket'])
            wickets_taken.append(twx)

        w3,w4,w5 = 0,0,0
        for z in wickets_taken:
            if z>=5:
                w5+=1
            elif z>=4:
                w4+=1
            elif z>=3:
                w3+=1
        try:
            lbws = len((byb[(byb['bowler']==team1[i]) & (byb['dismissal_kind']=='lbw')]))/mathces_played      
            bowled = len((byb[(byb['bowler']==team1[i]) & (byb['dismissal_kind']=='bowled')]))/mathces_played      
            wexp = w3/mathces_played*Bowling_points['3W'] + w4/mathces_played*Bowling_points['4W'] + w5/mathces_played*Bowling_points['5W'] + lbws*Bowling_points['LBW_Bowled'] + bowled*Bowling_points['LBW_Bowled']
        except:
            lbws, bowled, wexp = 0,0,0
        
        ffp = []
        for j in range(len(team2)):
            bat_vs_bowl = byb[(byb["batsman"]==team1[i]) & (byb["bowler"]==team2[j])]
            bowls_played = len(bat_vs_bowl.batsman_runs)
            runs_scored = sum(bat_vs_bowl.batsman_runs)
            fours = len(bat_vs_bowl[bat_vs_bowl['batsman_runs']==4])
            sixes = len(bat_vs_bowl[bat_vs_bowl['batsman_runs']==6])
            wicket = sum(bat_vs_bowl.is_wicket)
            if bowls_played <=6*10 and wicket >=5:
                penalty = -16
                print (team1[i], "ka wicket taken",wicket,"times by", team2[j])
            elif bowls_played <=6*8 and wicket >=4:
                penalty = -8
                print (team1[i], "ka wicket taken",wicket,"times by", team2[j])
            elif bowls_played <=6*6 and wicket >=3:
                penalty = -4
                print (team1[i], "'s wicket taken",wicket,"times by", team2[j])
            else:
                penalty = 0

            try:    
                strike_rate = int(runs_scored/bowls_played*100)
            except: 
                strike_rate = 'NA'            
            if bowls_played >=10 and strike_rate!='NA':
                if strike_rate >=170:
                    print (team1[i] ,"beaten", team2[j], "Runs", runs_scored,"bowls",bowls_played,"strike rate", strike_rate,'Out',wicket,'times', "Fours", fours,"Sixes", sixes)            
                elif strike_rate >=150:
                    print (team1[i] ,"beaten", team2[j], "Runs", runs_scored,"bowls",bowls_played,"strike rate", strike_rate,'Out',wicket,'times', "Fours", fours,"Sixes", sixes)            
   
            bowl_vs_bat = byb[(byb["bowler"]==team1[i]) & (byb["batsman"]==team2[j])]
            wicket_took = sum(bowl_vs_bat.is_wicket)
            fantasy_points1 = runs_scored + fours*Batsman_points['bFour'] + sixes*Batsman_points['bSix'] - wicket*Bowling_points['Wicket'] + wicket_took*Bowling_points['Wicket'] + penalty 
            ffp.append(fantasy_points1)
            print (team1[i] ,"against", team2[j], "Runs", runs_scored, 
                     "bowls",bowls_played,"strike rate", strike_rate,
                     'Out',wicket,'times', "Fours", fours,"Sixes", sixes, "fatansy points",fantasy_points1)
        sum_ffp = sum(ffp)
        if team1_fp[team1[i]] > 0:
            recent_performace_points = np.log(team1_fp[team1[i]])
        elif team1_fp[team1[i]] <0:
            recent_performace_points = -np.log(abs(team1_fp[team1[i]]))
        else:
            recent_performace_points = 0
        # Trying a new method for recent performancec point
        recent_performace_points = team1_fp[team1[i]]/3
        weight1 = 0.5
        weight2 = 1 - weight1
        final_fantasy_point = (sum_ffp + extra_points + wexp)*weight1 + recent_performace_points*weight2
        final_fantasy_point = round(final_fantasy_point,2)
        fantasy_team_players.append((final_fantasy_point,team1[i]))
        fantasy_team_players.sort(reverse=True)
        print ("Fatasy points of",team1[i],final_fantasy_point)
    return fantasy_team_players

#get_players(aus,ind,aus_fp)

t1 = get_players(aus,ind,aus_fp)
t2 = get_players(ind,aus,ind_fp)

# [(92.08, 'RD Gaikwad'), (69.23, 'MM Ali'), (44.61, 'RA Jadeja'), (33.57, 'MS Dhoni'), (17.5, 'DL Chahar'), (1.33, 'TU Deshpande'), (0.0, 'Simarjeet Singh'), (0.0, 'Shivam Dube'), (0.0, 'Shaik Rasheed'), (0.0, 'M Theekshana'), (-3.3, 'AM Rahane')]
# [(113.84, 'Shubman Gill'), (108.83, 'Mohammed Shami'), (90.09, 'WP Saha'), (74.7, 'Rashid Khan'), (49.64, 'DA Miller'), (34.43, 'KS Williamson'), (26.77, 'V Shankar'), (16.5, 'J Yadav'), (1.81, 'MS Wade'), (0.0, 'Noor Ahmad'), (0.0, 'DG Nalkande')]


t3 = t1 + t2
t3.sort(reverse=True)
Team = pd.DataFrame(t3)
Result = Team[1].head(11)
Result = pd.DataFrame(Result)
print('\nFinal Predicted Team',Result)