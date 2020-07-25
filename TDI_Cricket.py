import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import csv
import glob
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import math

# Defining a function to read all CSV files and parse data
def ReadCSVfiles(csv_fpath, game_type):
    filenames = glob.glob(csv_fpath + game_type + "\*.csv")
    CSVnames = []
    csv_data = []
    fname_list = []
    for filename in filenames:
        name = filename.split("\\")
        CSVnames.append(name[-1])
        with open(filename) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                csv_data.append(row)
                fname = filename.split("\\")
                fname_list.append(fname[-1])

    games_data_ALL = pd.DataFrame(csv_data, columns=['ball', 'inning', 'over', 'team', 'striker', 'non_striker', 'bowler', 'runs', 'extras', 'wicket', 'player_out'])
    games_data_ALL['CSVfname'] = fname_list
    return games_data_ALL, len(filenames), CSVnames

# Defining a function to calculate total scores by Team 1 and Team 2
def MatchInsights(games_data, games_tot, games_ind, games_fnames):
    cols_team1 = ['T1_TeamName', 'T1_TotalScore', 'T1_Wickets', 'T1_TotalOvers',
                  'T1_Score_10overs','T1_TotalRR','T1_10ovRR']
    cols_team2 = ['T2_TeamName', 'T2_TotalScore', 'T2_Wickets', 'T2_TotalOvers',
                  'T2_Score_10overs', 'T2_TotalRR', 'T2_10ovRR']
    team1_data = pd.DataFrame(index=None, columns=cols_team1)
    team2_data = pd.DataFrame(index=None, columns=cols_team2)
    match_date = []
    match_MoM = []
    team1_scores, team1_overs, team1_names, team1_10scores, team1_totRR, team1_10ovRR, team1_wickets = [], [], [], [], [], [], []
    team2_scores, team2_overs, team2_names, team2_10scores, team2_totRR, team2_10ovRR, team2_wickets = [], [], [], [], [], [], []
    
    for i in range(games_tot):
        st = games_ind[i]
        en = games_ind[i+1]
        temp = games_data.iloc[st:en] 

        # Match date
        date_id = temp[temp['inning'].isin(['date'])]
        full_date = date_id.iloc[0]["over"]
        match_date.append(datetime.strptime(full_date,'%Y/%m/%d'))

        # Player of the match: MoM
        MoM_id = temp[temp['inning'].isin(['player_of_match'])]
        if MoM_id.empty:
            match_MoM.append('No MoM')
        else:
            MoM_name = MoM_id.iloc[0]["over"]
            match_MoM.append(MoM_name)

        ## COLLECTING QUANTITATIVE INSIGHTS FOR TEAM1 ##
        temp1 = temp[temp['inning'] == '1']
        if temp1.empty:
            team1_scores.append(0)
            team1_overs.append(0)
            team1_names.append('NR')
            team1_10scores.append(0)
            team1_totRR.append(0)
            team1_10ovRR.append(0)
            team1_wickets.append(-1)
        else:
            # Total Score
            temp1_score = temp1['runs'].apply(pd.to_numeric).sum() + temp1['extras'].apply(pd.to_numeric).sum()
            team1_scores.append(temp1_score)
            # Total overs faced
            temp1 = temp1.reset_index()
            temp1_overs = float(temp1.loc[len(temp1)-1, "over"])
            team1_overs.append(temp1_overs)            
            # Team1 name
            temp1_name = temp1.loc[0, "team"]
            team1_names.append(temp1_name)
            # Total runs scored in first 10 overs: Relevant only for ODIs
            if temp1_overs > 10:
                temp1_10ov = temp1[temp1['over'].apply(pd.to_numeric) <= 10]
                temp1_10score = temp1_10ov['runs'].apply(pd.to_numeric).sum() + temp1_10ov['extras'].apply(pd.to_numeric).sum()
                team1_10scores.append(temp1_10score)
            else:
                team1_10scores.append(0)
            # Total Run-Rate (RR)
            temp1_ovsplit = math.modf(temp1_overs)
            temp1_balls = temp1_ovsplit[1]*6 + min(round(temp1_ovsplit[0]*10),6) # Eliminating extra balls bowled > 6
            team1_totRR.append(temp1_score/temp1_balls*6)
            # 10overs Run-Rate
            if temp1_overs > 10:
                team1_10ovRR.append(temp1_10score/10)
            else:
                team1_10ovRR.append(0)
            # Wickets
            temp1_wkts = [wkts1 for wkts1 in temp1["wicket"] if wkts1]
            team1_wickets.append(len(temp1_wkts))            
        
        ## COLLECTING QUANTITATIVE INSIGHTS FOR TEAM2 ##
        temp2 = temp[temp['inning'] == '2']
        if temp2.empty:
            team2_scores.append(0)
            team2_overs.append(0)
            team2_names.append('NR')
            team2_10scores.append(0)
            team2_totRR.append(0)
            team2_10ovRR.append(0)
            team2_wickets.append(-1)
        else:
            # Total Score
            temp2_score = temp2['runs'].apply(pd.to_numeric).sum() + temp2['extras'].apply(pd.to_numeric).sum()
            team2_scores.append(temp2_score)
            # Total overs faced
            temp2 = temp2.reset_index()
            temp2_overs = float(temp2.loc[len(temp2)-1, "over"])
            team2_overs.append(temp2_overs)            
            # Team2 name
            temp2_name = temp2.loc[0, "team"]
            team2_names.append(temp2_name)
            # Total runs scored in first 10 overs: Relevant only for ODIs
            if temp2_overs > 10:
                temp2_10ov = temp2[temp2['over'].apply(pd.to_numeric) <= 10]
                temp2_10score = temp2_10ov['runs'].apply(pd.to_numeric).sum() + temp2_10ov['extras'].apply(pd.to_numeric).sum()
                team2_10scores.append(temp2_10score)
            else:
                team2_10scores.append(0)
            # Total Run-Rate (RR)
            temp2_ovsplit = math.modf(temp2_overs)
            temp2_balls = temp2_ovsplit[1]*6 + min(round(temp2_ovsplit[0]*10),6) # Eliminating extra balls bowled > 6
            team2_totRR.append(temp2_score/temp2_balls*6)
            # 10overs Run-Rate
            if temp2_overs > 10:
                team2_10ovRR.append(temp2_10score/10)
            else:
                team2_10ovRR.append(0)
            # Wickets
            temp2_wkts = [wkts2 for wkts2 in temp2["wicket"] if wkts2]
            team2_wickets.append(len(temp2_wkts))
    
    # CSV Filenames
    team1_data = team1_data.assign(T1_TeamName=list(team1_names), T1_TotalScore=list(team1_scores), T1_Wickets=list(team1_wickets), T1_TotalOvers=list(team1_overs), T1_Score_10overs=list(team1_10scores), T1_TotalRR=list(team1_totRR), T1_10ovRR=list(team1_10ovRR))
    team2_data = team2_data.assign(T2_TeamName=list(team2_names), T2_TotalScore=list(team2_scores), T2_Wickets=list(team2_wickets), T2_TotalOvers=list(team2_overs), T2_Score_10overs=list(team2_10scores), T2_TotalRR=list(team2_totRR), T2_10ovRR=list(team2_10ovRR))

    return team1_data, team2_data, match_date, match_MoM

def MatchInfo(AllData, Flag, *ColInfo):
    ColInfo = list(ColInfo)
    info_id = AllData[AllData["inning"].isin(ColInfo)].reset_index()
    if Flag:
        ODI_info = list(info_id["over"])
    else:
        ODI_info = list(info_id["inning"])
    return ODI_info

## MAIN PROGRAM

path = r"C:\Users\aramanujam\Documents\Tha Data Incubator\Cricket\CSV format\MENS"

# Parsing data for ODI matches
ODI_games, ODI_tot, ODI_CSVfnames = ReadCSVfiles(path, '\ODI')
ODI_games_ind = ODI_games[ODI_games['ball'] == "version"].index.tolist()
ODI_games_ind.append(len(ODI_games))
ODI_Team1_AllData, ODI_Team2_AllData, ODI_Dates, ODI_MoMs = MatchInsights(ODI_games, ODI_tot, ODI_games_ind, ODI_CSVfnames)

# Initializing DataFrame for specific match info for ODIs
cols_games = ['Date', 'Venue', 'City', 'TossWinner',
              'TossDecision', 'MoM', 'Winner', 'WinMargin', 'WinMarginType']
ODI_game_info = pd.DataFrame(index=None, columns=cols_games)

ODI_Venues = MatchInfo(ODI_games, True, 'venue')
ODI_Cities = MatchInfo(ODI_games, True, 'city')
ODI_TossWinners = MatchInfo(ODI_games, True, 'toss_winner')
ODI_TossDecisions = MatchInfo(ODI_games, True, 'toss_decision')
ODI_Winners = MatchInfo(ODI_games, True, 'winner', 'outcome')
ODI_WinMargins = MatchInfo(ODI_games, True, 'winner_runs', 'winner_wickets', 'outcome')
ODI_WinMarginsType = MatchInfo(ODI_games, False, 'winner_runs', 'winner_wickets', 'outcome')

for i, WinBy in enumerate(ODI_WinMarginsType):
    str_split = WinBy.split('_')
    ODI_WinMarginsType[i] = str_split[-1]
    
ODI_game_info = ODI_game_info.assign(Date=ODI_Dates, Venue=ODI_Venues, City=ODI_Cities, TossWinner=ODI_TossWinners,
                                     TossDecision=ODI_TossDecisions, MoM=ODI_MoMs, Winner=ODI_Winners, WinMargin=ODI_WinMargins, WinMarginType=ODI_WinMarginsType)

ODIs_AllInfo = pd.concat([ODI_game_info, ODI_Team1_AllData, ODI_Team2_AllData], axis=1, sort=False)
ODIs_AllInfo = ODIs_AllInfo.assign(CSVfnames=ODI_CSVfnames)

## FILTERING DATASETS ##
ODIs_AllInfo_WithResult = ODIs_AllInfo[~ODIs_AllInfo['WinMargin'].isin(['no result', 'tie'])].reset_index()
ODIs_AllInfo_WithResult["Year"] = [yr.year for yr in ODIs_AllInfo_WithResult["Date"]]
ODIs_AllInfo_WithResult["WinMargin"]=ODIs_AllInfo_WithResult["WinMargin"].apply(pd.to_numeric)

for i, nm in enumerate(ODIs_AllInfo_WithResult["Winner"]):
    if nm == ODIs_AllInfo_WithResult.loc[i,"T1_TeamName"]:
        ODIs_AllInfo_WithResult.loc[i, "T1_W/L"] = 'W'
    else:
        ODIs_AllInfo_WithResult.loc[i, "T1_W/L"] = 'L'

for i, nm in enumerate(ODIs_AllInfo_WithResult["Winner"]):
    if nm == ODIs_AllInfo_WithResult.loc[i, "T2_TeamName"]:
        ODIs_AllInfo_WithResult.loc[i, "T2_W/L"] = 'W'
    else:
        ODIs_AllInfo_WithResult.loc[i, "T2_W/L"] = 'L'

ODIs_AllInfo_T1_50ov = ODIs_AllInfo_WithResult[ODIs_AllInfo_WithResult["T1_TotalOvers"] > 49.5]

## PLOTS ##

# Histogram for TotalScores from Team 1 and 2
fig = go.Figure()
fig.add_trace(go.Histogram(x=ODIs_AllInfo_WithResult['T1_TotalScore'], nbinsx=30))
fig.add_trace(go.Histogram(x=ODIs_AllInfo_WithResult['T2_TotalScore'], nbinsx=30))
fig.update_layout(barmode='overlay')  # Overlay both histograms
fig.update_traces(opacity=0.75)  # Reduce opacity to see both histograms
fig.show()

fig = px.box(ODIs_AllInfo_WithResult, x="Year", y="T1_TotalScore")
fig.show()

fig = px.box(ODIs_AllInfo_WithResult, x="Year", y="T2_TotalScore")
fig.show()

fig = px.scatter(ODIs_AllInfo_WithResult, x="T1_TotalScore", y="T2_TotalScore")
fig.show()

fig = px.scatter(ODIs_AllInfo_T1_50ov, x="Date", y="T1_TotalScore", color="T1_W/L")
fig.show()

