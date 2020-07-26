import warnings
import zipfile
import os
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
import cufflinks as cf
import chart_studio.plotly as py
sns.set_style("darkgrid")
warnings.filterwarnings("ignore")

# %%
# Function to read all CSV files and parse data
def ReadCSVfiles(csv_fpath, game_type):
    with zipfile.ZipFile(csv_fpath + game_type + "\CSVfiles.zip", "r") as zip_ref:
        zip_ref.extractall(csv_fpath + game_type)
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

# Function to compute details for Team 1 and Team 2
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

# Function to get match information
def MatchInfo(AllData, Flag, *ColInfo):
    ColInfo = list(ColInfo)
    info_id = AllData[AllData["inning"].isin(ColInfo)].reset_index()
    if Flag:
        ODI_info = list(info_id["over"])
    else:
        ODI_info = list(info_id["inning"])
    return ODI_info

# %%
## MAIN PROGRAM

path = os.getcwd()

# Parsing data for ODI matches
ODI_games, ODI_tot, ODI_CSVfnames = ReadCSVfiles(path, '\TDI_Cricket_CSVdata\ODI')
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


# %% TDI CODING CHALLENGE - EXPLORATORY PLOTS

# PLOT 1
# Filtering data for Team batting first i.e. Team 1, completing 50 overs and winning the game.
# X-axis is the total score after 50 overs, Y-axis shows the win margin
# This is to check the hypothesis that the more you score, the more convincingly you win the game (i.e. greater win margin)
sns.set(style="white", palette="muted", color_codes=True)

df1 = ODIs_AllInfo_T1_50ov[ODIs_AllInfo_T1_50ov['T1_W/L'] == 'W']
fig1_1 = sns.jointplot(
    x=df1["T1_TotalScore"], y=df1["WinMargin"], kind='hex', color='b')
fig1_1.savefig("PLOT1_1.png")
plt.show()

# Same data with a regression model
fig1_2 = sns.jointplot(
    x=df1["T1_TotalScore"], y=df1["WinMargin"], kind='reg', color='b')
fig1_2.savefig("PLOT1_2.png")
plt.show()

# PLOT 2
# Filtering data for Team batting first i.e. Team 1, and completing 50 overs regardless of the outcome.
# X-axis is the total score after the first 10 overs, Y-axis shows the total score after 50 overs.
# Does scoring heavily in the first 10 overs with fielding restrictions have an effect on the final score after 50 overs?
fig2_1 = sns.jointplot(
    x=ODIs_AllInfo_T1_50ov["T1_Score_10overs"], y=ODIs_AllInfo_T1_50ov["T1_TotalScore"],
    kind='hex', color='r')
fig2_1.savefig("PLOT2_1.png")
plt.show()

# Same data with a regression model
fig2_2 = sns.jointplot(
    x=ODIs_AllInfo_T1_50ov["T1_Score_10overs"], y=ODIs_AllInfo_T1_50ov["T1_TotalScore"],
    kind='reg', color='r')
fig2_2.savefig("PLOT2_2.png")
plt.show()

# PLOT 3
# Filtering data for Team batting first i.e. Team 1, and completing 50 overs
# Computing the win% for range of scores in steps of 25 runs. 
# At what total score can you ensure you have atleast 50% chance of winning the game?
# The box plot indicates that if the team scores atleast 250, their win% is >50% 
# while scoring >350 runs almost guarantees you a win.
step = 25
Score_range = list(np.arange(225, 425, step))
T1scores_YearlyWinPct = pd.DataFrame(index=Score_range)
T1_YrScoreWinpt = []
for yr in range(2006, 2021):
    T1scores_WinPct = []
    for i in Score_range:
        df = ODIs_AllInfo_T1_50ov[(ODIs_AllInfo_T1_50ov['T1_TotalScore'] <= i) &
                                  (ODIs_AllInfo_T1_50ov['T1_TotalScore'] > i-step) &
                                  (ODIs_AllInfo_T1_50ov['Date'].dt.year == yr)]
        if df.empty:
            T1scores_WinPct.append('')
            T1_YrScoreWinpt.append(['', '', ''])
        else:
            df_WinPct = len(df[df['T1_W/L'] == 'W'])/len(df)*100
            T1scores_WinPct.append(df_WinPct)
            T1_YrScoreWinpt.append([yr, str(i-step)+'-'+str(i), df_WinPct])
    T1scores_YearlyWinPct[str(yr)] = list(T1scores_WinPct)

T1_YrScoreWinpt = pd.DataFrame(T1_YrScoreWinpt, columns=[
                               'Year', 'Score Range', 'Win Pct'])
T1_YrScoreWinpt = T1_YrScoreWinpt[T1_YrScoreWinpt['Year'] != '']

# Box Plot
fig3 = go.Figure()
fig3 = px.box(T1_YrScoreWinpt, x="Score Range", y="Win Pct", points="all")
fig3.update_layout(title='Win Percentages for Range of Scores - Team Batting First',
                  xaxis_title='Score Range',
                  yaxis_title='Win Percentage')
fig3.show()

# PLOT 4
# History of ODI wins - The best teams between 2006-2020
TotWins_byCountry = ODIs_AllInfo_WithResult['Winner'].value_counts()
TotWins_byCountry = pd.DataFrame(TotWins_byCountry).reset_index()
TotWins_byCountry.columns = ["Country", "Total ODI Wins"]
fig4 = px.bar(TotWins_byCountry, x="Country",
             y="Total ODI Wins", color="Total ODI Wins", 
             text='Total ODI Wins', color_continuous_scale=px.colors.sequential.Cividis_r)
fig4.update_traces(textposition='outside')
fig4.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
fig4.show()
