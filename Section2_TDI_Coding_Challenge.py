import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.stats import chisquare
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
import requests
from bs4 import BeautifulSoup
import math
import statistics

############################ SECTION 2 ############################

# The City Record is the official journal of New York City, and provides information provided by city agencies. 
# This data is available in searchable form online at the City Record Online (CROL). For this challenge, 
# we will use a subset of the CROL data consisting only of procurement notices for goods and services.

## MAIN PROGRAM STARTS HERE ##

# Reading the CSV file
CROL_AllData = pd.read_csv('Recent_Contract_Awards.csv')

# %%
# Q1: 
# Keep only rows with a StartDate occurring from 2010 to 2019, inclusive. 
# Next, remove all rows for which the ContractAmount field is less than or equal to zero, or is missing entirely. 
# Use this filtered data for the rest of the challenge, as well. For the remaining data, what is the total sum of contract amounts?

CROL_AllData['StartDate'] = pd.to_datetime(CROL_AllData['StartDate'])
CROL_AllData['EndDate'] = pd.to_datetime(CROL_AllData['EndDate'])
CROL_FiltDate = CROL_AllData[(CROL_AllData['StartDate'].dt.year > 2009) & (CROL_AllData['StartDate'].dt.year < 2020)]
CROL_FiltDateAmount = CROL_FiltDate[(CROL_FiltDate.ContractAmount.notnull()) & (
    CROL_FiltDate['ContractAmount'] > 0)].reset_index()
TotSum_ContAmounts = CROL_FiltDateAmount['ContractAmount'].sum()

# %%
# Q2:
# Determine the number of contracts awarded by each agency. 
# For the top 5 agencies in terms the number of contracts, 
# compute the mean ContractAmount per contract. Among these values, 
# what is the ratio of the highest mean contract amount to the second highest?

ContractsPerAgency = CROL_FiltDateAmount['AgencyName'].value_counts().index.tolist() # This already sorts the data in descending order

Top5_MeanContAmounts = []
for nm in ContractsPerAgency[:5]:
    df = CROL_FiltDateAmount[CROL_FiltDateAmount['AgencyName'] == nm]
    Top5_MeanContAmounts.append(df['ContractAmount'].mean())

Top5_MeanContAmounts.sort(reverse=True)
Ratio_MeanContAmount = Top5_MeanContAmounts[0]/Top5_MeanContAmounts[1]

# %%
# Q3:
# Consider only procurements made by the Citywide Administrative Services (CAS) agency 
# and compute the sum contract amount awarded to each unique vendor. 
# What proportion of the total number of contracts in the data set were awarded 
# to the top 50 vendors?

CROL_FiltDateAmount_CAS = CROL_FiltDateAmount[CROL_FiltDateAmount['AgencyName']
                                              == 'Citywide Administrative Services']
SumContAmount_CASbyVendor = CROL_FiltDateAmount_CAS.groupby(['VendorName'])[['ContractAmount']].sum()
CAS_Vendors = SumContAmount_CASbyVendor.sort_values('ContractAmount',ascending=False).index.tolist()
CROL_FiltDateAmount_CAS_Top50 = CROL_FiltDateAmount_CAS[CROL_FiltDateAmount_CAS['VendorName'].isin(CAS_Vendors[:50])]
PropContracts_Top50 = len(CROL_FiltDateAmount_CAS_Top50)/len(CROL_FiltDateAmount_CAS)

# %%
# Q4:
# Do agencies publish procurement notices uniformly throughout the week? 
# As an example, consider the agency of Parks and Recreation (PnR). For this agency, 
# compute the weekday for which each notice was published, and perform a 
# Chi-squared test on the null hypothesis that each weekday occurs equally often. 
# Report the value of the test statistic.

CROL_FiltDateAmount_PnR = CROL_FiltDateAmount[CROL_FiltDateAmount['AgencyName'] 
                            == 'Parks and Recreation']
Weekday_PnR_Notices = []
for val in CROL_FiltDateAmount_PnR['StartDate']:
    wkday = datetime(val.year,val.month,val.day).weekday()
    Weekday_PnR_Notices.append(wkday)

fig = go.Figure()
fig.add_trace(go.Histogram(x=Weekday_PnR_Notices))
fig.update_traces(opacity=0.75)  # Reduce opacity
fig.show()

Weekday_counts = pd.Series(Weekday_PnR_Notices).value_counts().sort_index()
Weekday_counts_ChiSq = chisquare(Weekday_counts)

# %%
# Q5:
# For this question, consider only contracts with in the categories of 
# Construction Related Services and Construction/Construction Services (CS). 
# The ShortTitle field contains a description of the procured goods/services 
# for each contract. Compute the sum contract amount for contracts whose 
# ShortTitle refer to 'CENTRAL PARK' and for those which refer to 
# 'WASHINGTON SQUARE PARK'. What is the ratio of total construction and 
# contruction-related expenditure for the Central Park contracts compared to the 
# Washington Square Park contracts? Note: you should ensure that 'PARK' 
# appears on its own and not as the beginning of another word.

CROL_FiltDateAmount_CatCS = CROL_FiltDateAmount[CROL_FiltDateAmount['CategoryDescription'].isin(
    ['Construction Related Services','Construction/Construction Services'])]
CROL_FiltDateAmount_CatCS.reset_index(inplace=True)

ContAmount_CatCS_CP = []
ContAmount_CatCS_WSP = []
for i, nm in enumerate(CROL_FiltDateAmount_CatCS['ShortTitle']):
    if 'CENTRAL PARK' in nm.upper():    # Case insensitive
        ContAmount_CatCS_CP.append(CROL_FiltDateAmount_CatCS.loc[i,'ContractAmount'])
    if 'WASHINGTON SQUARE PARK' in nm.upper():  # Case insensitive
        ContAmount_CatCS_WSP.append(CROL_FiltDateAmount_CatCS.loc[i, 'ContractAmount'])

Ratio_CPtoWSP_Contracts = sum(ContAmount_CatCS_CP)/sum(ContAmount_CatCS_WSP)

# %%
# Q6:
# Is there a predictable, yearly pattern of spending for certain agencies? 
# As an example, consider the Environmental Protection agency (EPA). For each month 
# from 2010 through the end of 2019, compute the monthly expenditure for each agency. 
# Once again, use StartDate for the contract date. Then, with a lag of 12 months, 
# report the autocorrelation for total monthly expenditure.

CROL_FiltDateAmount_EPA = CROL_FiltDateAmount[CROL_FiltDateAmount['AgencyName']
                                              == 'Environmental Protection']

CROL_EPA_MonthlyExp = pd.pivot_table(CROL_FiltDateAmount_EPA, values='ContractAmount', 
    index=[CROL_FiltDateAmount_EPA['StartDate'].dt.year, CROL_FiltDateAmount_EPA['StartDate'].dt.month], 
    columns=None, aggfunc='sum', fill_value=0)

CROL_EPA_MonthlyExp.index = CROL_EPA_MonthlyExp.index.set_names(['Year', 'Month'])
CROL_EPA_MonthlyExp.reset_index(inplace=True)

ACorr_EPA_MonthlyExp = CROL_EPA_MonthlyExp['ContractAmount'].autocorr(lag=12)
pd.plotting.autocorrelation_plot(CROL_EPA_MonthlyExp['ContractAmount'])

# %%
# Q7:
# Consider only contracts awarded by the Citywide Administrative Services (CAS) agency 
# in the category Goods. Compute the total yearly expenditure (using StartDate) 
# for these contracts and fit a linear regression model to these values. 
# What is the R^2 value for this model?

CROL_FiltDateAmount_CAS_Goods = CROL_FiltDateAmount_CAS[CROL_FiltDateAmount_CAS['CategoryDescription'] == 'Goods']
CROL_CASgoods_YearlyExp = pd.pivot_table(CROL_FiltDateAmount_CAS_Goods, values='ContractAmount',
                                     index=None, columns=CROL_FiltDateAmount_CAS_Goods['StartDate'].dt.year,
                                     aggfunc='sum', fill_value=0)

RegModel_list = list(zip(list(CROL_CASgoods_YearlyExp),list(CROL_CASgoods_YearlyExp.loc['ContractAmount'])))
RegModel_df = pd.DataFrame(RegModel_list, columns=['Year','YearlyExp'])

fig = px.scatter(RegModel_df, x="Year", y="YearlyExp")
fig.show()

x_vals = np.array(list(CROL_CASgoods_YearlyExp)).reshape((-1, 1))
y_vals = np.array(list(CROL_CASgoods_YearlyExp.loc['ContractAmount']))

model = LinearRegression().fit(x_vals, y_vals)
r_sq = model.score(x_vals, y_vals)

# %%
# Q8:
# In this question, we will examine whether contract expenditure goes to companies 
# located within or outside of New York City. To do so, we will extract the ZIP codes 
# from the VendorAddress field. The ZIP codes pertaining to New York City can be found 
# at the following URL: https: // www.health.ny.gov/statistics/cancer/registry/appendix/neighborhoods.htm. 
# Looking only at contracts with a StartDate in 2018, compute the total expenditure for contracts awarded 
# to vendors listing NYC addresses and those located elsewhere. Report the proportion of the total 
# expenditures awarded to the NYC vendors.

CROL_FiltDateAmount_2018 = CROL_FiltDateAmount[CROL_FiltDateAmount['StartDate'].dt.year == 2018]
CROL_FiltDateAmount_2018.reset_index(inplace=True)

Zipcodes = []
for addr in CROL_FiltDateAmount_2018['VendorAddress']:
    zcode = addr.split(" ")
    Zipcodes.append(zcode[-1])

for i, zc in enumerate(Zipcodes):
    zcode = zc.split("-")
    Zipcodes[i] = zcode[0]

for i, zc in enumerate(Zipcodes):
    zcode = zc.split(",")
    Zipcodes[i] = zcode[-1]

URL = 'https://www.health.ny.gov/statistics/cancer/registry/appendix/neighborhoods.htm'
page = requests.get(URL)
soup = BeautifulSoup(page.content, 'html.parser')
results = soup.find(id='content')
zcode_elements = results.find_all('td', headers="header3")
NYzcodes = pd.Series()
for zc in zcode_elements:
    zc_split = pd.Series(zc.text.split(","))
    NYzcodes = NYzcodes.append(zc_split, ignore_index=True)

for i, zc in enumerate(NYzcodes):
    zc_split = zc.split(" ")
    NYzcodes[i] = zc_split[-1]

NYC_ContExp = []
for i in range(len(CROL_FiltDateAmount_2018)):
    if Zipcodes[i] in list(NYzcodes):
        NYC_ContExp.append(CROL_FiltDateAmount_2018.loc[i, 'ContractAmount'])

Prop_ExpNYCvendors = sum(NYC_ContExp)/CROL_FiltDateAmount_2018['ContractAmount'].sum()
