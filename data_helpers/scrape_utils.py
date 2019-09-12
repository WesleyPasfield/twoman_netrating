import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, Comment
import requests
import re
import html5lib
import lxml

def advanced_stats_crawl(year_list):

	'''
	Function that crawls advanced stats for years provided by user. 2010-2019 have been tested and work
	year_list : List containing string values of all years you'd like to pull information for
	'''

	advanced_list = []

	## Loop across each year and grab advanced stats

	for y in np.arange(len(year_list)):
		print('Starting: ' + str(year_list[y]))
		page = requests.get('https://www.basketball-reference.com/leagues/NBA_' + year_list[y] + '_advanced.html')
		soup = BeautifulSoup(page.content, 'html.parser')
		ret = soup.find_all('tr', attrs = {'class': 'full_table'})
		for val in np.arange(len(ret)):

			## Required to get data into columns appropriately

			listval = advanced_list.append((ret[val].getText(separator = '|') + '|' + year_list[y]).replace('|*','').split('|'))

	player_df = pd.DataFrame(advanced_list)
	player_df.columns = ['dropit', 'player', 'pos', 'age', 'team', 'games', 'mp', 'per', 'ts', '3par', 'ftr', 'orb',
					'drb', 'trb', 'ast', 'stl', 'blk', 'tov', 'usg', 'ows', 'dws', 'ws', 'ws48', 'obpm', 'dbpm',
					'bpm', 'vorp', 'year']
	player_df.drop(['dropit'], axis = 1, inplace = True)

	## Need to get names in first initial. Last name form to join with two man lineups (annoying)

	first_init = player_df['player'].str.split().str[0]
	first_init = [val[0] for val in first_init]
	last_name = [val for val in player_df['player'].str.split().str[1]]
	player_df['player_join'] = [a + '. ' +  b for a, b in zip(first_init, last_name)]

	return player_df

def roster_crawl(team_list, year_list):

	'''
	Function that grabs roster information for each team over specified time frame
	team_list : list of teams that are required. Team changes may affect what gets pulled
	year_list : years you'd like to pull data for
	'''

	roster = []

	## For each year loop over every team and get player information

	for y in np.arange(len(year_list)):
		print('Starting: ' + str(year_list[y]))
		for t in np.arange(len(team_list)):
			page = requests.get('https://www.basketball-reference.com/teams/' + team_list[t] + '/' + year_list[y] + '.html#all_advanced')
			soup = BeautifulSoup(page.content, 'html.parser')
			ret = soup.find_all('tr')
			for val in np.arange(len(ret)):

				## This is required to put into columns appropriately

				listval = (year_list[y] + '|' + ret[val].getText(separator = '|')).replace('|*','').split('|')

				## Dropping college information - inconsistently stored

				listval = listval[0:9]

				## Required to knock out header data

				if listval[1] == '\n':
					continue
				roster.append(listval)

	## Stick in a dataframe and make height & weight numeric values

	roster_df = pd.DataFrame(roster)
	roster_df.columns = ['year', 'jers_num', 'player', 'position', 'height', 'weight', 'bday', 'country', 'yrpro']
	roster_df = roster_df[~roster_df['player'].isin(['PG', 'SG', 'SF', 'PF', 'C'])]
	roster_df['yrpro'] = np.where(roster_df['yrpro'] == 'R', '0', roster_df['yrpro']).astype(int)
	roster_df.drop(['jers_num'], inplace = True, axis = 1)
	heightft = roster_df['height'].str.split('-').str[0]
	heightft = [int(val) for val in heightft]
	heightinch = roster_df['height'].str.split('-').str[1]
	heightinch = [int(val) for val in heightinch]
	height_fin = [a * 12 +  b for a, b in zip(heightft, heightinch)]
	roster_df['height'] = height_fin

	return roster_df

def twoman_crawl(year_list, min_lim, offset):
	'''
	Function that gets two man lineup data for years specified
	year_list : List of Years (string) you'd like to pull data for
	min_lim : Minimum number of minutes two-man lineups need to be kept (string)
	offset : Required because results return in values of 100. Increase options if some combinations don't return
	'''
	twoman_list = []

	## Try except paradigm in case offset returns nothing and it throws exceptions
	## Loop across years and offsets to get two man combinations

	try:
		for y in np.arange(len(year_list)):
			print('Starting ' + year_list[y])
			for q in np.arange(len(offset)):
				print('offset = ' + str(offset[q]))
				page = requests.get('https://www.basketball-reference.com/play-index/lineup_finder.cgi?request=1&match=single&lineup_type=2-man&output=per_poss&is_playoffs=N&year_id='
									+ year_list[y] + '&game_num_min=0&game_num_max=99&c1stat=mp&c1comp=ge&c1val=' + min_lim +
									'&order_by=diff_pts&order_by_asc=&offset=' + offset[q])

				soup = BeautifulSoup(page.content, 'html.parser')
				ret = soup.find_all('tr')

				for val in np.arange(len(ret)):

					## Required to knock out header rows

					if val < 2:
						continue

					## Required to get data into columnar format

					listval = (ret[val].getText(separator = '|') + '|' + year_list[y]).replace('|*','').split('|')

					if listval[0] == '\n':
						continue

					twoman_list.append(listval)

	except Exception as e:
		print(e)
		print(offset[q])

	## Create dataframe of data

	twoman_df = pd.DataFrame(twoman_list)
	twoman_df.columns = ['rank', 'player_1', 'drop1', 'drop2', 'player_2', 'team', 'drop3', 'games',
					'minutes', 'tm_poss', 'opp_poss', 'pace', 'fgm', 'fga', 'fg_perc', 'threep', 'threepa',
					'threepperc', 'efg', 'ftm', 'fta', 'ftperc', 'pts', 'year']
	twoman_df.drop(['drop1', 'drop2', 'drop3'], inplace = True, axis = 1)
	return twoman_df

def dataset_joiner(twoman_data, roster_data, advanced_data):
    '''
    Function that joins data across all sources
    twoman_data : DataFrame that contains two man lineup information
    roster_data : DataFrame that contains roster information
    advanced_data : DataFrame that contains advanced stats information
    '''

    ## Start joining roster & advanced data on player name & year. Team doesn't matter doensn't change per team

    advanced_roster = advanced_data.merge(roster_data, how = 'left', on = ['player', 'year'])
    advanced_twoman = twoman_data.merge(advanced_roster, how = 'left', left_on = ['player_1', 'year', 'team'],
                                        right_on = ['player_join', 'year', 'team'],
                                       suffixes = ('_base', '_left'))
    advanced_twoman = advanced_twoman.merge(advanced_roster, how = 'left', left_on = ['player_2', 'year', 'team'],
                                        right_on = ['player_join', 'year', 'team'],
                                        suffixes = ('_left', '_right'))
    advanced_twoman.drop(['player_join_left', 'player_join_right',
                         'bday_left', 'bday_right', 'rank',
                         'games_left', 'games', 'player_1', 'player_2'], inplace = True, axis = 1)
    advanced_twoman['team_year'] = advanced_twoman['team'] + '_' + advanced_twoman['year'].astype(str)

    advanced_twoman = advanced_twoman.dropna()
    advanced_twoman = advanced_twoman.reset_index(drop=True)


    return advanced_twoman

