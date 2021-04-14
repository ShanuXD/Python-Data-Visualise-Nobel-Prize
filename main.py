import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import numpy as np
import seaborn as sns
import statsmodels.api

pd.set_option('display.width', 640)
pd.set_option('display.max.columns', 20)
df_data = pd.read_csv('nobel_prize_data.csv')
# print(df_data.shape)
# print(df_data.info())
# print(df_data.describe())

# Nobel prize first awarded
# print(df_data.year.min())

# print(df_data.year.max())

# Check duplicate
# print(df_data.duplicated().values.any())

# Cheack Nan values
# print(df_data.isna().values.any())

# Nan value per column
# print(df_data.isna().sum())

col_subset = ['year','category', 'laureate_type', 'birth_date','full_name', 'organization_name']
# print(df_data.loc[df_data.birth_date.isna()][col_subset])
# print(df_data.loc[df_data.organization_name.isna()][col_subset])

df_data.birth_date = pd.to_datetime(df_data.birth_date)
separated_values = df_data['prize_share'].str.split('/', expand=True)
numerator = pd.to_numeric(separated_values[0])
denomenator = pd.to_numeric(separated_values[1])
df_data['share_pct'] = numerator/denomenator
# print(df_data.info())

gender_df = df_data.sex.value_counts()
# print(gender_df)
# fig = px.pie(labels=gender_df.index,
#              values=gender_df.values,
#              title='Men Vs Woman Winners',
#              names=gender_df.index,
#              hole=.6)
#
# fig.update_traces(textposition='inside', textfont_size=15, textinfo='percent')
# fig.show()

# The first 3 women to win
# print(df_data[df_data.sex == 'Female'].sort_values('year', ascending=True)[:3])

# The first 3 men to win
# print(df_data[df_data.sex == 'Male'].sort_values('year', ascending=True)[:3])

"""The Repeat Winners"""
#  Show all prize won
# print(df_data['full_name'].value_counts())

is_winner = df_data.duplicated(subset=['full_name'], keep=False)
multiple_winners = df_data[is_winner]
# print(f'There are {multiple_winners.full_name.nunique()} winners who were awarded the prize more than once.')

col_subset = ['year', 'category', 'laureate_type', 'full_name']
# print(multiple_winners[col_subset])

# print(df_data['category'].nunique())
prizes_per_category = df_data['category'].value_counts()
# print(prizes_per_category)

# bar = px.bar(x=prizes_per_category.index,
#              y=prizes_per_category.values,
#              color=prizes_per_category.values,
#              color_continuous_scale='Aggrnyl',
#              title="Prize Award per Category")
#
# bar.update_layout(xaxis_title="Nobel Prize Category",
#                   coloraxis_showscale=False,
#                   yaxis_title='Number Of prize')
# bar.show()

# print(df_data[df_data.category == 'Economics'].sort_values('year')[:3])

"""Male and Female Winner by Category"""
cat_men_woman = df_data.groupby(['category', 'sex'], as_index=False).agg({'prize': pd.Series.count})
cat_men_woman.sort_values('prize', ascending=False, inplace=True)
# print(cat_men_woman)

# bar_split = px.bar(x=cat_men_woman.category,
#                    y=cat_men_woman.prize,
#                    color=cat_men_woman.sex,
#                    title='Number Of Prize Awarded per Category')
#
# bar_split.update_layout(xaxis_title='Nobel Prize Category',
#                           yaxis_title='Number of Prizes')
# bar_split.show()

"""Number of Prizes Awarded over Time"""
# print(df_data['year'].value_counts())
prize_per_year = df_data.groupby(by='year').count().prize
# print(prize_per_year)
moving_average = prize_per_year.rolling(window=5).mean()
# print(moving_average)

# plt.figure(figsize=(6, 4), dpi=200)
# plt.title('Number of Nobel Prizes Awarded per Year', fontsize=8)
# plt.yticks(fontsize=7)
# plt.xticks(ticks=np.arange(1900, 2021, step=5),
#            fontsize=7,
#            rotation=45)
#
# ax = plt.gca()  # get current axis
# ax.set_xlim(1900, 2020)
#
# plt.scatter(x=prize_per_year.index,
#             y=prize_per_year.values,
#             c='dodgerblue',
#             alpha=.6,
#             s=50, )
#
# plt.plot(prize_per_year.index,
#          moving_average.values,
#          c='crimson',
#          linewidth=3, )
#
# plt.show()
""" The Prize Share of Laureates over Time """
yearly_avg_share = df_data.groupby(by='year').agg({'share_pct': pd.Series.mean})
# print(yearly_avg_share)
share_moving_average = yearly_avg_share.rolling(window=5).mean()
# print(share_moving_average)

# plt.figure(figsize=(6, 4), dpi=200)
# plt.title('Number of Nobel Prizes Awarded per Year', fontsize=7)
# plt.yticks(fontsize=7)
# plt.xticks(ticks=np.arange(1900, 2021, step=5),
#            fontsize=7,
#            rotation=45)

# ax1 = plt.gca()
# ax2 = ax1.twinx()  # create second y-axis
# ax1.set_xlim(1900, 2020)


# ax1.scatter(x=prize_per_year.index,
#             y=prize_per_year.values,
#             c='dodgerblue',
#             alpha=0.7,
#             s=100, )

# ax1.plot(prize_per_year.index,
#          moving_average.values,
#          c='crimson',
#          linewidth=3, )

# # Adding prize share plot on second axis
# ax2.plot(prize_per_year.index,
#          share_moving_average.values,
#          c='grey',
#          linewidth=3, )
# # Can invert axis
# ax2.invert_yaxis()

# plt.show()

"""Prize ranking by Country"""
birth_country_current = df_data.groupby(['birth_country_current'], as_index=False).agg({'prize': pd.Series.count})
birth_country_current.sort_values(by='prize', inplace=True)

top_20_country = birth_country_current[-20:]
# print(top_20_country)
#
# bar = px.bar(x=top_20_country.prize,
#              y=top_20_country.birth_country_current,
#              color=top_20_country.prize,
#              color_continuous_scale='Viridis',
#              title='Top 20 Countries By Number of Prizes')
#
# bar.update_layout(xaxis_title='Number of Prizes',
#                     yaxis_title='Country')
# bar.show()

"""Displaying the Data on a Map"""
df_countries = df_data.groupby(['birth_country_current', 'ISO'], as_index=False).agg({'prize': pd.Series.count})
df_countries.sort_values(by='prize', inplace=True, ascending=False)
# print(df_countries)

# world_map = px.choropleth(df_countries,
#                           locations='ISO',
#                           color='prize',
#                           hover_name='birth_country_current',
#                           color_continuous_scale=px.colors.sequential.matter)
#
# world_map.update_layout(coloraxis_showscale=True)
#
# world_map.show()

"""Country Bar Chart with Prize Category"""
cat_country = df_data.groupby(['birth_country_current', 'category'], as_index=False).agg({'prize': pd.Series.count})
cat_country.sort_values(by='prize', ascending=False, inplace=True)
# print(cat_country)

# Merging
merged_df = pd.merge(cat_country, top_20_country, on='birth_country_current')
merged_df.columns = ['birth_country_current', 'category', 'cat_prize', 'total_prize']
merged_df.sort_values(by='total_prize', inplace=True)
# print(merged_df)
#
# cat_cntry_bar = px.bar(x=merged_df.cat_prize,
#                        y=merged_df.birth_country_current,
#                        color=merged_df.category,
#                        orientation='h',
#                        title='Top 20 Countries by Number of Prizes and Category')
#
# cat_cntry_bar.update_layout(xaxis_title='Number of Prizes',
#                             yaxis_title='Country')
# cat_cntry_bar.show()

"""Country Prizes over Time"""
prize_by_year = df_data.groupby(by=['birth_country_current', 'year'], as_index=False).count()
# print(prize_by_year)
# Select only the required Columns
prize_by_year = prize_by_year.sort_values('year')[['year', 'birth_country_current', 'prize']]
# print(prize_by_year)

cumulative_prizes = prize_by_year.groupby(by=['birth_country_current', 'year']).sum().groupby(level=[0]).cumsum()
cumulative_prizes.reset_index(inplace=True)
# print(cumulative_prizes)

# chart = px.line(cumulative_prizes,
#                 x='year',
#                 y='prize',
#                 color='birth_country_current',
#                 hover_name='birth_country_current')
#
# chart.update_layout(xaxis_title='Year',
#                     yaxis_title='Number of Prizes')
#
# chart.show()

"""The 20 Top Research Organisations"""
top_20_organization = df_data['organization_name'].value_counts()[:20]
top_20_organization.sort_values(ascending=True, inplace=True)
# print(top_20_organization)

# org_bar = px.bar(x=top_20_organization.values,
#                  y=top_20_organization.index,
#                  orientation='h',
#                  color=top_20_organization.values,
#                  color_continuous_scale=px.colors.sequential.haline,
#                  title='Top 20 Research Institutions by Number of Prizes')
#
# org_bar.update_layout(xaxis_title='Number of Prizes',
#                       yaxis_title='Institution',
#                       coloraxis_showscale=False)
# org_bar.show()

"""Top 20 Research Cities"""
top_20_organization_cities = df_data.organization_city.value_counts()[:20]
top_20_organization_cities.sort_values(ascending=True, inplace=True)
# print(top_20_organization_cities)
# city_bar2 = px.bar(x=top_20_organization_cities.values,
#                    y=top_20_organization_cities.index,
#                    orientation='h',
#                    color=top_20_organization_cities.values,
#                    color_continuous_scale=px.colors.sequential.Plasma,
#                    title='Which Cities Do the Most Research')
#
# city_bar2.update_layout(xaxis_title='Number of Prizes',
#                         yaxis_title='City',
#                         coloraxis_showscale=False)
# city_bar2.show()
#

"""Laureate Birth Cities"""
top_20_cities = df_data.birth_city.value_counts()[:20]
top_20_cities.sort_values(ascending=True, inplace=True)
# print(top_20_cities)
# city_bar = px.bar(x=top_20_cities.values,
#                   y=top_20_cities.index,
#                   orientation='h',
#                   color=top_20_cities.values,
#                   color_continuous_scale=px.colors.sequential.Plasma,
#                   title='Where were the Nobel Laureates Born?')
#
# city_bar.update_layout(xaxis_title='Number of Prizes',
#                        yaxis_title='City of Birth',
#                        coloraxis_showscale=False)
# city_bar.show()

"""SunBurst Chart Country Vs City Vs Organization"""

country_city_organization_df = df_data.groupby(by=['organization_country', 'organization_city', 'organization_name'], as_index=False).agg({'prize': pd.Series.count})
country_city_organization_df.sort_values('prize', ascending=False, inplace=True)
# print(country_city_organization_df)

# burst = px.sunburst(country_city_organization_df,
#                     path=['organization_country', 'organization_city', 'organization_name'],
#                     values='prize',
#                     title='Where do Discoveries Take Place?',
#                     )
#
# burst.update_layout(xaxis_title='Number of Prizes',
#                     yaxis_title='City',
#                     coloraxis_showscale=False)
#
# burst.show()

"""How old are the Nobel laureates at the time when they win the prize"""
birthday_years = df_data['birth_date'].dt.year
df_data['winning_age'] = df_data.year - birthday_years
# print(df_data)

""" Oldest and Youngest Winners"""
# print(df_data[df_data.winning_age == df_data['winning_age'].min()])
# print(df_data[df_data.winning_age == df_data['winning_age'].max()])
# or
# print(df_data.nlargest(n=1, columns='winning_age'))
# print(df_data.nsmallest(n=1, columns='winning_age'))

# print(df_data.winning_age.describe())
#
# plt.figure(figsize=(6, 4), dpi=200)
# sns.histplot(data=df_data,
#              x=df_data.winning_age,
#              bins=50)
#
# plt.xlabel("Age")
# plt.title("Distribution of Age on Receipt of Prize")
# plt.show()

"""Winning Age Over Time (All Categories)"""
# plt.figure(figsize=(6, 4), dpi=200)
# with sns.axes_style('whitegrid'):
#     sns.regplot(data=df_data,
#                 x='year',
#                 y='winning_age',
#                 # lowess=True,
#                 scatter_kws={'alpha': 0.5},
#                 line_kws={'color': 'black'})
#
# plt.show()

"""Age Differences between Categories"""
# plt.figure(figsize=(6, 4), dpi=200)
# with sns.axes_style("whitegrid"):
#     sns.boxplot(data=df_data,
#                 x='category',
#                 y='winning_age')
#
# plt.show()

"""Laureate Age over Time by Category"""
# with sns.axes_style('whitegrid'):
#     sns.lmplot(data=df_data,
#                x='year',
#                y='winning_age',
#                row='category',
#                lowess=True,
#                aspect=2,
#                scatter_kws={'alpha': 0.6},
#                line_kws={'color': 'black'}, )
#
# plt.show()
"""To combine all these charts into the same chart"""
with sns.axes_style("whitegrid"):
    sns.lmplot(data=df_data,
               x='year',
               y='winning_age',
               hue='category',
               lowess=True,
               aspect=2,
               scatter_kws={'alpha': 0.5},
               line_kws={'linewidth': 5})

plt.show()
