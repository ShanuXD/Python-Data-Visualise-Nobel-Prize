import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import numpy as np

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

plt.figure(figsize=(6, 4), dpi=200)
plt.title('Number of Nobel Prizes Awarded per Year', fontsize=7)
plt.yticks(fontsize=7)
plt.xticks(ticks=np.arange(1900, 2021, step=5),
           fontsize=7,
           rotation=45)

ax1 = plt.gca()
ax2 = ax1.twinx()  # create second y-axis
ax1.set_xlim(1900, 2020)


ax1.scatter(x=prize_per_year.index,
            y=prize_per_year.values,
            c='dodgerblue',
            alpha=0.7,
            s=100, )

ax1.plot(prize_per_year.index,
         moving_average.values,
         c='crimson',
         linewidth=3, )

# Adding prize share plot on second axis
ax2.plot(prize_per_year.index,
         share_moving_average.values,
         c='grey',
         linewidth=3, )
# Can invert axis
ax2.invert_yaxis()

plt.show()