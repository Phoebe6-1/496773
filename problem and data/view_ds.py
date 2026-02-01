import pandas as pd 

datasets = pd.read_csv("2026_MCM_Problem_C_Data.csv")

# datasets['fan_vote_percent'] = round(datasets['estimated_fan_vote_share'],3)
# datasets['fan_vote_rank'] = datasets.groupby(['season','week'])['estimated_fan_vote_share'].rank(ascending=False, method='min').astype(int)
# datasets['total_percent'] = round(datasets['judge_percent'] + datasets['fan_vote_percent'],3)
# datasets['total_rank'] = datasets['judge_rank'] + datasets['fan_vote_rank']

# sel_columns = ['season','week','celebrity_name','judge_percent','judge_rank',
#               'fan_vote_percent','fan_vote_rank','total_percent','total_rank']
# df = datasets[sel_columns]

# df.to_csv("estimated_votes.csv",index = False)

# dancers = datasets['ballroom_partner'].value_counts()
# print(datasets['ballroom_partner'].nunique())
# print(dancers)

industry = datasets['celebrity_industry'].value_counts()
print(datasets['celebrity_industry'].nunique())
print(industry)

country = datasets['results'].value_counts()
print(country)