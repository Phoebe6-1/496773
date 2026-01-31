import pandas as pd 

datasets = pd.read_csv("estimated_votes.csv")

datasets['fan_rank'] = datasets.groupby(['season','week'])['estimated_fan_vote_share'].rank(ascending=False, method='min').astype(int)
datasets['total_percent'] = datasets['judge_percent'] + datasets['estimated_fan_vote_share']
datasets['total_rank'] = datasets['judge_rank'] + datasets['fan_rank']

datasets.to_csv("estimated_votes.csv",index = False)

