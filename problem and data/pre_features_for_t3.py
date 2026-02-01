import pandas as pd 

df1 = pd.read_csv("2026_MCM_Problem_C_Data.csv")
df2 = pd.read_csv("estimated_votes.csv")

avg_votes = df2.groupby(["season","celebrity_name"])["fan_vote_percent"].mean().reset_index()
avg_votes['fan_vote_percent'] = round(avg_votes['fan_vote_percent'],3)
avg_votes.rename(columns = {'fan_vote_percent':'fan_share_season'},inplace = True)
result_df = pd.merge(df1,avg_votes,on = ["season","celebrity_name"],how = 'left')

result_df.to_csv("t3_data.csv",index = False)