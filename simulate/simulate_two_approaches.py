import pandas as pd 
import numpy as np 

df = pd.read_csv("estimated_votes.csv")

# handle combined approach based on percent
percent_min_idx = df.groupby(["season","week"])["total_percent"].transform("idxmin")
df['is_eliminated_this_week_percent'] = (df.index == percent_min_idx).astype(int)

rank_max_idx = df.groupby(["season","week"])["total_rank"].transform("idxmax")
df['is_eliminated_this_week_rank'] = (df.index == rank_max_idx).astype(int)

def check_mismatch(group):
    idx1 = group.index[group['is_eliminated_this_week_rank'] == 1].tolist()
    idx2 = group.index[group['is_eliminated_this_week_percent'] == 1].tolist()
    return idx1 != idx2

# mismatch_group = df.groupby(["season","week"]).apply(check_mismatch)  # a Series whose index is ["season","week"]
# res_df = mismatch_group[mismatch_group].index.to_frame(index = False)

# # df.to_csv("two_approach_eliminated_res.csv",index = False)
# # res_tab.to_csv("mismatch_week.csv",index = False)

# mismatched_all_data = pd.merge(df, res_df, on=['season', 'week'], how='inner')
df.to_csv("all_data.csv",index = False)