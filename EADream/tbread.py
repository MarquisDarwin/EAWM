from tbparse import SummaryReader
import pandas as pd
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils
import matplotlib.pyplot as plt
import numpy as np
import json
hasseed=True
log_dir = "./lognoatt"
randhumandir="randhuman.txt"
randhumandf = pd.read_csv(randhumandir,sep="\t") 
randhumandf['score']=0.0
# print(randhumandf)  #TODO:{'name':{'seed':[]}} #26 'exp'+ mean, median, IQM, optimality gap list for each_step =32500

randhumanscore=randhumandf[['Random','Human']].values.tolist()
task_index_dict=dict(zip(randhumandf['task'].values.tolist(),randhumanscore))
reader = SummaryReader(log_dir, extra_columns={'dir_name'})
df = reader.scalars
logic1=df['step']==400000
logic2=df['tag']=="scalars/eval_return"
df=df[logic1&logic2]
df['dir_name'] = df['dir_name'].apply(lambda task: ''.join([x.capitalize() for x in task.split('_')[1:]]))

df=df.sort_values(by=['dir_name'],ascending=[True])
print(df)
task_hns=[]
def calc_hns(randomhuman:list,value):
  return round((value-randomhuman[0])/(randomhuman[1]-randomhuman[0]),4)
randhumandf_score_col_index =randhumandf.columns.get_loc('score')
seedid={}
randhumandf_dict={}
sumdf=randhumandf.copy()
score_dict={}
for _,row in df.iterrows():
  task_name,value=row['dir_name'],row['value']
  
  if hasseed:
    task_name,seed=task_name.split('-')
    seed=int(seed)
  else:
    seed=0
  score_dict.setdefault(task_name,[]).append(value)
  if seed not in seedid.keys():
    seedid[seed]=len(seedid)
    task_hns.append([])
    randhumandf_dict[seed]=randhumandf.copy()
  dftask_nameindex=randhumandf_dict[seed][randhumandf_dict[seed]['task']==task_name].index.tolist()[0]  
  randhumandf_dict[seed].iloc[dftask_nameindex,randhumandf_score_col_index]=value
  sumdf.iloc[dftask_nameindex,randhumandf_score_col_index]+=value
  task_hns[seedid[seed]].append(calc_hns(task_index_dict[task_name],value))
with open("./task_scores.json","w") as f:
  json.dump(score_dict,f,indent=4)
for _,row in sumdf.iterrows():
  run_num=len(df[df['dir_name']==row['task']])
  if run_num==0:
    row['score']=row['score']
  else:
    row['score']=row['score']/run_num
print(log_dir)
pd.set_option('display.float_format',  '{:<10.2f}'.format)
for seed, randhumandf in randhumandf_dict.items():
  print(f'seed:{seed}')
  print(randhumandf['score'].to_string(index=False))
if len(randhumandf_dict)>1:
  print("tot")
  print(sumdf['score'].to_string(index=False))
task_lens=set()
for t in task_hns:
  task_lens.add(len(t))
max_task_len=max(task_lens)
res_seed=[]
res_task=[]
for i,t in enumerate(task_hns):
  if len(t)==max_task_len:
    for key, val in seedid.items():
      if val == i:
        res_seed.append(key)
        break 
    res_task.append(t)
atari_200m_normalized_score_dict = {'mydream':np.array(res_task)}
print("for seed:")
print(res_seed)
aggregate_func = lambda x: np.array([
  metrics.aggregate_mean(x),
  metrics.aggregate_median(x),
  metrics.aggregate_iqm(x),
  metrics.aggregate_optimality_gap(x)])
aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
  atari_200m_normalized_score_dict, aggregate_func, reps=50000)
print(aggregate_scores)
fig, axes = plot_utils.plot_interval_estimates(
  aggregate_scores, aggregate_score_cis,
  metric_names=['Median', 'IQM', 'Mean', 'Optimality Gap'],
  algorithms=['mydream'], xlabel='Human Normalized Score')
plt.savefig('rliable.eps',bbox_inches='tight')
plt.savefig('rliable.pdf',bbox_inches='tight')