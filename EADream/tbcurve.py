from tbparse import SummaryReader
import pandas as pd
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils
import matplotlib.pyplot as plt
import numpy as np
import json
log_dir = "./logdmc"
randhumandir="randhuman.txt"
randhumandf = pd.read_csv(randhumandir,sep="\t") 
# print(randhumandf)  #TODO:{'name':{'seed':[]}} #26 'exp'+ mean, median, IQM, optimality gap list for each_step =32500
hns=False
randhumanscore=randhumandf[['Random','Human']].values.tolist()
task_index_dict=dict(zip(randhumandf['task'].values.tolist(),randhumanscore))
reader = SummaryReader(log_dir, extra_columns={'dir_name'})
df = reader.scalars
#logic1=df['step']==400000
logic2=df['tag']=="scalars/eval_return"
logicatari=df['dir_name'].str.contains('atari', regex=True)
logicdmc=df['dir_name'].str.contains('dmc', regex=True)
atari_df=df[logicatari&logic2]
atari_df['dir_name'] = atari_df['dir_name'].apply(lambda task: ' '.join([x.capitalize() for x in task.split('_')[1:]]))
dmc_df=df[logicdmc&logic2]
print(dmc_df)
task_hns=[]
atari_log_interval=32500
atari_start=10000
dmc_start=5000
dmc_log_interval=49750
def calc_hns(randomhuman:list,value):
  return round((value-randomhuman[0])/(randomhuman[1]-randomhuman[0]),4)
seed_num=5
def dfcurve(df,log_interval,first_log_step,atari=True):
  
  df_curve={}
  df=df.sort_values(by=['dir_name','step'],ascending=[True, True])
  last_task=None
  last_value=0
  last_seed=-1
  for _,row in df.iterrows():
    task_name,step,value=row['dir_name'],row['step'],row['value']

    task_name,seed=task_name.split('-')
    seed=int(seed)
    step=int(step)
    if last_task!=task_name or last_seed!=seed:
      last_task=task_name
      last_seed=seed
      last_step=step
      last_value=value
      if seed not in df_curve.setdefault(task_name,{}):
        df_curve[task_name][seed]=[value]
    else:
      n_step=(step-last_step)//log_interval
      if n_step==0:
        print(f"Warning:n_step=0,step={step},task_name={task_name},seed={seed}")
        if last_value<value:

          last_value=value
          df_curve[task_name][seed][-1]=value
      else:  
        diff=(value-last_value)/n_step
      for i in range(1,n_step+1):
        df_curve[task_name][seed].append(last_value+diff*i)
      if n_step!=0:
        last_step+=n_step*log_interval
        last_value+=diff*n_step
  if atari:

    task_num=len(df_curve)
    max_step=(400000-first_log_step)//log_interval+1
    for step in range(max_step):
      seedruns=np.zeros((seed_num,task_num))
      i=0
      for task_name,res_dict in df_curve.items():
        print(f"task_name:{task_name},res_dict:{res_dict}")
        if isinstance(res_dict,dict)==False:
          continue
        for s,(seed,runs) in enumerate(res_dict.items()):
          t=task_name
          t=t.replace(" ","")
          seedruns[s][i]=calc_hns(task_index_dict[t],runs[step])
        i+=1
      df_curve.setdefault('Mean',[]).append(metrics.aggregate_mean(seedruns))
      df_curve.setdefault('Median',[]).append(metrics.aggregate_median(seedruns))
      df_curve.setdefault('IQM',[]).append(metrics.aggregate_iqm(seedruns))
      df_curve.setdefault('Optimality Gap',[]).append(metrics.aggregate_optimality_gap(seedruns))
  return df_curve

#with open("./atari_curve.json","w") as f:
#atari_curve=dfcurve(atari_df,atari_log_interval,atari_start,True)
# json.dump(atari_curve,f,indent=4)
with open("./dmc_curve.json","w") as f:
   dmc_curve=dfcurve(dmc_df,dmc_log_interval,dmc_start,False)
   json.dump(dmc_curve,f,indent=4)
