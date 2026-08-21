import json, time
from biohub_tracking.eval.cv import evaluate_cv, cv_result_to_dict
t0=time.time()
champ = json.load(open('champion/config.json'))
cand = json.load(open('champion/candidates/sot2864-motion-model-link.json'))
out={}
for name,cfg in [('champion_ondisk',champ),('gain1_sot2864',cand)]:
    r=evaluate_cv(config=cfg)
    d=cv_result_to_dict(r)
    out[name]={'micro_adj':d.get('micro_adj_edge_jaccard') or d.get('score'),'score':d.get('score'),
               'per_dataset':[{'k':x.get('dataset') or x.get('name'),'adj':x.get('micro_adj_edge_jaccard') or x.get('adj')} for x in d.get('per_dataset',[])]}
    print(name, out[name]['micro_adj'], flush=True)
out['elapsed_s']=round(time.time()-t0,1)
json.dump(out, open('experiments/sot2909_promote_cv.json','w'), indent=2)
print('DONE', out['elapsed_s'])
