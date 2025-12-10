import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import time
import importlib
import numpy as np
from utils.dataloading import import_ts_data_unsupervised
from metrics import ts_metrics, point_adjustment, ts_metrics_enhanced

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
dataset_root = './data/'

parser = argparse.ArgumentParser()
parser.add_argument("--runs", type=int, default=5)
parser.add_argument("--output_dir", type=str, default='@records/')
parser.add_argument("--dataset", type=str, default='Epilepsy')
parser.add_argument("--entities", type=str, default='FULL')
parser.add_argument("--entity_combined", type=int, default=1)
parser.add_argument("--net", type=str, default="GRU",
                    choices=["GRU","LSTM","Transformer"],
                    help="Select STEN backbone network [GRU/LSTM/Transformer]")
parser.add_argument("--model", type=str, default='STEN', help="Main STEN model class")

parser.add_argument('--silent_header', action='store_true')
parser.add_argument("--flag", type=str, default='')
parser.add_argument("--note", type=str, default='')
parser.add_argument('--seq_len', type=int, default=10)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.00001)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--beta', type=float, default=1)
parser.add_argument('--delta', type=float, default=0.6)

args = parser.parse_args()

# ---------------- Dynamic Load Selected STEN Network ---------------- #
network_class_map = {
    "GRU": "STEN_GRU",
    "LSTM": "STEN_LSTM",
    "Transformer": "STEN_Transformer"
}
module_name = f"models.{network_class_map[args.net]}"
module = importlib.import_module(module_name)

# 注意：每個 STEN_XXX class 都需要命名為 STEN
model_class = getattr(module, "STEN")

print(f"\n==============================")
print(f"▶ Running Model: {args.model} | Network: {args.net}")
print(f"==============================\n")

# ---------------- Load Config File ---------------- #
path = 'configs.yaml'
with open(path) as f:
    d = yaml.safe_load(f)
    try:
        model_configs = d[args.model]
    except KeyError:
        print(f'config.yaml has no entry for {args.model}, using defaults.')
        model_configs = {}

# -------------- Inject Args into Config -------------- #
model_configs.update({
    'seq_len': args.seq_len,
    'stride': args.stride,
    'alpha': args.alpha,
    'beta': args.beta,
    'lr': args.lr,
    'batch_size': args.batch_size,
    'epoch': args.epoch,
    'hidden_dim': args.hidden_dim,
})

print(f"Model Configs: {model_configs}")

cur_time = time.strftime("%m-%d %H.%M.%S", time.localtime())
os.makedirs(args.output_dir, exist_ok=True)
result_file = os.path.join(args.output_dir, f'{args.model}.{args.net}.{args.flag}.csv')

# ---------------- Write Header ---------------- #
if not args.silent_header:
    with open(result_file,'a') as f:
        print('\n---------------------------------------------------------', file=f)
        print(f'Model: {args.model} | Network: {args.net} | Dataset: {args.dataset}', file=f)
        print(f'Runs: {args.runs} | Time: {cur_time}', file=f)
        for k,v in model_configs.items():
            print(f'Param,[{k}]={v}', file=f)
        print(f'Note: {args.note}', file=f)
        print('---------------------------------------------------------', file=f)

dataset_name_lst = args.dataset.split(',')

for dataset in dataset_name_lst:

    entity_metric_lst = []
    entity_metric_std_lst = []

    train_lst, test_lst, label_lst, name_lst = import_ts_data_unsupervised(
        dataset_root, dataset,
        entities=args.entities,
        combine=args.entity_combined
    )

    for train_data, test_data, labels, dataset_name in zip(train_lst,test_lst,label_lst,name_lst):

        entries = []
        t_lst = []

        for i in range(args.runs):
            print(f"\n===== Run {i+1}/{args.runs} | {args.model}-{args.net} "
                  f"on [{dataset_name}] =====")

            # measure time around fit + scoring
            start_t = time.time()
            clf = model_class(**model_configs, random_state=42+i)
            clf.fit(train_data)
            scores = clf.decision_function(test_data)
            end_t = time.time()

            run_time = end_t - start_t
            t_lst.append(run_time)   # <- push runtime so we can average later

            eval_metrics = ts_metrics(labels, scores)
            adj = ts_metrics(labels, point_adjustment(labels, scores))

            # -------- thresholding -------- #
            thresh = np.percentile(scores, 100 - args.delta)
            pred = (scores > thresh).astype(int)

            print(f"[{args.model}-{args.net}] Threshold = {thresh:.4f}")
            # optional: print per-run time
            print(f"Run time: {run_time:.4f} sec")

            txt = (f'{dataset_name}, '
                   + ', '.join(['%.4f' % a for a in eval_metrics])
                   + ', PA, '
                   + ', '.join(['%.4f' % a for a in adj])
                   + f' , [Network={args.net}]\n')
            print(txt)

            enriched = ts_metrics_enhanced(labels, point_adjustment(labels,scores), pred)
            entries.append(enriched)

        # after runs
        avg = np.mean(entries, axis=0)
        std = np.std(entries, axis=0)
        entity_metric_lst.append(avg)
        entity_metric_std_lst.append(std)

        # average runtime (seconds)
        avg_time = float(np.mean(t_lst)) if len(t_lst) > 0 else 0.0

        with open(result_file,'a') as f:
            # write header once (you already do this earlier; keep it if you want dataset-level header)
            print('data,auroc,std,aupr,std,best_f1,std,best_p,std,best_r,std,aff_p,std,aff_r,std,'
                  'vus_r_auroc,std,vus_r_aupr,std,vus_roc,std,vus_pr,std,time,Network', file=f)

            # build log — include avg_time before Network to match header
            log = ('%s, %.4f,%.4f,%.4f,%.4f,%.4f,%.4f, %.4f,%.4f,%.4f,%.4f,%.4f,%.4f,'
                   '%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f, %.4f, %s') % (
                dataset_name,
                avg[0], std[0], avg[1], std[1], avg[2], std[2], avg[3], std[3],
                avg[4], std[4], avg[5], std[5], avg[6], std[6], avg[7], std[7],
                avg[8], std[8], avg[9], std[9], avg[10], std[10],
                avg_time,            # <-- 加上 time（numeric）
                args.net
            )

            print(log)
            print(log, file=f)
