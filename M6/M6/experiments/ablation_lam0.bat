@echo off
cd /d C:\Users\zzx11\Documents\kimi\workspace\StyleRoute
.venv\Scripts\python.exe -m M6.experiments.ablation_batch --config M6/configs/experiment_lam0.json --out-root M6/outputs_ablation/lam0 --methods risk plain --budget-seconds 200 >> M6/outputs_ablation\lam0_batch.log 2>&1
