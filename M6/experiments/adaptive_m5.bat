@echo off
cd /d C:\Users\zzx11\Documents\kimi\workspace\StyleRoute
.venv\Scripts\python.exe -m M6.experiments.ablation_batch --config M6/configs/experiment.json --out-root M6/outputs_adaptive/m5 --team-config M4/resources/configs/team.json --methods risk plain --budget-seconds 200 >> M6/outputs_adaptive\m5_batch.log 2>&1
