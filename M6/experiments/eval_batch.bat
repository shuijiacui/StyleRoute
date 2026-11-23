@echo off
cd /d C:\Users\zzx11\Documents\kimi\workspace\StyleRoute
".venv\Scripts\python.exe" -m M6.experiments.eval_batch --budget-seconds 200 >> M6\outputs\eval_batch.log 2>&1
