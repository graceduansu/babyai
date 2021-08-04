import os

def launch_job(cmd, jobname, baseline=True):
    if baseline:
        cmd += ' --lr 5e-5 --tb --patience 0'
        os.system('CUDA_VISIBLE_DEVICES=7 scripts/LOG_train_il.py' + cmd)

    else:
        cmd += ' --concept_whitening'
        cmd += ' --lr 5e-5 --tb --patience 0'
        os.system('CUDA_VISIBLE_DEVICES=7 scripts/LOG_train_il.py' + cmd)