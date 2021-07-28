import os

def launch_job(cmd, jobname, baseline=None):
    if baseline is not None:
        cmd += ' --lr 5e-5 --tb'
        os.system('CUDA_VISIBLE_DEVICES=5 scripts/LOG_train_il.py' + cmd)

    else:
        cmd += ' --concept_whitening'
        cmd += ' --lr 5e-5 --tb'
        os.system('CUDA_VISIBLE_DEVICES=7 scripts/LOG_train_il.py' + cmd)