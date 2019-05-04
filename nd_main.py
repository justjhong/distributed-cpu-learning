import subprocess

cmds = [
        # ["python3", "nd_horovod_train.py", "--batch-size", "128"],
        ["python3", "nd_hessian_horovod_train.py", "--init-batch-size", "128"],
        ]

for cmd in cmds:
    try:
        subprocess.run(cmd)
    except Exception as e:
        break
