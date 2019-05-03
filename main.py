import subprocess

cmds = [
        ["mpirun", "-np", "16", "-H", "localhost:16", "--bind-to", "hwthread", "python3", "horovod_train.py", "--num-cores", "16"],
        ["mpirun", "-np", "8", "-H", "localhost:8", "--bind-to", "hwthread", "python3", "horovod_train.py", "--num-cores", "8"],
        ["mpirun", "-np", "4", "-H", "localhost:4", "--bind-to", "hwthread","python3", "horovod_train.py",  "--num-cores", "4"],
        ["mpirun", "-np", "16", "-H", "localhost:16", "--bind-to", "hwthread", "python3", "horovod_train.py", "--num-cores", "16", "--comm-interval", "5"],
        ["mpirun", "-np", "16", "-H", "localhost:16", "--bind-to", "hwthread", "python3", "horovod_train.py", "--num-cores", "16", "--comm-interval", "10"],
        ["mpirun", "-np", "16", "-H", "localhost:16", "--bind-to", "hwthread",  "python3", "hessian_horovod_train.py", "--num-cores", "16"],
        ["mpirun", "-np", "16", "-H", "localhost:16", "--bind-to", "hwthread","python3", "hessian_horovod_train.py",  "--num-cores", "16", "--batch-mult", "2"],
        ["mpirun", "-np", "16", "-H", "localhost:16", "--bind-to", "hwthread", "python3", "hessian_horovod_train.py", "--num-cores", "16", "--batch-mult", "4"],
        ["mpirun", "-np", "16", "-H", "localhost:16", "--bind-to", "hwthread", "python3", "hessian_horovod_train.py", "--num-cores", "16"],
        ["mpirun", "-np", "8", "-H", "localhost:8", "--bind-to", "hwthread", "python3", "hessian_horovod_train.py", "--num-cores", "8"],
        ["mpirun", "-np", "4", "-H", "localhost:4", "--bind-to", "hwthread", "python3", "hessian_horovod_train.py", "--num-cores", "4"],
        ["mpirun", "-np", "16", "-H", "localhost:16", "--bind-to", "hwthread", "python3", "hessian_horovod_train.py", "--num-cores", "16", "--comm-interval", "5"],
        ["mpirun", "-np", "16", "-H", "localhost:16", "--bind-to", "hwthread", "python3", "hessian_horovod_train.py", "--num-cores", "16", "--comm-interval", "10"],
        ["mpirun", "-np", "16", "-H", "localhost:16", "--bind-to", "hwthread", "python3", "hessian_horovod_train.py", "--num-cores", "16", "--eig_comm", "0"],
        ]

for cmd in cmds:
    try:
        subprocess.run(cmd)
    except Exception as e:
        break
