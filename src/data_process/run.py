import subprocess

# cmd = 'python3 user_log_train.py'
# subprocess.call(cmd, shell=True)
#
# cmd = 'python3 user_log_test.py'
# subprocess.call(cmd, shell=True)

cmd = 'python3 user_log_train_v2.py'
subprocess.call(cmd, shell=True)

cmd = 'python3 train_test.py'
subprocess.call(cmd, shell=True)