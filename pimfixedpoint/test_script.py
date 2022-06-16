import subprocess


def run_cmd(command):
    ret = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         encoding="utf-8", timeout=1)
    if ret.returncode == 0:
        print("success:", ret)
    else:
        print("error:", ret)

