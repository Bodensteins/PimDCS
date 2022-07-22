import subprocess


def run_cmd(command):
    ret = subprocess.run(command, shell=True)
    if ret.returncode != 0:
        print("run command: " + command + ", return error: " + ret.returncode)


run_cmd("ls -l > test.out 2> test.out")
