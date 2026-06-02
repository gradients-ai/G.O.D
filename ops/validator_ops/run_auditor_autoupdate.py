import argparse
import os
import subprocess
import time


def should_update_local(local_commit: str, remote_commit: str) -> bool:
    return local_commit != remote_commit


def run_auto_updater():
    while not os.path.exists(".vali.env"):
        time.sleep(10)

    launch_command = "task auditor"
    os.system(launch_command)
    time.sleep(60)

    while True:
        current_branch = subprocess.getoutput("git rev-parse --abbrev-ref HEAD")
        local_commit = subprocess.getoutput("git rev-parse HEAD")
        os.system("git fetch")
        remote_commit = subprocess.getoutput(f"git rev-parse origin/{current_branch}")

        if should_update_local(local_commit, remote_commit):
            reset_cmd = f"git reset --hard {remote_commit}"
            process = subprocess.Popen(reset_cmd.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
            if not error:
                os.system("./ops/validator_ops/autoupdate_auditor_steps.sh")
                time.sleep(20)

        time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run auto updates for an auditor")
    args = parser.parse_args()
    run_auto_updater()
