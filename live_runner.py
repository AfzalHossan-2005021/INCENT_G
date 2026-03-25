import sys
import subprocess

with open("live_output.log", "w") as f:
    process = subprocess.Popen([sys.executable, "test_chiral.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        f.write(line)
        f.flush()
    process.wait()
