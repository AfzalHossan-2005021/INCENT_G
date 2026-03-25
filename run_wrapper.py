import subprocess
import sys

def run():
    print("Running test_chiral.py")
    result = subprocess.run([sys.executable, "test_chiral.py"], capture_output=True, text=True)
    with open("test_output.log", "w") as f:
        f.write("=== STDOUT ===\n")
        f.write(result.stdout)
        f.write("\n=== STDERR ===\n")
        f.write(result.stderr)
        f.write(f"\nExit Code: {result.returncode}\n")

if __name__ == "__main__":
    run()
