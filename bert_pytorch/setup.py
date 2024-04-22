import os
import platform
import subprocess

def main():
    # Check the operating system
    os_name = platform.system()

    if os_name == "Windows":
        # Run the batch script
        subprocess.run(["cmd", "/c", "setup.bat"])
    else:
        # Run the shell script
        subprocess.run(["sh", "setup.sh"])

if __name__ == "__main__":
    main()