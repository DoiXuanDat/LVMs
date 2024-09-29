import shutil
import subprocess

script_path = "D:/python_code/LVMS/PaliGemma/launch.sh"

bash_path = shutil.which("bash")
if bash_path:
  subprocess.run([bash_path, script_path])
else:
  print("Error: Bash not found. Install Git Bash or a similar tool.")