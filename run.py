import os
import subprocess
import sys

# This is an example for BBCSport dataset.
dataset_name = 'BBCSport'
file_path = 'datas/' + dataset_name + '/user_guidance_information.pkl'

python_executable = sys.executable

if os.path.exists(file_path):
    print(f"Running PCDI.py...")
    subprocess.run([python_executable, 'PCDI.py'])
else:
    print(f"Running UIP.py first...")
    subprocess.run([python_executable, 'UIP.py'])
    print("Now running PCDI.py...")
    subprocess.run([python_executable, 'PCDI.py'])
