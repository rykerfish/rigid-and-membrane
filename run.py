import membrane_explicit

import time
from time import sleep

k_bends = [500.0, 1000.0, 2500.0, 5000.0, 7500.0, 10000.0]
# k_bends = [2500.0, 5000.0, 7500.0, 10000.0]

for k_bend in k_bends:
    print(f"Running simulation with k_bend = {k_bend}")
    while True:
        try:
            membrane_explicit.run(k_bend=k_bend)
            break  # exit the loop if successful
        except Exception as e:
            print(f"Error during run with k_bend = {k_bend}: {e}. Retrying...")
            sleep(5)
    print(f"Simulation with k_bend = {k_bend} completed.\n")
