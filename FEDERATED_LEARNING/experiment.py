import os
import argparse
import subprocess
from multiprocessing import Process
from model_init import model_initialization, generate_null_models
from config import RESULT_PATH, DATA_SOURCE

def run_script(data_source, lv, fl_round, device):
    subprocess.run([
        "python3", "local_fit.py", 
        "--dat_src", data_source,
        "--lv", str(lv),
        "--fl_round", str(fl_round),
        "--device", str(device)
    ])

def main(fl_end_round, lv):
    fl_rounds = range(1, fl_end_round)
    
    # Initialization
    if not os.path.exists(RESULT_PATH + f'lv{lv}/round0/shared_avg_layers.pth'):
        os.makedirs(RESULT_PATH + f'lv{lv}/round0/', exist_ok=True)
        generate_null_models(lv)
        model_initialization(lv, 0)

    for fl_round in fl_rounds:
        round_rst_path = RESULT_PATH + f'lv{lv}/round{fl_round}/'
        os.makedirs(round_rst_path, exist_ok=True)
        round_rst = round_rst_path + f'shared_avg_layers.pth'
        
        if os.path.exists(round_rst):
            print(f'Lv.{lv}: Round {fl_round} has already been completed.')
            continue
        else:
            # Parallelism start
            processes = [
                Process(target=run_script, args=(
                    dat_src, lv, fl_round, ((lv - 1) + DATA_SOURCE.index(dat_src)) % 4
                )) for dat_src in DATA_SOURCE
            ]
            
            for process in processes:
                process.start()

            for process in processes:
                process.join()
            # Parallelism end

            print(f'Round {fl_round}: All the local set was trained.')

            model_initialization(lv, fl_round)
            print(f'Lv.{lv}: Round {fl_round+1} has been initialized.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run federated learning training script")
    parser.add_argument("--fl_end_round", type=int, required=True, help="The final round number for FL training")
    parser.add_argument("--lv", type=int, choices=[1, 2, 3, 4], required=True, help="Level parameter for model initialization and training")

    args = parser.parse_args()
    main(args.fl_end_round, args.lv)