from mmap import ALLOCATIONGRANULARITY
import os
import subprocess
import shutil
import itertools
from tqdm import tqdm
# from ray import tune
import pandas as pd

# KNN params

obj = [
    "GOOD",
    "ESF",
    # "VFH",
    # "GRSD"
]
dist = ["chiSquared",
# "kLDivergance",
"motyka",
"divergence",
"euclidean",
# "intersection",
"manhattan",
# "cosine",
"dice",
"bhattacharyya",
"sorensen",
"canberra",
"pearson",
"neyman",
"gower"]

obj = [f"'object_descriptor':={x}" for x in obj]
dist = [f"'distance_function':={x}" for x in dist]
perc = [f"'K_for_KNN':={x}" for x in range(1,9,2)]

k_f = 'roslaunch rug_kfold_cross_validation kfold_cross_validation_hand_crafted_descriptor.launch '
main_path = "/home/robotics26/cognitive_robotics_ws/src/student_ws/rug_kfold_cross_validation/result/experiment_1/"
res_file_name = "KNNHand_results_with_time.csv"
res_2 = "/home/robotics26/cognitive_robotics_ws/src/student_ws/rug_kfold_cross_validation/result/results_of_COR_2021_HC_experiments.txt"

all_comb = [k_f+" ".join(x) for x in itertools.product(obj, dist, perc)]
print(f"Total : {len(all_comb)}")


if os.path.isfile(res_file_name):
    df_frame = pd.read_csv(res_file_name)
else:
    df_frame = pd.DataFrame(columns=["params", "global_acc", "TP", "FP", "FN", "Pres", "Rec", "F Mea", "Time"])

def tiny_parser(s): return float(str(s).split("=")[-1].strip())

def parse_out(f):
    file_r = open(f+"summary_of_experiment.txt", "r").readlines()
    s_space = file_r[789:821]
    with open(res_2, "r") as f:
        te = f.readlines()[-2].strip().split("\t")
        bins, ins_acc , avg_class_acc = te[3], te[7], te[8]  
        # print(te)

    try:
        tim = float(s_space[-19].strip().split(" ")[4])
        scores = {\
        "global_acc": tiny_parser(s_space[-1]),
        "TP": tiny_parser(s_space[-30]),
        "FP": tiny_parser(s_space[-29]),
        "FN": tiny_parser(s_space[-28]),
        "Pres": tiny_parser(s_space[-27]),
        "Rec": tiny_parser(s_space[-26]),
        "F Mea": tiny_parser(s_space[-25]),
        "Bins" : bins,
        "Ins Acc" : ins_acc,
        "Avg Class Acc": avg_class_acc,
        "Time": tim,
        }
    
        return scores
    except IndexError:
        return {}



def run_comm(obj):
    global df_frame
    global res_file_name
    res = subprocess.run(obj,shell = True, stdout=subprocess.PIPE)
    res = res.stdout
    params = obj.split(" ")[-3:]
    params = [x.split("=")[-1] for x in params]
    scores = parse_out(main_path)
    scores["params"] = " - ".join(params)
    df_frame = df_frame.append(scores, ignore_index = True)
    df_frame.to_csv(res_file_name)

import random
random.shuffle(all_comb)

max_s = len(all_comb)
# max_s = 20
# max_s = len(all_comb) -32

# with tqdm() as pbar:
for counti,i in tqdm(enumerate(all_comb), total=max_s):
    try:
        shutil.rmtree(main_path)
    except FileNotFoundError:
        pass
    run_comm(i)
    # if counti >=max_s:
    #     break
