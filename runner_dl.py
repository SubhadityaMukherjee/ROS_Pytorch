import random
import time
import os
import subprocess
import shutil
import itertools
from tqdm import tqdm
# from ray import tune
import pandas as pd

# KNN params

'''
Notes
- DenseNets too huge
- vgg too huge
- mobile net smallest and most accurate
- comment out what you need and do not need

'''

bn = [
"mobileNet",
# "mobileNetV2",
# "xception", # errors

# "nasnetMobile", 
# "inception" 
# "denseNet121",
# "denseNet169",
# "densenet201",
# "inceptionResnet", #errors 56M
# "nasnetLarge", #errors
# "resnet50",
# "vgg16_fc1",
# "vgg16_fc2",
# "vgg19_fc1",
# "vgg19_fc2",
]

dist = [
    # "bhattacharyya",
    "canberra", #!
    # "chiSquared",
    # "cosine",
    "dice",
    # "divergence",
    "euclidean", #!
    # "gower"
    "intersection", #!
    "kLDivergance", #!
    # "manhattan",
    # "motyka",
    # "neyman",
    "pearson", #!
    # "sorensen",
]

pool = ["AVG"]
# pool = [f"'pooling_function':={x}" for x in pool]
pool = [f"'pooling_function':={x}" for x in pool]
obj = [f"'base_network':={x}" for x in bn]
dist = [f"'distance_function':={x}" for x in dist]
perc = [f"'K_for_KNN':={x}" for x in [1]]
# res = ["'orthographic_image_resolution':=299"]

k_f = 'roslaunch rug_kfold_cross_validation kfold_cross_validation_RGBD_deep_learning_descriptor.launch '
# main_path = "/home/robotics26/cognitive_robotics_ws/src/student_ws/rug_kfold_cross_validation/result/experiment_1/"
main_path = "/home/cognitiverobotics/cognitive_robotics_ws/src/student_ws/rug_kfold_cross_validation/result/experiment_1/"
res_file_name = "Deeplearning_results_with_time.csv"
# res_2 = "/home/robotics26/cognitive_robotics_ws/src/student_ws/rug_kfold_cross_validation/result/results_of_COR_2021_DL_experiments.txt"
res_2 = "/home/cognitiverobotics/cognitive_robotics_ws/src/student_ws/rug_kfold_cross_validation/result/results_of_COR_2021_DL_experiments.txt"

all_comb = [k_f+" ".join(x) for x in itertools.product(obj, dist, perc, pool)]
all_comb.sort()
# print(all_comb)
print(f"Total : {len(all_comb)}")


if os.path.isfile(res_file_name):
    df_frame = pd.read_csv(res_file_name)
else:
    df_frame = pd.DataFrame(
        columns=["params", "global_acc", "TP", "FP", "FN", "Pres", "Rec", "F Mea", "Time"])


def tiny_parser(s): return float(str(s).split("=")[-1].strip())


def parse_out(f):
    file_r = open(f+"summary_of_experiment.txt", "r").readlines()
    s_space = file_r[789:821]
    with open(res_2, "r") as f:
        te = f.readlines()[-2].strip().split("\t")
        # print(te)
        network, ins_acc, avg_class_acc = te[1], te[11], te[13]

    try:
        tim = float(s_space[-19].strip().split(" ")[4])
        scores = {
            "network": network, 
            "global_acc": tiny_parser(s_space[-1]),
            "TP": tiny_parser(s_space[-30]),
            "FP": tiny_parser(s_space[-29]),
            "FN": tiny_parser(s_space[-28]),
            "Pres": tiny_parser(s_space[-27]),
            "Rec": tiny_parser(s_space[-26]),
            "F Mea": tiny_parser(s_space[-25]),
            "Ins Acc": ins_acc,
            "Avg Class Acc": avg_class_acc,
            "Time": tim,
        }

        return scores
    except IndexError:
        return {}


def run_comm(obj):
    global df_frame
    global res_file_name
    res = subprocess.run(obj, shell=True, stdout=subprocess.PIPE)
    res = res.stdout
    params = obj.split(" ")[-3:]
    params = [x.split("=")[-1] for x in params]
    scores = parse_out(main_path)
    scores["params"] = " - ".join(params)
    df_frame = df_frame.append(scores, ignore_index=True)
    df_frame.to_csv(res_file_name)



max_s = len(all_comb)
# max_s = 30
# max_s = len(all_comb) -32
# with tqdm() as pbar:
for counti, i in tqdm(enumerate(all_comb), total=max_s):
    current_net = i.split(" ")[3].split("=")[-1]
    # print(current_net)
    # print("")
    os.system(
        f"rosrun rug_deep_feature_extraction multi_view_RGBD_object_representation.py {current_net} &")
    try:
        shutil.rmtree(main_path)
    except FileNotFoundError:
        pass
    time.sleep(22)
    run_comm(i)
    os.system("killall rosrun &")
    if counti == max_s:
        break

os.system("killall roscore && killall rosrun")