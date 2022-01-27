import sys
# setting path
sys.path.append('../reinforcement_learning')
import cv2
import numpy as np
import glob
import os
from gym import spaces
import json
from tensorflow.keras.losses import MSE, MAE
from DDQN_double import  DDQNAgent, linear_unbin_steering, linear_unbin_throttle

img_rows, img_cols = 120, 160


def eval_model(model_path, img_path, json_path):
    img_frames = 4
    state_size = (img_cols, img_rows, img_frames)
    action_space = spaces.Box(
        low=np.array([np.float32(-1), np.float32(0)]),
        high=np.array([np.float32(1), np.float32(1)]),
        dtype=np.float32
    )
    agent = DDQNAgent(state_size, action_space)
    agent.load_model(model_path)
    # agent.model.summary()

    y_pred_steering = []
    y_pred_throttle = []
    y_true_steering = []
    y_true_throttle = []

    obs = cv2.imread(img_path[0], 0)
    file = json.load(open(json_path[0], "r"))

    x_t = agent.process_image(obs)
    a = (x_t,)
    for _ in range(img_frames - 1):
        a = a + (x_t,)
    s_t = np.stack(a, axis=2)
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  # 1*80*80*4

    for img, json_ in zip(img_path[1:], json_path[1:]):

        q_value = agent.model.predict(s_t)

        y_pred_steering.append(linear_unbin_steering(q_value[0][0]))
        y_pred_throttle.append(linear_unbin_throttle(q_value[1][0]))
        y_true_steering.append(file["user/angle"])
        y_true_throttle.append(file["user/throttle"])

        obs = cv2.imread(img, 0)
        file = json.load(open(json_, "r"))

        x_t1 = agent.process_image(obs)
        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)
        s_t = np.append(x_t1, s_t[:, :, :, :img_frames - 1], axis=3)
    
    MSE_stering = MSE(y_true_steering, y_pred_steering)
    MSE_throttle = MSE(y_true_throttle, y_pred_throttle)
    MAE_stering = MAE(y_true_steering, y_pred_steering)
    MAE_throttle = MAE(y_true_throttle, y_pred_throttle)
    print(f"Loss MSE steering= {MSE_stering} - throttle= {MSE_throttle}")
    print(f"Loss MAE steering= {MAE_stering} - throttle= {MAE_throttle}")

    return MSE_stering, MSE_throttle, MAE_stering, MAE_throttle
        
if __name__=="__main__":
    path = "C:/Users/david/Documents/project/gym-donkeycar/examples/reinforcement_learning/data/right_lane_simul_recalibrated/test_set"
    img_list =  glob.glob(os.path.join(path, "*.jpg"))
    json_list = glob.glob(os.path.join(path,"*.json"))

    print(len(img_list), len(json_list))
    model_path = "C:\\Users\\david\\Documents\\project\\gym-donkeycar\\examples\\reinforcement_learning\\models"
    name_model = "model_withacc_ep50k_24gen2021.h5"
    MSE_stering, MSE_throttle, MAE_stering, MAE_throttle = eval_model(os.path.join(model_path, name_model), img_list, json_list)