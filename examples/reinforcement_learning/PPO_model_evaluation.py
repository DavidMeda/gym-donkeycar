import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn import L1Loss, MSELoss
from torchvision import transforms
from PIL import Image
import numpy as np
import glob
import json
from sklearn.model_selection import train_test_split
import shutil
import matplotlib.pyplot as plt
from stable_baselines3.ppo import PPO
import warnings
from autoencoder import load_ae
from tqdm import tqdm
warnings.filterwarnings('ignore')


def create_testset(path):

    paths = glob.glob(os.path.join(path, "*.json"))
    img_name = []
    steering_values = []
    throttle_values = []
    for p in paths:
        json_file = json.load(open(p, "r"))
        img_name.append(json_file["cam/image_array"])
        steering_values.append(json_file["user/angle"])
        throttle_values.append(json_file["user/throttle"])

    dt = pd.DataFrame()
    dt['user/angle'] = steering_values
    dt['img_name'] = img_name
    # dt["user/throttle"] = throttle_values
    bins_index = pd.cut(dt['user/angle'], 20, labels=False)
    record_train, record_test, label_train, label_test = train_test_split(
        np.asarray(dt['img_name']), np.asarray(dt['user/angle']),
        stratify=bins_index, test_size=(1. - 0.9), random_state=444)
    os.makedirs(os.path.join(path, "test_set"), exist_ok=True)

    for f in record_test:
        num_file = f.split("_")[0]
        # copy image
        shutil.copy(os.path.join(path, f), os.path.join(path, "test_set"))
        #copy json
        shutil.copy(os.path.join(path, "record_" + num_file + ".json"), os.path.join(path, "test_set"))
    test_set_path = os.path.join(path, "test_set")
    print(len(glob.glob(os.path.join(test_set_path, "*.json"))))
    return record_train, record_test, label_train, label_test

class CustomImageDataset(Dataset):
    def __init__(self, data_dir, labels_name_arr):
        self.data_dir = data_dir
        self.label_list = labels_name_arr
        self.convert_tensor = transforms.ToTensor()
    

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        label_json = json.load(open(self.label_list[idx], "r"))
        img_name = label_json["cam/image_array"]
        image = Image.open(os.path.join(self.data_dir, img_name))
        if encoder is not None:
            image = np.asarray(image)
            image = encoder.encode_from_raw_image(image).flatten()
        else:
            image = self.convert_tensor(image).permute(1, 2, 0)
        label = torch.tensor([label_json["user/angle"], label_json["user/throttle"]], dtype=torch.float32)
        return image, label

def myLoss(output, target):
    return (1 + torch.exp(torch.abs(target))) * torch.abs(target - output)

def eval_model(model, test_set, loss_func1, loss_func2, encoder=None,):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("Torch device avariable:", device)
    loss1_steering = []
    loss1_throttle = []
    loss2_steering = []
    loss2_throttle = []

    with torch.no_grad():
        # model.eval()
        pbar = tqdm(total=len(test_set))
        for i, (x_test, y_test) in enumerate(test_set):
            y_pred = model.predict(x_test)
            # model.predict return tuple (array[steering, throttle], None)
            y_pred = torch.tensor(y_pred[0])
            # print(f"y_pred={y_pred}")
            # print(f"y_true={y_test}", "\n")
            # print(f"Iter {i}: y_pred= {y_pred} - y_true= {y_test}")
            
            loss1_steering.append(loss_func1(y_pred[0], y_test[0]))
            loss1_throttle.append(loss_func1(y_pred[1], y_test[1]))
            loss2_steering.append(loss_func2(y_pred[0], y_test[0]))
            loss2_throttle.append(loss_func2(y_pred[1], y_test[1]))
            # print(f"Loss1 (MSE) result= {loss1_steering[0]} - {loss1_throttle[0]}")
            # print(f"Loss2 (MAE) result= {loss2_steering[0]} - {loss2_throttle[0]}")
            pbar.update(1)
        pbar.close()
    # print(len(test_set), len(loss1_throttle))
    return np.mean(loss1_steering), np.mean(loss1_throttle), np.mean(loss2_steering), np.mean(loss2_throttle)



if __name__ == "__main__":
    # path = "C:/Users/david/Documents/project/gym-donkeycar/examples/reinforcement_learning/data/right_lane_simul_recalibrated/test_set"
    # path = "C:\\Users\\david\\Documents\\project\\gym-donkeycar\\examples\\reinforcement_learning\\data\\right_lane_simul\\test_set"
    path = "C:\\Users\\david\\Documents\\project\\gym-donkeycar\\examples\\reinforcement_learning\\data\\right_lane_road_gen_test_set"
    label_list = glob.glob(os.path.join(path, "*.json"))
    print(len(label_list))
    # record_train, record_test, label_train, label_test = create_testset(path)
    # print("train len:", len(record_train), len(label_train))
    # print("test len:", len(record_test), len(label_test))

    dataset = CustomImageDataset(path, label_list)
    #test_set = DataLoader(dataset, batch_size=64, shuffle=True)

    # test_img, train_labels = next(iter(dataset))
    # print(f"Feature batch shape: {test_img.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")
    # for i in range (10):
    #     img = test_img[i].permute(1, 2, 0).squeeze()
    #     label = train_labels[i]
    #     print(f"Label: {label}")
    #     plt.imshow(img)
    #     plt.show()
    log_dir = "C:/Users/david/Documents/project/gym-donkeycar/examples/reinforcement_learning/models"
    name_model = "PPO_encoder_1M_best_model"
    
    model = PPO.load(os.path.join(log_dir, name_model))
    # print("Loaded model\n", "-" * 30, "\n", model.policy, "\n", "-" * 30)
    encoder = None
    name_encoder = "encoder_1000_transfer_best.pkl"
    encoder = load_ae(os.path.join(log_dir, name_encoder))

    loss1_steering, loss1_throttle, loss2_steering, loss2_throttle = \
        eval_model(model, dataset,  MSELoss(), L1Loss(), encoder)
    print("Name model: ", name_model)
    print(
        f"MSE_steering={loss1_steering}, MSE_throttle={loss1_throttle}, \nMAE_steering={loss2_steering}, MAE_throttle={loss2_throttle}")




    
