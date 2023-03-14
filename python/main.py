import sys
from FaceDataset import FaceDataset
import utils
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights, resnet34, ResNet34_Weights
from pre_processing import PreProcessor
from image_entity import ImageEntity
import matplotlib.pyplot as plt
import time
import copy
import os

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
from torchvision.io import read_image

cudnn.benchmark = True

from torch.utils.data import DataLoader

import wandb

USE_WANDB = False


def load_data():
    train_indices = []
    test_indices = []

    train_tensors = []
    test_tensors = []

    file = open("data/face_synthetics/tensors_train_0/indices.txt","r")
    for line in file.readlines():
        for token in line.split(' '):
            train_indices.append(int(token))
    file.close()

    file = open("data/face_synthetics/tensors_test_0/indices.txt","r")
    for line in file.readlines():
        for token in line.split(' '):
            test_indices.append(int(token))
    file.close()

    for index in train_indices:
        tensor = torch.load("data/face_synthetics/tensors_train_0/"+index+"_tensor.pt")
        train_tensors.append(tensor)

    for index in test_indices:
        tensor = torch.load("data/face_synthetics/tensors_test_0/"+index+"_tensor.pt")
        test_tensors.append(tensor)


    return (train_tensors, test_tensors)


def compute_average_face_landmarks():
    sum_vector = np.array(70,2)
    # for i in range(0, 100000):



def read_landmarks(file_path):
    result = []
    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.split(' ')
            x = float(tokens[0])
            y = float(tokens[1])
            result.append({"x":x,"y":y})
    return result

def compute_mean():
    counter = 0
    mean = np.zeros((70,2), dtype="float64")
    for i in range(0, 100000):
        id = str(i).zfill(6)
        path = "data/face_synthetics_full/dataset_100000/"+id+"_ldmks.txt"
        ldmks = read_landmarks(path)
        counter += 1
        local_index = 0
        for land in ldmks:
            mean[local_index, 0] += land["x"]
            mean[local_index, 1] += land["y"]
            local_index += 1
    
    return mean / counter


def show_example_image():
    img = ImageEntity("data/face_synthetics/000001.png", "data/face_synthetics/000001_ldmks.txt")
    img.show()


def compute_show_mean_ldmks():
    image = cv2.imread("data/face_synthetics_full/dataset_100000/000000.png")
    means = load_mean_ldmks()
    print(means)
    for i in range(0, means.shape[0]):
        image = cv2.circle(image, (int(means[i, 0]), int(means[i, 1])), radius=2, color=(255,255,0), thickness=-1)
    plt.imshow(image)
    plt.show()

def load_mean_ldmks():
    landmarks = read_landmarks("mean_ldmks.txt")
    ldmks_array = np.zeros((70, 2))

    local_index = 0
    for landmark in landmarks:
        ldmks_array[local_index, 0] = landmark['x']
        ldmks_array[local_index, 1] = landmark['y']
        local_index += 1

    return ldmks_array

def compute_transformations_from_mean_ldmks():
    mean_ldmks = load_mean_ldmks()

    for i in range(0, 100000):
        id = str(i).zfill(6)
        path = "data/face_synthetics_full/dataset_100000/"+id+"_ldmks.txt"
        ldmks = read_landmarks(path)
        print("Process: " + path)
        # counter += 1
        local_index = 0
        offset_tensor = np.zeros(mean_ldmks.shape)
        for land in ldmks:
            offset_tensor[local_index, 0] = land["x"] - mean_ldmks[local_index, 0]
            offset_tensor[local_index, 1] = land["y"] - mean_ldmks[local_index, 1]
            local_index += 1

        out_path = "data/face_synthetics_full/offset_ldmks/"+id+"_ldmks_offset.txt"
        print("    out => " + out_path)
        
        out_file = open(out_path, "w")
        for i in range(0, offset_tensor.shape[0]):
            out_file.write(str(offset_tensor[i, 0]) + " " + str(offset_tensor[i, 1]) + '\n')
        out_file.write('\n')
        out_file.close()


def create_random_indices():
    (a,b) = utils.create_random_int_arrays(20000, 2000)
    
    file = open("train_indices_20000.txt","w")
    for x in a:
        file.write(str(x) + "\n")
    file.close()

    file = open("test_indices_2000.txt","w")
    for x in b:
        file.write(str(x) + "\n")
    file.close()


def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=25, is_inception=False, model_name="model.pt"):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device).float()
                labels = labels.to(device).float()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs).float()
                        loss1 = criterion(outputs, labels).float()
                        loss2 = criterion(aux_outputs, labels).float()
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs).float()
                        loss = criterion(outputs, labels).float()

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # print("before loss: ", inputs.dtype, loss.dtype)
                        loss.backward()
                        optimizer.step()

                # statistics
                # print(loss.item(), " /// ", inputs.size(0))
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(outputs == labels)
            # print(running_loss, " // // ", len(dataloaders[phase]))
            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects / len(dataloaders[phase])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            if USE_WANDB:
                if phase == "train":
                    wandb.log({"train_loss": epoch_loss})
                    wandb.log({"train_accuracy": epoch_acc})
                else:
                    wandb.log({"test_loss": epoch_loss})
                    wandb.log({"test_accuracy": epoch_acc})


            torch.save(model, "models/"+model_name)
            with open("models/meta.txt","w") as file:
                file.write("{} ".format(epoch))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

        
def main():
    # Init Weights and Biases
    if USE_WANDB :
        wandb.init(project="face-landmarks")
    # Initialisation
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print(f"Main: Device: {device}")
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    data_transforms = {
        'train':
        transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            normalize
        ]),
        'test':
        transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            normalize
        ]),
    }
    
    #weights = ResNet34_Weights.DEFAULT
    
    #pre_processor = weights.transforms()
    
    model = resnet34().to(device).float()

    # model = ResNetP().to(device).float() # resnet50(weights = weights).to(device).float()    
    
    # for param in model.conv1.parameters():
    #     param.requires_grad = False
    # for param in model.bn1.parameters():
    #     param.requires_grad = False
    # for param in model.relu.parameters():
    #     param.requires_grad = False
    # for param in model.maxpool.parameters():
    #     param.requires_grad = False
    # for param in model.layer1.parameters():
    #     param.requires_grad = False
    # for param in model.layer2.parameters():
    #     param.requires_grad = False
    

    # Reset fully connected
    # num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
               nn.Linear(512, 512),
               nn.Linear(512, 256),
               nn.Linear(256, 10)).to(device).float()

    # Create loss function
    criterion = torch.nn.MSELoss().float()
    # criterion = torch.nn.L1Loss().float()

    # Create optimizer
    # optimizer = optim.Adam(model.parameters(), lr=0.001)# , momentum=0.9)
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    # Create scheduler
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    train_dataset = FaceDataset("train", "train_indices_20000.txt","data/crop_256","data/crop_256_ldmks", transform=data_transforms["train"]) #, transform2 = pre_processor)
    test_dataset = FaceDataset("test", "test_indices_2000.txt","data/crop_256","data/crop_256_ldmks",  transform=data_transforms["test"]) #, transform2 = pre_processor)

    dataloaders = {
        "train":DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8),
        "test":DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=8)
    }
    
    model = train_model(model, dataloaders, criterion, optimizer, device, num_epochs=500, model_name="model2.pt")
    
    return 0


def read_landmarks(file_path):
        result = []
        with open(file_path) as f:
            lines = f.readlines()
            for line in lines:
                tokens = line.split(' ')
                if(len(tokens) != 2):
                    # print(f"WARN: (read_landmarks) in line:'{line}', found less than 2 tokens. filepath={file_path}")
                    continue
                x = float(tokens[0])
                y = float(tokens[1])
                result.append({"x":x,"y":y})
        if(len(result) != 70):
            raise Exception(f"Feature length must be 70. Currently: {len(result)}")
        return result

def show_face_with_landmarks(image, landmarks, p1:tuple, p2:tuple):
    for ldmk in landmarks:
        cv2.circle(image, (int(ldmk[0]), int(ldmk[1])), 2, (255,0,0), 1)
        
    cv2.rectangle(image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 255, 0), 2 )

    cv2.imshow("Image", image)
    cv2.waitKey()


def crop_images(debug = False):
    for i in range(0, 10):
        id = str(i).zfill(6)

        # get image
        filename = f"{id}.png"
        # img_path = os.path.join("data/face_synthetics_full/dataset_100000/", filename)
        img_path = os.path.join("data/1000_dataset/", filename)
        # image = read_image(img_path)
        image = cv2.imread(img_path)
        
        # get landmarks
        filename = f"{id}_ldmks.txt"
        # path = os.path.join("data/face_synthetics_full/dataset_100000/", filename)
        path = os.path.join("data/1000_dataset/", filename)
        landmarks = read_landmarks(path)
        print("Read ldmks: ", filename)
        # compute center
        sum_xy = np.zeros((1,2))
        max_btlr = np.zeros((4,2))
        
        # initialize maxs
        max_btlr[0, 0] = landmarks[0]["x"] # bot-x
        max_btlr[0, 1] = landmarks[0]["y"] # bot-y

        max_btlr[1, 0] = landmarks[0]["x"] # top-x
        max_btlr[1, 1] = landmarks[0]["y"] # top-y

        max_btlr[2, 0] = landmarks[0]["x"] # left-x
        max_btlr[2, 1] = landmarks[0]["y"] # left-y

        max_btlr[3, 0] = landmarks[0]["x"] # right-x
        max_btlr[3, 1] = landmarks[0]["y"] # right-y
        
        
        
        cpt = 0
        for ldmk in landmarks:
            # if debug :
            #     cv2.circle(image, (int(ldmk["x"]), int(ldmk["y"])), 2, (255,0,255), 1)
            if debug and cpt == 30:
                cv2.circle(image, (int(ldmk["x"]), int(ldmk["y"])), 2, (255,0,0), 1)
            if debug and (cpt == 48 or cpt == 54):
                cv2.circle(image, (int(ldmk["x"]), int(ldmk["y"])), 2, (255,255,0), 1)
            if debug and (cpt == 68 or cpt == 69):
                cv2.circle(image, (int(ldmk["x"]), int(ldmk["y"])), 2, (255,0,255), 1)
            
            cpt += 1

            sum_xy[0, 0] += ldmk["x"]
            sum_xy[0, 1] += ldmk["y"]
            
            # bot (y)
            if ldmk["y"] > max_btlr[0, 1]:
                max_btlr[0, 0] = ldmk["x"]
                max_btlr[0, 1] = ldmk["y"]

            # top (y)
            if ldmk["y"] < max_btlr[1, 1]:
                max_btlr[1, 0] = ldmk["x"]
                max_btlr[1, 1] = ldmk["y"]
            
            # left (x)
            if ldmk["x"] < max_btlr[2, 0]:
                max_btlr[2, 0] = ldmk["x"]
                max_btlr[2, 1] = ldmk["y"]

            # right (x)
            if ldmk["x"] > max_btlr[3, 0]:
                max_btlr[3, 0] = ldmk["x"]
                max_btlr[3, 1] = ldmk["y"]

        if debug:
            print("Max bbox: ", max_btlr)
            cv2.circle(image, (int(max_btlr[0,0]), int(max_btlr[0,1])), 4, (0,255,0), 1)
            cv2.circle(image, (int(max_btlr[1,0]), int(max_btlr[1,1])), 4, (0,255,0), 1)
            cv2.circle(image, (int(max_btlr[2,0]), int(max_btlr[2,1])), 4, (0,255,0), 1)
            cv2.circle(image, (int(max_btlr[3,0]), int(max_btlr[3,1])), 4, (0,255,0), 1)

        top_left_point = (max_btlr[2,0],  max_btlr[1,1])
        bot_right_point = (max_btlr[3,0],  max_btlr[0,1])
        
        target = 256.0

        # if the width is less than 128px, grow equaly to reach 128px
        if bot_right_point[0] - top_left_point[0] < target :
            difference = target - (bot_right_point[0] - top_left_point[0])
            diff_half = difference / 2.0
            bot_right_point = (bot_right_point[0] + diff_half, bot_right_point[1]) 
            top_left_point = (top_left_point[0] - diff_half, top_left_point[1])

        # test the same thing but for the height
        if bot_right_point[1] - top_left_point[1] < target :
            difference = target - (bot_right_point[1] - top_left_point[1])
            diff_half = difference / 2.0
            bot_right_point = (bot_right_point[0], bot_right_point[1] + diff_half) 
            top_left_point = (top_left_point[0], top_left_point[1] - diff_half)

        if debug:
            cv2.rectangle(image, (int(top_left_point[0]), int(top_left_point[1])), (int(bot_right_point[0]), int(bot_right_point[1])), (255, 255, 0), 2 )
        
        # Crop the image around the bouding box
        sub_image = image[int(top_left_point[1]):int(bot_right_point[1]), int(top_left_point[0]):int(bot_right_point[0])]
        
        # Resize images 
        # cv2.resize(sub_image, (128, 128))

        # Write the file back
        # out_filename = f"c_256_{id}.png"
        # out_img_path = os.path.join("data/crop_256/", out_filename)
        # cv2.imwrite(out_img_path, sub_image)
        
        # Compute new origin vector
        new_origin = top_left_point
        new_landmarks = []
        # Compute landmarks new coordinates 
        for landmark in landmarks:
            tmp = [landmark["x"], landmark["y"]]
            tmp[0] -= new_origin[0]
            tmp[1] -= new_origin[1]
            new_landmarks.append(tmp)

        # Write landmarks files
        # out_ldmks_filename = f"c_256_ldkms_{id}.txt"
        # out_ldmks_path = out_img_path = os.path.join("data/crop_256_ldmks/", out_ldmks_filename)
        # file = open(out_ldmks_path, "w")
        # for landmark in new_landmarks:
        #     file.write(f"{landmark[0]:.5f} {landmark[1]:.5f} \n")
        # file.close()

        # show_face_with_landmarks(sub_image,new_landmarks,(0,0), (256,256))

        if debug:
            cv2.imshow("Out", sub_image)
            cv2.waitKey()

        ctr_xy = sum_xy / len(landmarks)
        ctr_x = ctr_xy[0,0]
        ctr_y = ctr_xy[0,1]

        # compute bounding box
        width = 128
        height = 128
        bbox = [
            [ctr_x, ctr_y],     # bot left
            [ctr_x, ctr_y],     # top left
            [ctr_x, ctr_y],     # top right
            [ctr_x, ctr_y]      # bot left
        ]
        # get ROI from image
        # if bouding box get out of the image print to see if there is some with problems
        # save ROI in folders somewhere else

def reduce_image_size():
    for i in range(0, 1000):
        id = str(i).zfill(6)
        # get image
        filename = f"{id}.png"
        img_path = os.path.join("data/face_synthetics_full/dataset_100000/", filename)
        image = cv2.imread(img_path)
        resized = cv2.resize(image, (128,128))
        out_path = os.path.join("data/face_128/", filename)
        cv2.imwrite(out_path, resized)

def test():
    # (train_data, test_data) = load_data()
    # training_data = FaceDataset()
    # train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    # compute_transformations_from_mean_ldmks()
    # compute_show_mean_ldmks()
    # create_random_indices()
    # crop_images(debug=True)
    # reduce_image_size()
    print("== test end ==")


if __name__ == "__main__":
    print(sys.argv)
    if(len(sys.argv) > 1):
        if(sys.argv[1] == "main"):
            if len(sys.argv) > 2 and sys.argv[2] == "wandb":
                USE_WANDB = True
            sys.exit(main())
        elif(sys.argv[1] == "test"):
            sys.exit(test())
        else:
            print("Choose main or test. ex: 'python main.py main' or 'python main.py test'")
    else:
        sys.exit(main())