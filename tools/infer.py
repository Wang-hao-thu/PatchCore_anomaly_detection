import argparse
import os
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from PIL import Image
import pickle
import pytorch_lightning as pl
from torchvision import transforms

sys.path.append('/mnt/lustre/wanghao3/projects/PatCore/PatchCore_anomaly_detection')
from train_wh import KNN, cvt2heatmap, min_max_norm, embedding_concat, reshape_embedding


def save_normally_map(anomaly_map, input_img, save_path, filename):
    if anomaly_map.shape != input_img.shape:
        anormaly_map = cv2.resize(anomaly_map, (input_img.shape[0], input_img.shape[1]))
    anormaly_map_norm = min_max_norm(anormaly_map)
    anormaly_map_norm = cvt2heatmap(anormaly_map_norm * 255)

    #anormaly map on img
    heatmap = cvt2heatmap(anormaly_map_norm * 255)
    hm_on_img = heatmap_on_image(heatmap, input_img)
    cv2.imwrite(os.path.join(save_path, f'{filename}_amap.jpg'), anormaly_map_norm)
    cv2.imwrite(os.path.join(save_path, f'{fileneme}_amap_on_img.jpg'), hm_on_img)

def transform_data(x, hparams):
    data_transforms = transforms.Compose([
        transforms.Resize((hparams.load_size, hparams.load_size), Image.ANTIALIAS),
        transforms.CenterCrop(hparams.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_train,
            std=std_train)])
    return data_transforms(x)





class STPM(pl.LightningModule):
    def __init__(self,hparams):
        super(STPM, self).__init__()
        self.init_features()
        def hook_t(module, input, output):
            self.features.append(output)
        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet50_2', pretrained=True).cuda()
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.layer2[-1].register_forward_hook(hook_t)
        self.model.layer3[-1].register_forward_hook(hook_t)
        self.model.eval()

        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.data_transforms = transforms.Compose([
            transforms.Resize((hparams.load_size, hparams.load_size), Image.ANTIALIAS),
            transforms.CenterCrop(hparams.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_train,
                std=std_train)])

    def init_features(self):
        self.features = []

    def forward(self, x):
        self.init_features()
        _ = self.model(x)
        return self.features

def test_step(arg, model, embedding_coreset, file_list, save_file):
    model = model.cuda()
    f1 = open(file_list,'r')
    f2 = open(save_file, 'w')
    for line in tqdm(f1.readlines()):
        filepath, label = line.strip().split(' ')
        filename = filepath.split('/')[-1]
        img = Image.open(filepath).convert('RGB')
        x = transform_data(img, arg).cuda()
        x = x.unsqueeze(0)

        features = model(x)

        embeddings = []
        for feature in features:
            m = torch.nn.AvgPool2d(3, 1, 1)
            embeddings.append(m(feature))
        embedding_ = embedding_concat(embeddings[0], embeddings[1])
        embedding_test = np.array(reshape_embedding(embedding_.cpu().detach().numpy()))
        #NN
        knn = KNN(torch.from_numpy(embedding_coreset).cuda(), k=9)
        score_pathes = knn(torch.from_numpy(embedding_test).cuda())[0].cpu().detach().numpy()



        anormaly_map = score_pathes[:, 0].reshape((28, 28))
        N_b = score_pathes[np.argmax(score_pathes[:, 0])]
        w = (1 - (np.max(np.exp(N_b)) / np.sum(np.exp(N_b))))
        score = w * max(score_pathes[:, 0])#image_level score
        anormaly_map_resize = cv2.resize(anormaly_map, (224, 224))
        anormaly_map_resize_blur = gaussian_filter(anormaly_map_resize, sigma=4)

        #save_file
        f2.write(filepath + ' ' + str(label) + ' ' + str(score) + '\n')


        #save_img
        #需要将x转换为test 的 tensor 才可以这样转换
        #x = transform_data(img, arg)
        #x = x.squeeze(0)
        #x = inv_normalize(x)
        #x = x.unsqueeze(0)
        #input_x = cv2.cvtColor(x.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
        #save_normally_map(anormaly_map_resize_blur, input_x, save_path, filename)
        #return anormaly_map_resize_blur

def get_args():
    parser = argparse.ArgumentParser(description='infer')
    parser.add_argument('--file_list', type=str, default='/mnt/lustre/wanghao3/projects/PatCore/PatchCore_anomaly_detection/datapath/0908-0924/ceshi/test.lst')
    parser.add_argument('--save_file', type=str, default='./tmp5/result.txt')
    parser.add_argument('--embedding_path', type=str, default='/mnt/lustre/wanghao3/projects/PatCore/PatchCore_anomaly_detection/tools/embeddings/cu_wai_1000/embedding.pickle')
    parser.add_argument('--load_size', type=int, default=256)
    parser.add_argument('--input_size', type=int, default=224)

    args = parser.parse_args()
    return args

mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]
def main():
    args = get_args()
    file_list = args.file_list
    save_file = args.save_file
    embedding_path = args.embedding_path
    print(embedding_path)
    embedding_coreset = pickle.load(open(embedding_path, 'rb'))

     #get_model
    model = STPM(hparams=args)
    test_step(args, model, embedding_coreset, file_list, save_file)

if __name__ == '__main__':
    main()











