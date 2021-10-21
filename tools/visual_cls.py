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
import pdb

sys.path.append('/mnt/lustre/wanghao3/projects/PatCore/PatchCore_anomaly_detection')
from train_wh import KNN, cvt2heatmap, min_max_norm, embedding_concat, reshape_embedding, heatmap_on_image


def save_normally_map(anomaly_map, input_img, save_path, filepath):
    img_cls = filepath.split('/')[-2]
    save_path_cls = os.path.join(save_path, img_cls)
    os.makedirs(save_path_cls, exist_ok=True)
    filename = filepath.split('/')[-1]
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    if anomaly_map.shape != img.shape:
        anormaly_map = cv2.resize(anomaly_map, (img.shape[0], img.shape[1]))
    anormaly_map_norm = min_max_norm(anormaly_map)
    anormaly_map_norm = cvt2heatmap(anormaly_map_norm * 255)
    anormaly_map_norm_1 = cv2.resize(anormaly_map_norm, (img.shape[1], img.shape[0]))
    # import pdb; pdb.set_trace()
    #anormaly map on img

    heatmap = cv2.resize(anormaly_map_norm, (img.shape[1], img.shape[0]))
    hm_on_img = heatmap_on_image(heatmap, img)
    cv2.imwrite(os.path.join(save_path_cls, f'{filename}.jpg'), img)
    cv2.imwrite(os.path.join(save_path_cls, f'{filename}_amap.jpg'), anormaly_map_norm_1)
    cv2.imwrite(os.path.join(save_path_cls, f'{filename}_amap_on_img.jpg'), hm_on_img)

def transform_data(x, args):
    data_transforms = transforms.Compose([
        transforms.Resize((args.load_size, args.load_size), Image.ANTIALIAS),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_train,
            std=std_train)])
    return data_transforms(x)

def inv_nomarlize(x):
    inv_ = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255], std=[1 / 0.229,
                                                                                            1 / 0.224,
                                                                                            1 / 0.255])
    return inv_(x)





class STPM(pl.LightningModule):
    def __init__(self,hparams):
        super(STPM, self).__init__()
        self.init_features()
        def hook_t(module, input, output):
            self.features.append(output)
        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet50_2', pretrained=True).cuda()
        self.model.eval()
        self.model.layer2[-1].register_forward_hook(hook_t)
        self.model.layer3[-1].register_forward_hook(hook_t)
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
        x_t = self.data_transforms(x).cuda()
        x_t = x_t.unsqueeze(0)
        #pdb.set_trace()
        _ = self.model(x_t)
        return self.features

def test_step(args, model, embedding_coreset, file_list, save_path):
    os.makedirs(save_path, exist_ok=True)
    model = model.cuda()
    f1 = open(file_list,'r')
    for line in tqdm(f1.readlines()):
        filepath, label = line.strip().split(' ')
        filename = filepath.split('/')[-1]
        img = Image.open(filepath).convert('RGB')
        features = model(img)
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

        x = transform_data(img, args)
        x = x.squeeze(0)
        x = inv_nomarlize(x)
        x = x.unsqueeze(0)
        input_x = cv2.cvtColor(x.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
        save_normally_map(anormaly_map_resize_blur, input_x, save_path, filepath)

def get_args():
    parser = argparse.ArgumentParser(description='infer')
    #parser.add_argument('--file_list', type=str, default='/mnt/lustre/wanghao3/projects/PatCore/PatchCore_anomaly_detection/tools/1015/all_50000_1083/move.sh')
    parser.add_argument('--file_list', type=str, default='/mnt/lustre/wanghao3/projects/PatCore/PatchCore_anomaly_detection/datapath/hongzao/test/test.lst')
    #parser.add_argument('--save_file', type=str, default='./tmp3/result.txt')
    parser.add_argument('--save_path', type=str, default=None)
    #parser.add_argument('--embedding_path', type=str, default='/mnt/lustre/wanghao3/projects/PatCore/PatchCore_anomaly_detection/tools/embeddings/ciwa_1_1000/embedding.pickle')
    parser.add_argument('--embedding_path', type=str, default='/mnt/lustre/wanghao3/projects/PatCore/PatchCore_anomaly_detection/tools/embeddings/hongzao/embedding.pickle')
    #parser.add_argument('--embedding_path', type=str, default='aa')
    parser.add_argument('--load_size', type=int, default=256)
    parser.add_argument('--input_size', type=int, default=224)

    args = parser.parse_args()
    return args

mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]
def main():
    args = get_args()
    file_list = args.file_list
    save_path = args.save_path
    embedding_path = args.embedding_path
    embedding_coreset = pickle.load(open(embedding_path, 'rb'))

     #get_model
    model = STPM(hparams=args)
    test_step(args, model, embedding_coreset, file_list, save_path)

if __name__ == '__main__':
    main()








