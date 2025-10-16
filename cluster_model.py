#encoding=utf8

print('importing')
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transforms
import torch
import glob
from log import log
import os
import pickle
import shutil
import cv2

class ClusterModel: 
    '''
    cluster model to cluster state images.
    '''

    def __init__(self): 
        '''
        init
        '''
        # load resnet18 model and remove the last layer.
        self.feature_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.feature_model = torch.nn.Sequential(*list(self.feature_model.children())[:-1])
        self.feature_model.eval()
        if torch.cuda.is_available(): 
            self.feature_model = self.feature_model.cuda()

        self.eval_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.data_path = './images/*.png'
        self.arr_image_path = glob.glob(self.data_path)
        self.features = []

        self.CLUSTER_MODEL_FILE = 'model.cluster.kmeans'
        self.n_clusters = 32
        self.max_iter = 10000
        self.tol = 1e-4
        self.cluster_model = KMeans(n_clusters=self.n_clusters, max_iter=self.max_iter, tol=self.tol)


    def extract_features(self): 
        '''
        extract features from images dir
        '''
        log.info('extract_features')
        i = 0
        total_count = len(self.arr_image_path)
        print('total_count:', total_count)

        for image_path in self.arr_image_path: 
            # print('image_path: ', image_path)
            pil_image = Image.open(image_path)
            # print('old mode: ', pil_image.mode)
            if pil_image.mode == 'RGBA': 
                pil_image = pil_image.convert('RGB')
            # print('new mode: ', pil_image.mode)
            # (301, 301)
            # print('pil_image size: ', pil_image.size)

            pil_image = self.eval_transform(pil_image)
            inputs = pil_image.unsqueeze(0)
            # [1, 3, 224, 224]
            # print('inputs shape:', inputs.shape)

            with torch.no_grad():
                if torch.cuda.is_available(): 
                    inputs = inputs.cuda()
                outputs = self.feature_model(inputs)
                # shape 512
                feature = outputs.squeeze()
                self.features.append(feature.cpu())

            i += 1
            if i % 100 == 0: 
                log.debug('%s / %s' % (i, total_count))

        log.info('extract_features finished, total: %s' % (len(self.features)))


    def train(self): 
        '''
        train the model
        '''

        log.info('train')

        self.extract_features()
        self.cluster_model.fit(self.features)
        labels = self.cluster_model.labels_
        log.info('train finished, dump it')
        with open(self.CLUSTER_MODEL_FILE, 'wb') as f: 
            pickle.dump(self.cluster_model, f)
        log.info('dumped')
        log.info('labels: %s %s', len(labels), labels)

        log.info('copying files')
        for i in range(self.n_clusters):
            image_dir = 'images/%s' % (i)
            if os.path.exists(image_dir): 
                shutil.rmtree(image_dir)
            os.mkdir(image_dir)

        for label, img_path in zip(labels, self.arr_image_path): 
            shutil.copy(img_path, os.path.join('./images/', str(label), os.path.basename(img_path)))

        log.info('done')


    def predict_file(self, image_path): 
        '''
        predict class label from image
        '''
        log.info('predict')
        pil_image = Image.open(image_path).convert('RGB')
        cv2_image = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
        '''
        if self.has_danger(cv2_image): 
            return self.n_clusters
        '''

        pil_image = self.eval_transform(pil_image)
        inputs = pil_image.unsqueeze(0)
        with torch.no_grad():
            if torch.cuda.is_available(): 
                inputs = inputs.cuda()
            outputs = self.feature_model(inputs)
            feature = outputs.squeeze()
            log.info('feature.shape: %s' % (feature.shape))

        result = self.cluster_model.predict([feature.cpu()])
        return result[0]


    def load(self): 
        '''
        load pre-trained cluster model
        '''
        log.info('loading cluster model...')
        with open(self.CLUSTER_MODEL_FILE, 'rb') as f: 
            self.cluster_model = pickle.load(f)
        log.info('loaded.')


    def predict_env_inputs(self, inputs, image): 
        '''
        project env input to some class
        '''
        with torch.no_grad():
            if torch.cuda.is_available(): 
                inputs = inputs.cuda()

            outputs = self.feature_model(inputs)
            feature = outputs.squeeze()

        result = self.cluster_model.predict([feature.cpu()])
        return result[0]


    def has_danger(self, image): 
        '''
        wei
        @param: image, cv2 BGR format
        '''
        return False

        # (301, 301, 3)

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_1 = np.array([0, 120, 70])
        upper_1 = np.array([10, 255, 255])
        mask_1 = cv2.inRange(hsv_image, lower_1, upper_1)
        # res_1 = cv2.bitwise_and(image, image, mask=mask_1)
     
        lower_2 = np.array([170, 120, 70])
        upper_2 = np.array([180, 255, 255])
        mask_2 = cv2.inRange(hsv_image, lower_2, upper_2)
        # res_2 = cv2.bitwise_and(image, image, mask=mask_2)
     
        mask_3 = mask_1 + mask_2
        red_mask = mask_3

        
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_region = cv2.bitwise_and(image, image, mask=red_mask)

        cnt = np.sum(red_mask) / 255
        print('cnt: ', cnt)

        '''
        cv2.imshow('image', image)
        # cv2.imshow('red_region', red_region)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''

        if cnt > 222: 
            return True

        return False


if __name__ == '__main__': 
    m = ClusterModel()
    # m.train()
    m.load()

    '''
    for image_path in m.arr_image_path: 
        result = m.predict_file(image_path)
        print('predict result: ', result)
    '''

    image_path = './images/20251012_172225_0.png'
    result = m.predict_file(image_path)
    print('predict result: ', result)

    image_path = './images/20251012_172838_36.png'
    result = m.predict_file(image_path)
    print('predict result: ', result)

    image_path = './images/20251012_173251_121.png'
    result = m.predict_file(image_path)
    print('predict result: ', result)

    image_path = './images/20251012_173646_141.png'
    result = m.predict_file(image_path)
    print('predict result: ', result)

    image_path = './images/20251012_173858_3.png'
    result = m.predict_file(image_path)
    print('predict result: ', result)

    image_path = './images/20251012_173858_12.png'
    result = m.predict_file(image_path)
    print('predict result: ', result)

    image_path = './images/20251012_172838_37.png'
    result = m.predict_file(image_path)
    print('predict result: ', result)

    image_path = './images/20251012_172838_38.png'
    result = m.predict_file(image_path)
    print('predict result: ', result)

    image_path = './images/20251012_172838_39.png'
    result = m.predict_file(image_path)
    print('predict result: ', result)

    image_path = './images/20251012_173050_64.png'
    result = m.predict_file(image_path)
    print('predict result: ', result)

    image_path = './images/20251012_173251_133.png'
    result = m.predict_file(image_path)
    print('predict result: ', result)

    image_path = './images/20251012_173646_139.png'
    result = m.predict_file(image_path)
    print('predict result: ', result)

    image_path = './images/20251012_173646_140.png'
    result = m.predict_file(image_path)
    print('predict result: ', result)


