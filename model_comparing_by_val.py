import SimpleITK
import numpy as np
from matplotlib import pyplot as plt

from pandas import DataFrame
from scipy.ndimage import center_of_mass, label

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from evalutils import DetectionAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)
from skimage import transform
import json
from typing import Dict

import train_val_utils.utils as utils
from train_val_utils.dataset import CXRNoduleDataset, get_transform
from train_val_utils.engine import train_one_epoch
from train_val_utils.engine import evaluate

import os
import itertools
from pathlib import Path
from postprocessing import get_NonMaxSup_boxes

'''
NODE21 template nodule detection codebase
Author: Ecem Sogancioglu
email: ecemsogancioglu@gmail.com
'''

# This parameter adapts the paths between local execution and execution in docker. You can use this flag to switch between these two modes.
# For building your docker, set this parameter to True. If False, it will run process.py locally for test purposes.
execute_in_docker = True

class Noduledetection(DetectionAlgorithm):
    def __init__(self, input_dir, output_dir, train=False, retrain=False, retest=False):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
            input_path = Path(input_dir),
            output_file = Path(os.path.join(output_dir,'nodules.json'))
        )
        
        #------------------------------- LOAD the model here ---------------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_path, self.output_path = input_dir, output_dir
        print('using the device ', self.device)





        #------------------------------- LOAD the node21 baseline model ---------------------------------
        self.model_1 = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=True)
        num_classes = 2  # 1 class (nodule) + background
        in_features = self.model_1.roi_heads.box_predictor.cls_score.in_features
        self.model_1.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)        
               
        if train:
            # validation
            print('loading the model file pretrained_model/model.pth :')
            self.model_1.load_state_dict(
            torch.load(
                Path("pretrained_model/model.pth"),
                map_location=self.device,
                )
            ) 
            
        self.model_1.to(self.device)
        
        
        #--------------------- LOAD the model of process_faster_rcnn_val.py -----------------------     
        self.model_2 = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
        num_classes = 2  # 1 class (nodule) + background
        in_features = self.model_2.roi_heads.box_predictor.cls_score.in_features
        self.model_2.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)        
               
        if train:
            # validation
            print('loading the model file pretrained_model/model_faster_rcnn.pth :')
            self.model_2.load_state_dict(
            torch.load(
                Path("pretrained_model/model_faster_rcnn.pth"),
                map_location=self.device,
                )
            ) 
            
        self.model_2.to(self.device)        
        

        #--------------------- LOAD the model of process_faster_rcnn_r101_val.py -----------------------
        backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone('resnet101',pretrained=True)
        self.model_3 = torchvision.models.detection.FasterRCNN(backbone,num_classes=2)        
               
        if train:
            # validation
            print('loading the model file pretrained_model/model_faster_rcnn_r101.pth :')
            self.model_3.load_state_dict(
            torch.load(
                Path("pretrained_model/model_faster_rcnn_r101.pth"),
                map_location=self.device,
                )
            ) 
            
        self.model_3.to(self.device)
        
        
        
        #--------------------- LOAD the model of process_retinanet_val.py -----------------------       
        self.model_4 = torchvision.models.detection.retinanet_resnet50_fpn(
                                                                        pretrained=False, 
                                                                        pretrained_backbone=False, 
                                                                        num_classes=2)        
               
        if train:
            # for validation
            print('loading the model file pretrained_model/model_retinanet.pth :')
            self.model_4.load_state_dict(
            torch.load(
                Path("pretrained_model/model_retinanet.pth"),
                map_location=self.device,
                )
            ) 
            
        self.model_4.to(self.device)   



    def save(self):
        with open(str(self._output_file), "w") as f:
            json.dump(self._case_results[0], f)
            
    # TODO: Copy this function for your processor as well!
    def process_case(self, *, idx, case):
        '''
        Read the input, perform model prediction and return the results. 
        The returned value will be saved as nodules.json by evalutils.
        process_case method of evalutils
        (https://github.com/comic/evalutils/blob/fd791e0f1715d78b3766ac613371c447607e411d/evalutils/evalutils.py#L225) 
        is overwritten here, so that it directly returns the predictions without changing the format.
        
        '''
        # Load and test the image for this case
        input_image, input_image_file_path = self._load_input_image(case=case)
        
        # Detect and score candidates
        scored_candidates = self.predict(input_image=input_image)
        
        # Write resulting candidates to nodules.json for this case
        return scored_candidates
    
   
    
    #--------------------Write your retrain function here ------------
    def train(self, num_epochs = 50):
        '''
        input_dir: Input directory containing all the images to train with
        output_dir: output_dir to write model to.
        num_epochs: Number of epochs for training the algorithm.
        '''
        
        
        

        # create training dataset and defined transformations

        val_dir = self.input_path+"/test"
        val_dataset = CXRNoduleDataset(val_dir, os.path.join(val_dir, 'test.csv'), get_transform(train=False))
        
            
        val_data_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=2, shuffle=True, num_workers=4,
            collate_fn=utils.collate_fn)
    

        print("\n\n\nevaluate node21 baseline model on the test dataset")       
        self.model_1.eval()  
        evaluate(self.model_1, val_data_loader, device=self.device)      

        print("\n\n\nevaluate model of process_faster_rcnn_val.py on the test dataset")
        self.model_2.eval()  
        evaluate(self.model_2, val_data_loader, device=self.device)  

        print("\n\n\nevaluate model of process_faster_rcnn_r101_val.py on the test dataset")
        self.model_3.eval()  
        evaluate(self.model_3, val_data_loader, device=self.device)  

        print("\n\n\nevaluate model of process_retinanet_val.py on the test dataset")
        self.model_4.eval()  
        evaluate(self.model_4, val_data_loader, device=self.device)          
            
          
      

    def format_to_GC(self, np_prediction, spacing) -> Dict:
        '''
        Convenient function returns detection prediction in required grand-challenge format.
        See:
        https://comic.github.io/grandchallenge.org/components.html#grandchallenge.components.models.InterfaceKind.interface_type_annotation
        
        
        np_prediction: dictionary with keys boxes and scores.
        np_prediction[boxes] holds coordinates in the format as x1,y1,x2,y2
        spacing :  pixel spacing for x and y coordinates.
        
        return:
        a Dict in line with grand-challenge.org format.
        '''
        # For the test set, we expect the coordinates in millimeters. 
        # this transformation ensures that the pixel coordinates are transformed to mm.
        # and boxes coordinates saved according to grand challenge ordering.
        x_y_spacing = [spacing[0], spacing[1], spacing[0], spacing[1]]
        boxes = []
        for i, bb in enumerate(np_prediction['boxes']):
            box = {}   
            box['corners']=[]
            x_min, y_min, x_max, y_max = bb*x_y_spacing
            x_min, y_min, x_max, y_max  = round(x_min, 2), round(y_min, 2), round(x_max, 2), round(y_max, 2)
            bottom_left = [x_min, y_min,  np_prediction['slice'][i]] 
            bottom_right = [x_max, y_min,  np_prediction['slice'][i]]
            top_left = [x_min, y_max,  np_prediction['slice'][i]]
            top_right = [x_max, y_max,  np_prediction['slice'][i]]
            box['corners'].extend([top_right, top_left, bottom_left, bottom_right])
            box['probability'] = round(float(np_prediction['scores'][i]), 2)
            boxes.append(box)
        
        return dict(type="Multiple 2D bounding boxes", boxes=boxes, version={ "major": 1, "minor": 0 })
        
    def merge_dict(self, results):
        merged_d = {}
        for k in results[0].keys():
            merged_d[k] = list(itertools.chain(*[d[k] for d in results]))
        return merged_d
        
    def predict(self, *, input_image: SimpleITK.Image) -> DataFrame:
        self.model.eval() 
        
        image_data = SimpleITK.GetArrayFromImage(input_image)
        spacing = input_image.GetSpacing()
        image_data = np.array(image_data)
        
        if len(image_data.shape)==2:
            image_data = np.expand_dims(image_data, 0)
            
        results = []
        # operate on 3D image (CXRs are stacked together)
        for j in range(len(image_data)):
            # Pre-process the image
            image = image_data[j,:,:]
            # The range should be from 0 to 1.
            image = image.astype(np.float32) / np.max(image)  # normalize
            image = np.expand_dims(image, axis=0)
            tensor_image = torch.from_numpy(image).to(self.device)#.reshape(1, 1024, 1024)
            with torch.no_grad():
                prediction = self.model([tensor_image.to(self.device)])
            
            prediction = [get_NonMaxSup_boxes(prediction[0])]
            
            
            # Following bbox plot code borrowed from 
            # https://github.com/zhangxiann/PyTorch_Practice/blob/master/lesson8/detection_demo.py
            out_boxes = prediction[0]["boxes"]
            out_scores = prediction[0]["scores"]
            num_boxes = len(out_boxes)
            
            # maximum draw 10 bbox
            max_vis = 10
            
            # only draw bbox with probability >= 0.1
            thres = 0.1
            
            class_name = "Nodule"            
            
            fig, ax = plt.subplots(figsize=(12, 12))
            ax.imshow(image[0], cmap='gray') 

            for idx in range(0, min(num_boxes, max_vis)):
                if(self.device == 'cpu'):
                    score = out_scores[idx].numpy()
                    bbox = out_boxes[idx].numpy()     
                else:    
                    score = out_scores[idx].cpu().numpy()
                    bbox = out_boxes[idx].cpu().numpy()      

                if score < thres:
                    continue

                ax.add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], fill=False,
                                           edgecolor='red', linewidth=3.5))
                ax.text(bbox[0], bbox[1] - 2, '{:s} {:.3f}'.format(class_name, score), bbox=dict(facecolor='blue', alpha=0.5),
                        fontsize=14, color='white')

            fig.savefig('./test_figs_with_bbox/image_with_bbox_'+str(j)+'.png')
            
            
            # convert predictions from tensor to numpy array.
            np_prediction = {str(key):[i.cpu().numpy() for i in val]
                   for key, val in prediction[0].items()}
            np_prediction['slice'] = len(np_prediction['boxes'])*[j]
            results.append(np_prediction)
        
        predictions = self.merge_dict(results)
        data = self.format_to_GC(predictions, spacing)
        print(data)
        return data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        prog='process.py',
        description=
            'Reads all images from an input directory and produces '
            'results in an output directory')

    parser.add_argument('input_dir', help = "input directory to process")
    parser.add_argument('output_dir', help = "output directory generate result files in")
    parser.add_argument('--train', action='store_true', help = "Algorithm on train mode.")
    parser.add_argument('--retrain', action='store_true', help = "Algorithm on retrain mode (loading previous weights).")
    parser.add_argument('--retest', action='store_true', help = "Algorithm on evaluate mode after retraining.")

    parsed_args = parser.parse_args()  
    if (parsed_args.train or parsed_args.retrain):# train mode: retrain or train
        Noduledetection(parsed_args.input_dir, parsed_args.output_dir, parsed_args.train, parsed_args.retrain, parsed_args.retest).train()
    else:# test mode (test or retest)
        Noduledetection(parsed_args.input_dir, parsed_args.output_dir, retest=parsed_args.retest).process()
            
    
   
    
    
    
    
    
