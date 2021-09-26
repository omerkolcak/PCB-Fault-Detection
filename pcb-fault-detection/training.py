import os
from numpy import zeros, asarray

import mrcnn.utils
import mrcnn.config
import mrcnn.model


CLASS_IDS = {"1" : "open", "2" : "short", "3" : "mousebite", "4" : "spur", "5" : "copper", "6" : "pin-hole"}

class PCBFaultDataset(mrcnn.utils.Dataset):
    
    def load_dataset(self, dataset_dir, is_train=True):
        
        self.add_class("pcb-fault",1,"open")
        self.add_class("pcb-fault",2,"short")
        self.add_class("pcb-fault",3,"mousebite")
        self.add_class("pcb-fault",4,"spur")
        self.add_class("pcb-fault",5,"copper")
        self.add_class("pcb-fault",6,"pin-hole")
        
        if is_train:
            read_file = "trainval.txt"
        else:
            read_file = "test.txt"
            
        with open("PCBData/" + read_file) as f:
            
            path = f.readline()
            
            image_id = 0
            
            while path:
                image,annotation = path.split(" ")
                
                annotation = annotation.split("\n")[0]
                
                image = dataset_dir + "/" + image
                annotation = dataset_dir + "/" + annotation
                
                self.add_image("dataset",image_id=image_id, path=image, annotation=annotation)
                
                image_id = image_id + 1
                path = f.readline()
                
    
    def load_mask(self,image_id):
        info = self.image_info[image_id]
        info = self.image_info[image_id]
        path = info['annotation']
        boxes, w, h, classes = self.extract_boxes(path)
        masks = zeros([h, w, len(boxes)], dtype='uint8')

        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index(classes[i]))
            
        return masks, asarray(class_ids, dtype='int32')
        
    
    def extract_boxes(self, filename):
        
        boxes = []
        classes = []
        
        with open(filename) as f:
            
            values = f.readline()
            
            while values:
                
                xmin,ymin,xmax,ymax,class_id = values.split(" ")
                class_id = class_id.split("\n")[0]
                
                class_name = CLASS_IDS[class_id]
                classes.append(class_name)
                
                box = [xmin, ymin, xmax, ymax]
                boxes.append(box)
                
                values = f.readline()
        
        return boxes, 600, 600, classes
    
    
class PCBFaultConfig(mrcnn.config.Config):
    NAME = "pcb-fault"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    NUM_CLASSES = 4

    STEPS_PER_EPOCH = 131
    

# Train
train_dataset = PCBFaultDataset()
train_dataset.load_dataset(dataset_dir='PCBData', is_train=True)
train_dataset.prepare()

# Validation
validation_dataset = PCBFaultDataset()
validation_dataset.load_dataset(dataset_dir='./pcb-fault-detection/PCBData', is_train=False)
validation_dataset.prepare()

# Configuration
pcb_config = PCBFaultConfig()
 
 
# Build the Mask R-CNN Model Architecture
model = mrcnn.model.MaskRCNN(mode='training', 
                             model_dir='./', 
                             config=pcb_config)

model.load_weights(filepath='mask_rcnn_coco.h5', 
                   by_name=True, 
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

model.train(train_dataset=train_dataset, 
            val_dataset=validation_dataset, 
            learning_rate=pcb_config.LEARNING_RATE, 
            epochs=1, 
            layers='heads')
