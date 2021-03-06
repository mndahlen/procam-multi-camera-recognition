import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image, ImageOps

"""
Wrapper of Resnet(Or any model for extracting features) for use in main script.
"""

device = torch.device("cpu")

class FeatureExtractor(object):
    def __init__(self, model_path, zero_pad = False):
        self.zero_pad = zero_pad
        self.resnet = torch.load(model_path)
        self.layer = self.resnet._modules.get('avgpool')
        self.resnet.eval()
        self.scaler = transforms.Scale((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    def get_feature_embedding(self,img):
        # Copied from https://becominghuman.ai/extract-a-feature-vector-for-any-image-with-pytorch-9717561d1d4c
        # 1. Create a PyTorch Variable with the transformed image
        if self.zero_pad:
            img = self.resize_with_padding(img, (224,224))
        t_img = Variable(self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0))
        # 2. Create a vector of zeros that will hold our feature vector
        #    The 'avgpool' layer has an output size of 512
        my_embedding = torch.zeros([1,512,1,1])
        # 3. Define a function that will copy the output of a layer
        def copy_data(m, i, o):
            my_embedding.copy_(o.data)
        # 4. Attach that function to our selected layer
        h = self.layer.register_forward_hook(copy_data)
        # 5. Run the model on our transformed image
        self.resnet(t_img)
        # 6. Detach our copy function from the layer
        h.remove()
        # 7. Return the feature vector
        return my_embedding

    def get_cam_persons(self,bbox,im,persons,T_bbox=5,T_sim = 0):
        num_detected_persons = bbox.shape[0]
        persons_empty = False

        if T_sim < 0:
            print("T_sim must be non-negative!")
            exit(1)

        if not persons:
            persons_empty = True
            persons = {}    

        assigned_persons = []
        persons_detected_not_assigned = []

        # First loop through all detected persons bbox and try to match with bbox from previous frame 
        for i in range(0,num_detected_persons):
            if persons_empty:
                persons_detected_not_assigned.append(i)
            else:
                b = bbox[i]
                match_found = False
                for person_id in persons:
                    if not match_found:
                        prev_frame_feature = persons[person_id]["feature"]
                        if self.bbox_centers_are_close(b, persons[person_id]["bbox"],T_bbox):
                            persons[person_id]["bbox"] = b
                            assigned_persons.append(person_id)
                            match_found = True
                if not match_found:
                    persons_detected_not_assigned.append(i)

        # For the ones that got bbox match, add new features to average feature
        for person_id in assigned_persons:
            b = persons[person_id]["bbox"]
            x1 = int(b[0])
            y1 = int(b[1])
            x2 = int(b[2])
            y2 = int(b[3])
            person_im = im[y1:y2,x1:x2] 
            feature = self.get_feature_embedding(Image.fromarray(person_im)).flatten().numpy()
            history = persons[person_id]["history"]
            combined_features = (history*prev_frame_feature + feature)/(history + 1)
            persons[person_id]["feature"] = combined_features
            persons[person_id]["history"] = history + 1

        # For those that did not get bbox match, try to match on feature, otherwise is new person
        for detected_person_id in persons_detected_not_assigned: 
            b = bbox[detected_person_id]
            x1 = int(b[0])
            y1 = int(b[1])
            x2 = int(b[2])
            y2 = int(b[3])
            person_im = im[y1:y2,x1:x2] 
            detected_person_feature = self.get_feature_embedding(Image.fromarray(person_im)).flatten().numpy()           
            
            if persons_empty:
                persons[detected_person_id] = {"feature":detected_person_feature, "bbox":b, "history":1}
            else:                    
                match_found = False
                for person_id in persons:
                    if not match_found:
                        if person_id not in assigned_persons: # (1)
                            feature = persons[person_id]["feature"]
                            sim = self.get_cosine_sim(feature,detected_person_feature)
                            if sim >= T_sim:
                                history = persons[person_id]["history"]
                                combined_features = (history*prev_frame_feature + feature)/(history + 1)
                                persons[person_id]["feature"] = combined_features
                                persons[person_id]["history"] = history + 1
                                persons[person_id]["bbox"] = b
                                assigned_persons.append(person_id) # make sure this person is not used again above (1)
                                match_found = True

                if not match_found:
                    new_id = max(persons.keys()) + i + 1
                    persons[new_id] = {"feature":detected_person_feature, "bbox":b, "history":1}

        return persons

    def bbox_are_close(self, b1, b2, T):
        x1_1 = int(b1[0])
        y1_1 = int(b1[1])
        x1_2 = int(b1[2])
        y1_2 = int(b1[3])
        x2_1 = int(b2[0])
        y2_1 = int(b2[1])
        x2_2 = int(b2[2])
        y2_2 = int(b2[3])

        DX1 = x1_1 - x2_1
        DY1 = y1_1 - y2_1
        DX2 = x1_2 - x2_2
        DY2 = y1_2 - y2_2

        D1 = (DX1**2 + DY1**2)**(1/2)
        D2 = (DX2**2 + DY2**2)**(1/2)

        if D1 > T or D2 > T:
            return False
        print(D1,D2)

        #print("TRUE")
        return True

    def bbox_centers_are_close(self, b1, b2, T):
        x1_1 = int(b1[0])
        y1_1 = int(b1[1])
        x1_2 = int(b1[2])
        y1_2 = int(b1[3])
        
        x2_1 = int(b2[0])
        y2_1 = int(b2[1])
        x2_2 = int(b2[2])
        y2_2 = int(b2[3])

        center_1_x = (x1_2 + x1_1)/2
        center_1_y = (y1_2 + y1_1)/2
        center_2_x = (x2_2 + x2_1)/2
        center_2_y = (y2_2 + y2_1)/2

        diff_center_x = center_1_x - center_2_x
        diff_center_y = center_1_y - center_2_y

        sqr_diff = diff_center_x**2 + diff_center_y**2

        if sqr_diff <= T:
            return True
        return False

    def bbox_get_center_dist(self, b1, b2):
        x1_1 = int(b1[0])
        y1_1 = int(b1[1])
        x1_2 = int(b1[2])
        y1_2 = int(b1[3])
        
        x2_1 = int(b2[0])
        y2_1 = int(b2[1])
        x2_2 = int(b2[2])
        y2_2 = int(b2[3])

        center_1_x = (x1_2 + x1_1)/2
        center_1_y = (y1_2 + y1_1)/2
        center_2_x = (x2_2 + x2_1)/2
        center_2_y = (y2_2 + y2_1)/2

        diff_center_x = center_1_x - center_2_x
        diff_center_y = center_1_y - center_2_y

        sqr_diff = diff_center_x**2 + diff_center_y**2
        
        return sqr_diff**(1/2)

    def get_cosine_sim(self,v1,v2):
        lower_limit = 1e-6
        norm_product = max(np.linalg.norm(v1)*np.linalg.norm(v2),lower_limit)
        cosinesim = np.dot(v1,v2)/norm_product
        return  cosinesim

    def resize_with_padding(self, img, expected_size):
        img.thumbnail((expected_size[0], expected_size[1]))
        delta_width = expected_size[0] - img.size[0]
        delta_height = expected_size[1] - img.size[1]
        pad_width = delta_width // 2
        pad_height = delta_height // 2
        padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
        return ImageOps.expand(img, padding)

    def get_closest_persons(self, main_idx, main, sub_1, sub_2, sub_3):
        main_feature = main[main_idx]["feature"]

        closest_sub_1_val = 0
        closest_sub_1_idx = 0
        for idx in sub_1:
            sub_feature= sub_1[idx]["feature"]
            sim = self.get_cosine_sim(sub_feature,main_feature)
            if sim > closest_sub_1_val:
                closest_sub_1_val = sim
                closest_sub_1_idx = idx

        closest_sub_2_val = 0
        closest_sub_2_idx = 0
        for idx in sub_2:
            sub_feature= sub_2[idx]["feature"]
            sim = self.get_cosine_sim(sub_feature,main_feature)
            if sim > closest_sub_2_val:
                closest_sub_2_val = sim
                closest_sub_2_idx = idx

        closest_sub_3_val = 0
        closest_sub_3_idx = 0
        for idx in sub_3:
            sub_feature= sub_3[idx]["feature"]
            sim = self.get_cosine_sim(sub_feature,main_feature)
            if sim > closest_sub_3_val:
                closest_sub_3_val = sim
                closest_sub_3_idx = idx

        return (main_idx,closest_sub_1_idx,closest_sub_2_idx,closest_sub_3_idx)