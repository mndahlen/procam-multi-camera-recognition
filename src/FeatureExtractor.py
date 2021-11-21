import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image

class FeatureExtractor(object):
    def __init__(self):
        self.resnet = torch.load("../models/resnet18_hallway_1192_3_20.tar")
        self.layer = self.resnet._modules.get('avgpool')
        self.resnet.eval()
        self.scaler = transforms.Scale((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    def get_feature_embedding(self,img):
        # 1. Create a PyTorch Variable with the transformed image
        t_img = Variable(self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0))
        # 2. Create a vector of zeros that will hold our feature vector
        #    The 'avgpool' layer has an output size of 512
        my_embedding = torch.zeros([1,512,1,1])
        test = None
        # 3. Define a function that will copy the output of a layer
        def copy_data(m, i, o):
            #print(o.data.shape)
            #test = o.data
            my_embedding.copy_(o.data)
        # 4. Attach that function to our selected layer
        h = self.layer.register_forward_hook(copy_data)
        # 5. Run the model on our transformed image
        self.resnet(t_img)
        # 6. Detach our copy function from the layer
        h.remove()
        # 7. Return the feature vector
        return my_embedding

    def get_cam_persons(self,bbox,im,persons,T):
        num_detected_persons = bbox.shape[0]
        persons_empty = False

        if not persons:
            persons_empty = True
            persons = {}    

        for i in range(0,num_detected_persons):
            b = bbox[i]
            x1 = int(b[0])
            y1 = int(b[1])
            x2 = int(b[2])
            y2 = int(b[3])
            person_im = im[y1:y2,x1:x2] 
            feature = self.get_feature_embedding(Image.fromarray(person_im)).flatten().numpy()
            
            if persons_empty:
                persons[i] = {"feature":0, "bbox":0, "history":1}
                persons[i]["feature"] = feature
                persons[i]["bbox"] = b
            else:
                max_id = 0
                max_sim = 0
                for person_id in persons:
                    prev_frame_feature = persons[person_id]["feature"]
                    sim = self.get_cosine_sim(feature,prev_frame_feature)
                    if sim >= max_sim:
                        max_sim = sim
                        max_id = person_id
                if max_sim >= T:
                    history = persons[max_id]["history"]
                    # Calculate mean for all time
                    combined_features = (history*prev_frame_feature + feature)/(history + 1)
                    persons[max_id]["feature"] = combined_features
                    persons[max_id]["bbox"] = b
                    persons[max_id]["history"] = history + 1
                else:
                    new_id = max(persons.keys()) + i + 1
                    persons[new_id] = {"feature":0, "bbox":0, "history":1}
                    persons[new_id]["feature"] = feature
                    persons[new_id]["bbox"] = b
        return persons

    def get_cosine_sim(self,v1,v2):
        lower_limit = 1e-6
        norm_product = max(np.linalg.norm(v1)*np.linalg.norm(v2),lower_limit)
        cosinesim = np.dot(v1,v2)/norm_product
        return  cosinesim
