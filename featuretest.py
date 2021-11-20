import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
from PIL import Image
import numpy as np
import os

# Constants
HALLWAYDIR = "data/hallway"
PERSONSDIR = "data/feature_testing"

# Model
yolo = models.resnet18(pretrained=True)

# Persons
person_1_1 = Image.open(os.path.join(PERSONSDIR,"person_1_1.png"))
person_1_2 = Image.open(os.path.join(PERSONSDIR,"person_1_2.png"))
person_2_1 = Image.open(os.path.join(PERSONSDIR,"person_2_1.png"))
person_2_2 = Image.open(os.path.join(PERSONSDIR,"person_2_2.png"))
person_3_1 = Image.open(os.path.join(PERSONSDIR,"person_3_1.png"))
person_3_2 = Image.open(os.path.join(PERSONSDIR,"person_3_2.png"))
person_4_1 = Image.open(os.path.join(PERSONSDIR,"person_4_1.png"))
person_4_2 = Image.open(os.path.join(PERSONSDIR,"person_4_2.png"))
person_5_1 = Image.open(os.path.join(PERSONSDIR,"person_5_1.png"))
person_5_2 = Image.open(os.path.join(PERSONSDIR,"person_5_2.png"))
person_6_1 = Image.open(os.path.join(PERSONSDIR,"person_6_1.png"))
person_6_2 = Image.open(os.path.join(PERSONSDIR,"person_6_2.png"))
person_7_1 = Image.open(os.path.join(PERSONSDIR,"person_7_1.png"))
person_7_2 = Image.open(os.path.join(PERSONSDIR,"person_7_2.png"))
person_8_1 = Image.open(os.path.join(PERSONSDIR,"person_8_1.png"))
person_8_2 = Image.open(os.path.join(PERSONSDIR,"person_8_2.png"))
person_9_1 = Image.open(os.path.join(PERSONSDIR,"person_9_1.png"))
person_9_2 = Image.open(os.path.join(PERSONSDIR,"person_9_2.png"))
person_10_1 = Image.open(os.path.join(PERSONSDIR,"person_10_1.png"))
person_10_2 = Image.open(os.path.join(PERSONSDIR,"person_10_2.png"))

# Hard examples
person_1_hard = Image.open(os.path.join(PERSONSDIR,"person_1_hard.png"))
person_2_hard = Image.open(os.path.join(PERSONSDIR,"person_2_hard.png"))
person_3_hard = Image.open(os.path.join(PERSONSDIR,"person_3_hard.png"))
person_4_hard = Image.open(os.path.join(PERSONSDIR,"person_4_hard.png"))
person_5_hard = Image.open(os.path.join(PERSONSDIR,"person_5_hard.png"))

# Copied from https://becominghuman.ai/extract-a-feature-vector-for-any-image-with-pytorch-9717561d1d4c
# Load the pretrained model
model = models.resnet18(pretrained=True)
model = torch.load("models/resnet_hallway_639_1_10.tar")

# Use the model object to select the desired layer
#print(model._modules)
layer = model._modules.get('avgpool')
# Set model to evaluation mode
model.eval()
scaler = transforms.Scale((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

def get_vector(img):
    # 1. Create a PyTorch Variable with the transformed image
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
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
    h = layer.register_forward_hook(copy_data)
    # 5. Run the model on our transformed image
    model(t_img)
    # 6. Detach our copy function from the layer
    h.remove()
    # 7. Return the feature vector
    return my_embedding

def get_cosine_sim(v1,v2):
    lower_limit = 1e-6
    norm_product = max(np.linalg.norm(v1)*np.linalg.norm(v2),lower_limit)
    cosinesim = np.dot(v1,v2)/norm_product
    return  cosinesim

# Get embeddings for all person images
embedding_1_1 = get_vector(person_1_1).flatten().numpy()
embedding_1_2 = get_vector(person_1_2).flatten().numpy()
embedding_2_1 = get_vector(person_2_1).flatten().numpy()
embedding_2_2 = get_vector(person_2_2).flatten().numpy()
embedding_3_1 = get_vector(person_3_1).flatten().numpy()
embedding_3_2 = get_vector(person_3_2).flatten().numpy()
embedding_4_1 = get_vector(person_4_1).flatten().numpy()
embedding_4_2 = get_vector(person_4_2).flatten().numpy()
embedding_5_1 = get_vector(person_5_1).flatten().numpy()
embedding_5_2 = get_vector(person_5_2).flatten().numpy()
embedding_6_1 = get_vector(person_6_1).flatten().numpy()
embedding_6_2 = get_vector(person_6_2).flatten().numpy()
embedding_7_1 = get_vector(person_7_1).flatten().numpy()
embedding_7_2 = get_vector(person_7_2).flatten().numpy()
embedding_8_1 = get_vector(person_8_1).flatten().numpy()
embedding_8_2 = get_vector(person_8_2).flatten().numpy()
embedding_9_1 = get_vector(person_9_1).flatten().numpy()
embedding_9_2 = get_vector(person_9_2).flatten().numpy()
embedding_10_1 = get_vector(person_10_1).flatten().numpy()
embedding_10_2 = get_vector(person_10_2).flatten().numpy()

embeddings = [embedding_1_1,embedding_1_2,embedding_2_1,embedding_2_2,embedding_3_1,embedding_3_2,
              embedding_4_1,embedding_4_2,embedding_5_1,embedding_5_2,embedding_6_1,embedding_6_2,
              embedding_7_1,embedding_7_2,embedding_8_1,embedding_8_2,embedding_9_1,embedding_9_2,
              embedding_10_1,embedding_10_2]
#
sim_matrix = np.zeros((20,20))

i = 0
j = 1
for emb_1 in embeddings:
    rest = embeddings[j:]
    col_idx = j
    for emb_2 in rest:
        sim_matrix[i,col_idx] = get_cosine_sim(emb_1,emb_2)
        col_idx += 1
    j += 1
    i += 1

print(sim_matrix)