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
PERSONSDIR = "procam-reid/persons_detected"

# Model
yolo = models.resnet18(pretrained=True)

# Persons
person_1 = Image.open(os.path.join(PERSONSDIR,"person_0_143.png"))
person_2 = Image.open(os.path.join(PERSONSDIR,"person_0.png"))

# Copied from https://becominghuman.ai/extract-a-feature-vector-for-any-image-with-pytorch-9717561d1d4c
# Load the pretrained model
model = models.resnet18(pretrained=True)
# Use the model object to select the desired layer
print(model._modules)
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

embedding_1 = get_vector(person_1).flatten().numpy()
embedding_2 = get_vector(person_2).flatten().numpy()
lower_limit = 1e-6
norm_product = max(np.linalg.norm(embedding_1)*np.linalg.norm(embedding_2),lower_limit)

cosinesim = np.dot(embedding_1,embedding_2)/norm_product

print(cosinesim)
if 0:
    # Using PyTorch Cosine Similarity
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    cos_sim = cos(embedding_1,
                embedding_2)
    print('\nCosine similarity: {0}\n'.format(cos_sim))

    #print(embedding_1.flatten().shape)