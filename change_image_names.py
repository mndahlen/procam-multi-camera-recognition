import os

directory = "data\hallway_persons_0\person7"
idx = 0
for filename in os.listdir(directory):
    if filename.endswith(".png"): 
            os.rename(os.path.join(directory, filename), os.path.join(directory, "p7_{}.png".format(idx)))
            print(os.path.join(directory, filename))
            idx += 1
