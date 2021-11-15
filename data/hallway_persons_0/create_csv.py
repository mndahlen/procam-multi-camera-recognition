import pandas as pd
from torch.utils import data

datalabels = pd.DataFrame(columns=["imname","label"])

person = 1
for i in range(0,140 + 1):
    datalabels = datalabels.append({'imname': "p{}_{}.png".format(person,i), 'label':person}, ignore_index = True)

person = 2
for i in range(0,129 + 1):
    datalabels = datalabels.append({'imname': "p{}_{}.png".format(person,i), 'label':person}, ignore_index = True)

person = 3
for i in range(0,126 + 1):
    datalabels = datalabels.append({'imname': "p{}_{}.png".format(person,i), 'label':person}, ignore_index = True)

person = 4
for i in range(0,111 + 1):
    datalabels = datalabels.append({'imname': "p{}_{}.png".format(person,i), 'label':person}, ignore_index = True)

person = 5
for i in range(0,80 + 1):
    datalabels = datalabels.append({'imname': "p{}_{}.png".format(person,i), 'label':person}, ignore_index = True)

person = 6
for i in range(0,29 + 1):
    datalabels = datalabels.append({'imname': "p{}_{}.png".format(person,i), 'label':person}, ignore_index = True)

person = 7
for i in range(0,17 + 1):
    datalabels = datalabels.append({'imname': "p{}_{}.png".format(person,i), 'label':person}, ignore_index = True)

datalabels.to_csv("data_labels.csv", index=False)