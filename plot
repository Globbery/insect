import matplotlib.pyplot as plt
import pandas as pd



name = 'shuffflenet'
logs = pd.read_csv(f'{name}.log')

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(logs['loss'], label='train')
plt.plot(logs['test_loss'], label='val')
plt.legend()
plt.title('loss')
plt.xlabel('epoch')

plt.subplot(1, 2, 2)
plt.plot(logs['acc'], label='train')
plt.plot(logs['test_acc'], label='val')
plt.legend()
plt.title('acc')
plt.xlabel('epoch')

plt.tight_layout()
plt.savefig(f'{name}_curve.png')




import os, cv2, torch, itertools, tqdm
import time

import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image
import torch.nn as nn

def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues, name='test', save_name=None):
    plt.figure(figsize=(11, 11))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    trained_classes = classes
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(name + title, fontsize=18)
    tick_marks = np.arange(len(classes))
    plt.xticks(np.arange(len(trained_classes)), classes, rotation=90, fontsize=14)
    plt.yticks(tick_marks, classes, fontsize=14)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, np.round(cm[i, j], 2), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=15)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.gcf().subplots_adjust(left=0.2)
    plt.savefig(f"{save_name}_cm.png", dpi=150)
    return cm

data_transforms = {
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)

    with open('classes.txt') as f:
        label = list(map(lambda x:x.strip(), f.readlines()))
        print(label)

    name = 'mobilenetv3_2'
    model = torch.load(f'{name}.pt').to(DEVICE)

    y_pred, y_true = [], []
    for i in tqdm.tqdm(os.listdir('test')):
        base_path = os.path.join('test', i)
        for j in os.listdir(base_path):
            img = Image.open(os.path.join(base_path, j)).convert('RGB')
            img = data_transforms['test'](img) # 3*224*224
            img = torch.unsqueeze(img, dim=0).to(DEVICE).float() # bs*3*224*224 -> 1*3*224*224
            pred = np.argmax(model(img).cpu().detach().numpy()[0])
            y_pred.append(pred)
            y_true.append(int(i))

    y_pred, y_true = np.array(y_pred), np.array(y_true)
    print(classification_report(y_true, y_pred, target_names=label))
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, classes=label, save_name=name)
    plt.show()
