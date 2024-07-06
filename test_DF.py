from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score


# import subprocess

# print("Running preprocess_dataset.py ...")
# subprocess.run(["python", "preprocess_dataset1.py"])
# print("preprocess_dataset.py finished.")



def auc(real, fake):
    label_all = []
    target_all = []
    for ind in real:
        target_all.append(1)
        label_all.append(-ind)
    for ind in fake:
        target_all.append(0)
        label_all.append(-ind)
    # print("length: target = ",len(target_all)," label = ",len(label_all))

    # from sklearn.metrics import roc_auc_score
    # return roc_auc_score(target_all, label_all)
    
    print("length: target = ",len(target_all)," label = ",len(label_all))
    # print("target = ",target_all," label = ",label_all)
    return roc_auc_score(target_all, label_all)


def compute_accuracy(real, fake, threshold= 1451.34):
    # Convert distances to predicted labels
    predicted_labels = [0 if x >= threshold else 1 for x in real + fake]
    # Actual labels are 1 for 'real' and 0 for 'fake'
    actual_labels = [1] * len(real) + [0] * len(fake)
    return accuracy_score(actual_labels, predicted_labels)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.num_threads = 4
    opt.batch_size = 1
    opt.serial_batches = False
    opt.no_flip = True
    opt.display_id = -1
    opt.mode = 'test'
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)

    model.eval()

    dataset_size = len(dataset)
    print('The number of test images dir = %d' % dataset_size)

    total_iters = 0
    label = None
    real = []
    fake = []

    with tqdm(total=dataset_size) as pbar:
        for i, data in enumerate(dataset):
            input_data = {'img_real': data['img_real'],
                          'img_fake': data['img_fake'],
                          'aud_real': data['aud_real'],
                          'aud_fake': data['aud_fake'],
                          }
            model.set_input(input_data)

            dist_AV, dist_VA = model.val()
            real.append(dist_AV.item())
            for i in dist_VA:
                fake.append(i.item())
            total_iters += 1
            pbar.update()

    # print('The auc is %.3f'%(auc(real, fake)))
    auc_score = auc(real, fake)
    accuracy = compute_accuracy(real, fake)
    print('The AUC is %.3f' % auc_score)
    print('The accuracy is %.3f' % accuracy)
