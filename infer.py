"""
 @Time    : 2021/7/6 14:36
 @Author  : Haiyang Mei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : CVPR2021_PFNet
 @File    : infer.py
 @Function: Inference
 
"""
import time
import datetime

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from collections import OrderedDict
from numpy import mean

from config import *
from misc import *
from PFNet import PFNet

torch.manual_seed(2021)
device_ids = [1]

def eval_mae(y_pred, y):
    """
    evaluate MAE (for test or validation phase)
    :param y_pred:
    :param y:
    :return: Mean Absolute Error
    """
    return torch.abs(y_pred - y).mean()


def numpy2tensor(numpy):
    """
    convert numpy_array in cpu to tensor in gpu
    :param numpy:
    :return: torch.from_numpy(numpy).cuda()
    """
    return torch.from_numpy(numpy).cuda()

results_path = 'D:/Olia/結構化機器學習/CVPR2021_PFNet-main/results'
check_mkdir(results_path)
exp_name = 'PFNet'
args = {
    'scale': 416,
    'save_results': True
}

print(torch.__version__)

img_transform = transforms.Compose([
    transforms.Resize((args['scale'], args['scale'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

to_pil = transforms.ToPILImage()

to_test = OrderedDict([
                       ('CHAMELEON', chameleon_path),
                       ('CAMO', camo_path),
                       ('COD10K', cod10k_path)
                       ])

results = OrderedDict()

def main():
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device_ids = list(range(torch.cuda.device_count()))
        device = torch.device(f'cuda:{device_ids[0]}')
    else:
        device_ids = []
        device = torch.device('cpu')

    net = PFNet(backbone_path).to(device)
    net.load_state_dict(torch.load('D:/Olia/結構化機器學習/CVPR2021_PFNet-main/ckpt/PFNet/45.pth'))
    print('Model loaded successfully.')

    net.eval()

    dataset_mae = {}  # 儲存各資料集的平均 MAE
    all_mae = []      # 儲存所有 MAE 數值（計算整體平均）

    with torch.no_grad():
        start = time.time()

        for name, root in to_test.items():
            time_list = []
            mae_list = []

            image_path = os.path.join(root, 'Imgs')
            gt_path_base = os.path.join(root, 'GT')

            if args['save_results']:
                check_mkdir(os.path.join(results_path, exp_name, name))

            img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith('jpg')]
            for idx, img_name in enumerate(img_list):
                img = Image.open(os.path.join(image_path, img_name + '.jpg')).convert('RGB')
                w, h = img.size
                img_var = img_transform(img).unsqueeze(0).to(device)

                start_each = time.time()
                _, _, _, prediction = net(img_var)
                time_each = time.time() - start_each
                time_list.append(time_each)

                # Ground truth
                gt_path = os.path.join(gt_path_base, img_name + '.png')
                gt = Image.open(gt_path).convert('L')
                gt_np = np.array(gt).astype(np.float32)
                gt_np /= (gt_np.max() + 1e-8)

                # Resize prediction
                prediction_resized = np.array(transforms.Resize(gt_np.shape)(to_pil(prediction.data.squeeze(0).cpu())))
                prediction_norm = (prediction_resized - prediction_resized.min()) / (prediction_resized.max() - prediction_resized.min() + 1e-8)

                prediction_tensor = numpy2tensor(prediction_norm)
                gt_tensor = numpy2tensor(gt_np)

                mae = eval_mae(prediction_tensor, gt_tensor)
                mae_list.append(mae.item())

                if args['save_results']:
                    Image.fromarray(prediction_resized.astype(np.uint8)).convert('L').save(
                        os.path.join(results_path, exp_name, name, img_name + '.png'))

            # 儲存該資料集的平均 MAE
            avg_mae = np.mean(mae_list)
            dataset_mae[name] = avg_mae
            all_mae.extend(mae_list)

        end = time.time()
        print("\n===== [Inference Complete] =====")
        print("Total Testing Time: {}".format(str(datetime.timedelta(seconds=int(end - start)))))

        # 統一列出所有資料集的 MAE
        print("\n===== [MAE Summary] =====")
        for dataset, mae_val in dataset_mae.items():
            print("Dataset: {:10s} | Average MAE: {:.4f}".format(dataset, mae_val))

        if all_mae:
            print("-----------------------------")
            print("All Dataset Average MAE: {:.4f}".format(np.mean(all_mae)))
            print("=============================\n")



if __name__ == '__main__':
    main()