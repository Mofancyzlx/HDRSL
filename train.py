import cv2
import os
import torch
import logging
import wandb
import argparse

from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from loss import FocalLoss, SSIM, MseDirectionLoss
from tqdm import tqdm

from model_with_attention import ReconstructiveSubNetwork_CBAMattention, ReconstructiveSubNetwork_Spattention
from model_with_attention import ReconstructiveSubNetwork_Chattention, ReconstructiveSubNetwork_Spattention_only_encoder
from ResUNet import ResUNet
from unet import UNet, UNet_attention

from evaluate import evaluate
from utils.data_loading import BasicDataset, BasicDataset_High_Reflect

import warnings 
warnings.filterwarnings('ignore')

# 离线运行
os.environ['WANDB_MODE'] = 'dryrun'

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train_on_device(
        args, 
        model_student,  # 学生模型
        model_teacher,  # 教师模型
        model_result,   # 结果模型
        device,         # 训练设备
        epochs: int = 5,            # 训练轮数
        batch_size: int = 1,        # 批次大小
        learning_rate: float = 1e-5, # 学习率
        val_percent: float = 0.1    # 验证集比例
 ):

    # create dataset
    dir_img_plat = '/home/wangh20/data/structure/metal_dataset/qiu_plat_fenzi_fenmu/plat/images_GT'
    dir_mask_plat = '/home/wangh20/data/structure/metal_dataset/qiu_plat_fenzi_fenmu/plat'
    dir_img_qiu1 = '/home/wangh20/data/structure/metal_dataset/qiu_plat_fenzi_fenmu/qiu_1/images_GT'
    dir_mask_qiu1 = '/home/wangh20/data/structure/metal_dataset/qiu_plat_fenzi_fenmu/qiu_1'
    dir_img_qiu2 = '/home/wangh20/data/structure/metal_dataset/qiu_plat_fenzi_fenmu/qiu_2/images_GT'
    dir_mask_qiu2 = '/home/wangh20/data/structure/metal_dataset/qiu_plat_fenzi_fenmu/qiu_2'
    dir_img_jieti = '/home/wangh20/data/structure/metal_dataset/qiu_plat_fenzi_fenmu/jieti/images_GT'
    dir_mask_jieti = '/home/wangh20/data/structure/metal_dataset/qiu_plat_fenzi_fenmu/jieti'
    try:
        metal_dataset = BasicDataset_High_Reflect(args.dir_img, args.dir_mask)
        dataset_plat = BasicDataset_High_Reflect(dir_img_plat, dir_mask_plat)
        dataset_qiu1 = BasicDataset_High_Reflect(dir_img_qiu1, dir_mask_qiu1)
        dataset_qiu2 = BasicDataset_High_Reflect(dir_img_qiu2, dir_mask_qiu2)
        dataset_jieti = BasicDataset_High_Reflect(dir_img_jieti, dir_mask_jieti)
    except (AssertionError, RuntimeError, IndexError):
        metal_dataset = BasicDataset_High_Reflect(args.dir_img, args.dir_mask)
        dataset_plat = BasicDataset_High_Reflect(dir_img_plat, dir_mask_plat)
        dataset_qiu1 = BasicDataset_High_Reflect(dir_img_qiu1, dir_mask_qiu1)
        dataset_qiu2 = BasicDataset_High_Reflect(dir_img_qiu2, dir_mask_qiu2)
        dataset_jieti = BasicDataset_High_Reflect(dir_img_jieti, dir_mask_jieti)
    
    n_train = int(len(metal_dataset) * 0.8)
    n_val = int(len(metal_dataset) * 0.1)
    n_test = len(metal_dataset) - n_val - n_train
    
    train_set, val_set, test_set = random_split(metal_dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(0)) 

    train_set = torch.utils.data.ConcatDataset([train_set, dataset_plat, dataset_qiu1, dataset_qiu2, dataset_jieti])
    logging.info(f'Number of training images: {len(train_set)}')
    logging.info(f'Number of validation images: {len(val_set)}')
    logging.info(f'Number of validation images: {len(test_set)}')

    experiment = wandb.init(project='HDR_fenzi_fenmu', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=args.save_checkpoint_path)
    )
    logging.info(f'''Starting training:
    Epochs:          {epochs}
    Batch size:      {batch_size}
    Learning rate:   {learning_rate}
    Training size:   {n_train}
    Validation size: {n_val}
    Device:          {device.type}
    ''')    

    # create dataloader
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=8)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=2, drop_last=False, num_workers=8)
    test_loder = DataLoader(test_set, shuffle=False, batch_size=2, drop_last=False, num_workers=8)
    optimizer = torch.optim.Adam([{"params": model_student.parameters(), "lr": learning_rate},
                                  {"params": model_result.parameters(), "lr": learning_rate}])
    optimizer_teacher = torch.optim.Adam(model_teacher.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs, eta_min=1e-06, last_epoch=-1)
    scheduler_teacher = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_teacher, T_max = epochs, eta_min=1e-06, last_epoch=-1)

    # define loss function
    criterion_MSE = torch.nn.MSELoss()
    criterion_MAE = torch.nn.L1Loss()
    criterion_SSIM = SSIM(device=device)
    Distillation_loss = MseDirectionLoss(0.1)
    global_step = 0
    # begin train
    best_loss = 10000
    for epoch in range(1, epochs + 1):
        model_student.train()
        model_teacher.train()
        model_result.train()
        if epoch <= 10:
            for i in tqdm(range(3)):
                for batch in train_loader:
                    imgs = batch['image'].to(device)
                    teacher_rec, _ = model_teacher(imgs) # HDR GT, shape [B, 4, H, W] 的图片喂入 teacher
                    l2_loss_teacher = torch.sqrt(criterion_MSE(teacher_rec, imgs)) # HDR GT 既是 teacher 的输入，也是 teacher 的输出
                    l1_loss_teacher = criterion_MAE(teacher_rec, imgs)
                    ssim_teacher = criterion_SSIM(teacher_rec, imgs)
                    loss_teacher = l2_loss_teacher + l1_loss_teacher + ssim_teacher
                    optimizer_teacher.zero_grad()
                    loss_teacher.backward()
                    optimizer_teacher.step()
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                imgs_aug = batch['image_aug'] # 4 channel LDR images, shape [B, 4, H, W], 这里似乎只使用了低曝光的四个频率分量的图片
                # 判断输入是不是灰度图
                assert imgs.shape[1] == 4, f'Network has been designed to work with 1 channel images, but received {imgs.shape[1]} channels'
                assert imgs_aug.shape[1] == 4, f'Network has been designed to work with 1 channel images, but received {imgs.shape[1]} channels'
                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                imgs_aug = imgs_aug.to(device=device, dtype=torch.float32)

                pred, fea_student = model_student(imgs_aug)
                pred_teacher, fea_teacher = model_teacher(imgs)
                joined_in = torch.cat([pred, imgs_aug], dim=1)

                out, _ = model_result(joined_in)
                # student loss
                l2_loss_student = torch.sqrt(criterion_MSE(pred, imgs))
                l1_loss_student = criterion_MAE(pred, imgs)
                ssim_student = criterion_SSIM(pred, imgs)
                distillation_loss = Distillation_loss(fea_student, fea_teacher)
                loss_student = l2_loss_student + l1_loss_student + ssim_student + distillation_loss
                # teacher loss 
                l2_loss_teacher = torch.sqrt(criterion_MSE(pred_teacher, imgs))
                l1_loss_teacher = criterion_MAE(pred_teacher, imgs)
                ssim_teacher = criterion_SSIM(pred_teacher, imgs)
                loss_teacher = l2_loss_teacher + l1_loss_teacher + ssim_teacher
                # result loss
                l2_loss_result =  torch.sqrt(criterion_MSE(out, true_masks))
                l1_loss_result = criterion_MAE(out, true_masks)
                loss_result = l2_loss_result + l1_loss_result # loss_results 中由于 out 带有 pred 的信息，所以 loss_result 不只是训练 Phase Net，它还会通过 pred 反向传播到 Student

                #update 
                if epoch > 10:
                    loss = loss_student + loss_result + loss_teacher
                    optimizer.zero_grad()
                    optimizer_teacher.zero_grad()
                    loss.backward()
                    optimizer.step()
                    optimizer_teacher.step()
                else:
                    loss = loss_student + loss_result
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(imgs.shape[0])
                global_step += 1
                experiment.log({
                'train loss': loss.item(),
                'l1_loss_student': l1_loss_student.item(),
                'l2_loss_student': l2_loss_student.item(),
                'ssim_student': ssim_student.item(),
                'distillation_loss': distillation_loss.item(),   
                'loss_student': loss_student.item(),

                'l1_loss_result': l1_loss_result.item(),
                'l2_loss_result': l2_loss_result.item(),
                'loss_result': loss_result.item(),
                'step': global_step,
                'epoch': epoch
                })
            # evaluate the model on the validation set

            histograms = {}
            for tag, value in model_result.named_parameters():
                tag = tag.replace('/', '.')
                if not (torch.isinf(value) | torch.isnan(value)).any():
                    histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                    histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
            val_score_MSE, val_score_MAE = evaluate(model_student, model_result, test_loder, device)
            val_loss = val_score_MAE

            try:
                experiment.log({
                    'learning rate': optimizer.param_groups[0]['lr'],
                    'validation L2_Score': val_score_MSE,
                    'validation L1_Score': val_score_MAE,
                    'images': wandb.Image(imgs[0].cpu()),
                    'masks': {
                        'true': wandb.Image(true_masks[0][3].float().cpu()),
                        'pred': wandb.Image(out[0][3].float().cpu()),
                    },
                    'step': global_step,
                    'epoch': epoch,
                })
            except:
                pass
            # 保存最好的epoch
            logging.info(f'min_loss_current: {round(best_loss, 4)} | {round(val_loss.item(), 4)}')
            if val_loss.item() < best_loss:
                best_loss = val_loss.item()
                save_checkpoint_path = args.save_checkpoint_path
                os.makedirs(save_checkpoint_path, exist_ok=True)
                state_dict_teacher = model_teacher.state_dict()
                state_dict_student = model_student.state_dict()
                state_dict_result = model_result.state_dict()
                torch.save(state_dict_teacher, os.path.join(save_checkpoint_path, f'teacher.pth'))# state_dict['mask_va                lues'] = dataset.mask_values
                torch.save(state_dict_student, os.path.join(save_checkpoint_path, f'student.pth'))
                torch.save(state_dict_result, os.path.join(save_checkpoint_path, f'result.pth'))  # state_dict['mask_values'] = dataset.mask_values
                logging.info(f'Best model saved at epoch {epoch}')
        scheduler.step()
        scheduler_teacher.step()
        
def get_args():
    parser = argparse.ArgumentParser(description='Train the DSAR model on images and target masks')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--validation', type=int, default=10)
    parser.add_argument('--dir_img', type=str, required=True)
    parser.add_argument('--dir_mask', type=str, required=True)
    parser.add_argument('--save_checkpoint_path', type=str, required=True)
    parser.add_argument('--load_checkpoint', type=bool, default=False)
    return parser.parse_args()

if __name__ == "__main__":
    
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    # train model 

    model_student = UNet_attention(4, 4, False)
    model_teacher = UNet(4, 4, False)
    model_result = UNet(8, 8, False)

    if args.load_checkpoint:
        model_student.load_state_dict(torch.load(os.path.join(args.load_checkpoint, 'student_checkpoint.pth')))
        model_teacher.load_state_dict(torch.load(os.path.join(args.load_checkpoint, 'teacher_checkpoint.pth')))
        model_result.load_state_dict(torch.load(os.path.join(args.load_checkpoint, 'result_checkpoint.pth')))
        logging.info(f'Model loaded from {args.load_checkpoint}')
    else:
        logging.info('Starting training from scratch')
    
    model_student.to(device)
    model_teacher.to(device)
    model_result.to(device)


    try:
        train_on_device(args, 
                        model_student=model_student,
                        model_teacher=model_teacher,
                        model_result=model_result, 
                        device=device,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        learning_rate=args.learning_rate,
                        val_percent=args.validation / 100
                        )
        
    except torch.cuda.OutofMemoryError:
        logging.error('train_error! please check')
        torch.cuda.empty_cache()
        exit(0)
