import os
import random

import wandb 
wandb_on = True
import cv2
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import CTCLoss, NLLLoss, MSELoss
import torch.nn.functional as F

from dataset import Synth90kDataset, synth90k_collate_fn, ICDAR13Dataset, icdar13_collate_fn, IIIT5KDataset, iiit5k_collate_fn, CocoTextV2Dataset, cocotextv2_collate_fn
from model import CRNN, CRNN_small
from evaluate import evaluate
from config import distill_config as config


def distill_batch(crnn, crnn_teacher, data, optimizer, criterion, device):
    crnn.train()
    crnn_teacher.eval()
    images, targets, target_lengths = [d.to(device) for d in data]

    logits_student = crnn(images)
    logits_teacher = crnn_teacher(images)
    #print(targets)
    
    batch_size = images.size(0)
    if config['distil_mode'] == "hard":
        loss = teacher_as_label(logits_student, logits_teacher, 0.5, batch_size)
    elif config['distil_mode'] == "soft":
        loss = calculate_kd_loss(logits_student, logits_teacher, targets, target_lengths, criterion, batch_size, distil_weight=config["distil_weight"], temp=config["temperature"])
    else:
        raise ValueError

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(crnn.parameters(), 5) # gradient clipping with 5
    optimizer.step()
    return loss.item()
    

def teacher_as_label(student_out, teacher_out, temp, batch_size):
    log_probs_student = torch.nn.functional.log_softmax(student_out, dim=2)
    log_probs_teacher = torch.nn.functional.log_softmax(teacher_out, dim=2)
    student_reshaped = log_probs_student.permute(1, 0, 2).reshape(-1, log_probs_student.size(2))
    teacher_reshaped = log_probs_teacher.permute(1, 0, 2).reshape(-1, log_probs_teacher.size(2))

    teacher_targets = teacher_reshaped.argmax(dim=1)

    teachercrit = NLLLoss()
    # Compute the negative log-likelihood loss
    loss = teachercrit(student_reshaped, teacher_targets) * temp
    return loss
    
  

def calculate_kd_loss(student_out, teacher_out, targets, target_lengths, criterion, batch_size, distil_weight=0.8, temp=2.0):
    log_probs_student = F.log_softmax(student_out / temp, dim=2)
    log_probs_teacher = F.log_softmax(teacher_out / temp, dim=2)
    input_lengths = torch.LongTensor([student_out.size(0)] * batch_size)
    target_lengths = torch.flatten(target_lengths)
    teachercrit = MSELoss()
    
    loss = (1 - distil_weight) * criterion(log_probs_student, targets, input_lengths, target_lengths)
    loss += (distil_weight * temp * temp) * teachercrit(log_probs_student, log_probs_teacher)
    return loss


def main(gridconfig = None):
    if gridconfig:
        config.update(gridconfig)
        print("Config was changed to: \n", gridconfig)
    epochs = config['epochs']
    train_batch_size = config['train_batch_size']
    eval_batch_size = config['eval_batch_size']
    lr = config['lr']
    show_interval = config['show_interval']
    valid_interval = config['valid_interval']
    save_interval = config['save_interval']
    cpu_workers = config['cpu_workers']
    checkpoints_dir = config['checkpoints_dir']
    reload_checkpoint = config['reload_checkpoint']
    teacher = config['teacher']
    valid_max_iter = config['valid_max_iter']
    img_width = config['img_width']
    dataset = config['dataset']
    img_height = config['img_height']
    data_dir = config['data_dir']
    torch.manual_seed(config['seed']) # Seed for Weight Initialization
    random.seed(config['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    if dataset == "Synth90k":
        train_dataset = Synth90kDataset(root_dir=data_dir, mode='train',
                                        img_height=img_height, img_width=img_width)
        valid_dataset = Synth90kDataset(root_dir=data_dir, mode='dev',
                                        img_height=img_height, img_width=img_width)
        collate_fn = synth90k_collate_fn
    elif dataset == "ICDAR2013":
        train_dataset = ICDAR13Dataset(root_dir=data_dir, mode='train',
                                        img_height=img_height, img_width=img_width)
        valid_dataset = ICDAR13Dataset(root_dir=data_dir, mode='dev',
                                        img_height=img_height, img_width=img_width)
        collate_fn = icdar13_collate_fn
    elif dataset == "IIIT5K":
        train_dataset = IIIT5KDataset(root_dir=data_dir, mode='train',
                                        img_height=img_height, img_width=img_width)
        valid_dataset = IIIT5KDataset(root_dir=data_dir, mode='dev',
                                        img_height=img_height, img_width=img_width)
        collate_fn = iiit5k_collate_fn
    elif dataset == "CocoTextV2":
        train_dataset = CocoTextV2Dataset(root_dir=data_dir, mode='train',
                                        img_height=img_height, img_width=img_width)
        valid_dataset = CocoTextV2Dataset(root_dir=data_dir, mode='dev',
                                        img_height=img_height, img_width=img_width)
        collate_fn = cocotextv2_collate_fn
    else:
        raise ValueError
        
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=cpu_workers,
        collate_fn=collate_fn)
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=eval_batch_size,
        shuffle=True,
        num_workers=cpu_workers,
        collate_fn=collate_fn)
    
    num_class = len(Synth90kDataset.LABEL2CHAR) + 1
                
    crnn = CRNN_small(config['channels'], img_height, img_width, num_class,
                map_to_seq_hidden=config['map_to_seq_hidden'],
                rnn_hidden=config['rnn_hidden'],
                leaky_relu=config['leaky_relu'], dropout=config['dropout'])
    if reload_checkpoint:
        crnn.load_state_dict(torch.load(os.path.join(checkpoints_dir, reload_checkpoint), map_location=device))
    if teacher:
        crnn_teacher = CRNN(config['channels'], img_height, img_width, num_class,
                map_to_seq_hidden=config['map_to_seq_hidden_teacher'],
                rnn_hidden=config['rnn_hidden_teacher'],
                leaky_relu=config['leaky_relu'], dropout=config['dropout'])
        crnn_teacher.load_state_dict(torch.load(os.path.join(checkpoints_dir, teacher), map_location=device))
        crnn_teacher.to(device)
        crnn_teacher.eval()
        
    crnn.to(device)
    crnn.train()

    optimizer = optim.Adam(crnn.parameters(), lr=lr) #optim.RMSprop(crnn.parameters(), lr=lr)
    criterion = CTCLoss(reduction='sum', zero_infinity=True)
    criterion.to(device)
    # WandDB Logging
    if wandb_on:
        wandb.login()
        run = wandb.init(
          project="CRNN-Network", 
          reinit=gridconfig is not None,
          config={
          "name": config['name'],
          "seed": config['seed'],
          "learning_rate": lr,
          "train_batch_size": config['train_batch_size'],
          "dropout": config['dropout'],
          "architecture": "CRNN - Distillation",
          "dataset": config['dataset'],
          "epochs": epochs,
          "distil_weight": config["distil_weight"],
          "temperature": config["temperature"],
          "map_to_seq_hidden": config['map_to_seq_hidden'],
          "rnn_hidden": config['rnn_hidden'],
          "map_to_seq_hidden_teacher": config['map_to_seq_hidden_teacher'],
          "rnn_hidden_teacher": config['rnn_hidden_teacher'],
          "student": config['reload_checkpoint'],
          "teacher": config['teacher'],
          "epochs": epochs,
          "distil_mode": config['distil_mode'],
          })
    # End
    assert save_interval % valid_interval == 0
    i = 1
    for epoch in range(1, epochs + 1):
        print(f'epoch: {epoch}')
        tot_train_loss = 0.
        tot_train_count = 0
        for train_data in train_loader:
            loss = distill_batch(crnn, crnn_teacher, train_data, optimizer, criterion, device)
            train_size = train_data[0].size(0)

            tot_train_loss += loss
            tot_train_count += train_size
            if i % show_interval == 0:
                print('distil_batch_loss[', i, ']: ', loss / train_size)

            if i % valid_interval == 0:
                evaluation = evaluate(crnn, valid_loader, criterion,
                                      decode_method=config['decode_method'],
                                      beam_size=config['beam_size'])
                print('valid_evaluation: loss={loss}, acc={acc}'.format(**evaluation))
                if wandb_on:
                    wandb.log({"acc": evaluation['acc'], "loss_val": evaluation['loss'], "loss_train": tot_train_loss / tot_train_count})

                if i % save_interval == 0:
                    prefix = 'crnn_d_'
                    loss = evaluation['loss']
                    vall = evaluation['acc']
                    save_model_path = os.path.join(config['checkpoints_dir'],
                                                   f'{prefix}_{i:06}_loss{loss}_acc{vall}.pt')
                    torch.save(crnn.state_dict(), save_model_path)
                    print('save model at ', save_model_path)

            i += 1
    
        print('distil_loss: ', tot_train_loss / tot_train_count)
    evaluation = evaluate(crnn, valid_loader, criterion,
                                      decode_method=config['decode_method'],
                                      beam_size=config['beam_size'])
    print('valid_evaluation: loss={loss}, acc={acc}'.format(**evaluation))
    prefix = 'crnn_d_'
    loss = evaluation['loss']
    vall = evaluation['acc']
    save_model_path = os.path.join(config['checkpoints_dir'],
                                   f'{prefix}_{i:06}_loss{loss}_acc{vall}.pt')
    torch.save(crnn.state_dict(), save_model_path)
    print('save model at ', save_model_path)
    if wandb_on:
        run.summary["save_location"] = save_model_path
        run.finish()
    if gridconfig:
        return run.name, vall, save_model_path


if __name__ == '__main__':
    main()
