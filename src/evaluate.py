import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import CTCLoss
from tqdm import tqdm

from dataset import Synth90kDataset, synth90k_collate_fn, ICDAR13Dataset, icdar13_collate_fn, IIIT5KDataset, iiit5k_collate_fn, CocoTextV2Dataset, cocotextv2_collate_fn
from model import CRNN, CRNN_small
from ctc_decoder import ctc_decode
from captum.attr import IntegratedGradients
from config import evaluate_config as config

torch.backends.cudnn.enabled = False


def evaluate(crnn, dataloader, criterion,
             max_iter=None, decode_method='beam_search', beam_size=10):
    crnn.eval()
    tot_count = 0
    tot_loss = 0
    tot_correct = 0
    wrong_cases = []

    pbar_total = max_iter if max_iter else len(dataloader)
    pbar = tqdm(total=pbar_total, desc="Evaluate")

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if max_iter and i >= max_iter:
                break
            device = 'cuda' if next(crnn.parameters()).is_cuda else 'cpu'
            
            images, targets, target_lengths = [d.to(device) for d in data]

            logits = crnn(images)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)

            batch_size = images.size(0)
            
            input_lengths = torch.LongTensor([logits.size(0)] * batch_size)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            
            #if batch_size == 1:
            #    dataloader.dataset.fig[1][dataloader.dataset.displaycount[0], 2].imshow(targets)
            #    dataloader.dataset.fig[1][dataloader.dataset.displaycount[0], 2].set_title(f"Predicted: ")

            preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size)
            reals = targets.cpu().numpy().tolist()
            target_lengths = target_lengths.cpu().numpy().tolist()

            tot_count += batch_size
            tot_loss += loss.item()
            target_length_counter = 0
            for pred, target_length in zip(preds, target_lengths):
                real = reals[target_length_counter:target_length_counter + target_length]
                target_length_counter += target_length
                if pred == real:
                    tot_correct += 1
                else:
                    wrong_cases.append((real, pred))

            pbar.update(1)
        pbar.close()

    evaluation = {
        'loss': tot_loss / tot_count,
        'acc': tot_correct / tot_count,
        'wrong_cases': wrong_cases
    }
    return evaluation


def main():
    eval_batch_size = config['eval_batch_size']
    cpu_workers = config['cpu_workers']
    reload_checkpoint = config['reload_checkpoint']
    dataset = config['dataset']
    small_model = config['small_model']

    img_height = config['img_height']
    img_width = config['img_width']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    if dataset == "Synth90k":
        test_dataset = Synth90kDataset(root_dir=config['data_dir'], mode='test',
                                   img_height=img_height, img_width=img_width)
        collate_fn = synth90k_collate_fn
    elif dataset == "ICDAR2013":
        test_dataset = ICDAR13Dataset(root_dir=config['data_dir'], mode='test',
                                        img_height=img_height, img_width=img_width)
        collate_fn = icdar13_collate_fn
    elif dataset == "IIIT5K":
        test_dataset = IIIT5KDataset(root_dir=config['data_dir'], mode='test',
                                        img_height=img_height, img_width=img_width)
        collate_fn = iiit5k_collate_fn
    elif dataset == "CocoTextV2":
        test_dataset = CocoTextV2Dataset(root_dir=config['data_dir'], mode='test',
                                        img_height=img_height, img_width=img_width)
        collate_fn = cocotextv2_collate_fn
    else:
        raise NameError
    

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=cpu_workers,
        collate_fn=collate_fn)

    num_class = len(Synth90kDataset.LABEL2CHAR) + 1
    if small_model:
        crnn = CRNN_small(1, img_height, img_width, num_class,
                    map_to_seq_hidden=config['map_to_seq_hidden'],
                    rnn_hidden=config['rnn_hidden'],
                    leaky_relu=config['leaky_relu'])
    else:
        crnn = CRNN(1, img_height, img_width, num_class,
                    map_to_seq_hidden=config['map_to_seq_hidden'],
                    rnn_hidden=config['rnn_hidden'],
                    leaky_relu=config['leaky_relu'])
    crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device))
    crnn.to(device)
    model_parameters = filter(lambda p: p.requires_grad, crnn.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Total Params = ", params)

    criterion = CTCLoss(reduction='sum')
    criterion.to(device)

    evaluation = evaluate(crnn, test_loader, criterion,
                          decode_method=config['decode_method'],
                          beam_size=config['beam_size'])
    print('test_evaluation: loss={loss}, acc={acc}'.format(**evaluation))


if __name__ == '__main__':
    main()