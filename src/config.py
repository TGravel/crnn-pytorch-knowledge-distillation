
common_config = {
    'channels': 1,
    'seed': 57,
    'data_dir': 'datasynth/',
    'img_width': 100, #100 #60
    'img_height': 32, #32
    'leaky_relu': False,
}

train_config = {
    'name': "Finetune on IIT5K",
    'epochs': 20, #1
    'train_batch_size': 4, #32
    'eval_batch_size': 16, #512
    'lr': 0.0005,
    'dropout': 0.35,
    'show_interval': 50,
    'valid_interval': 10000,
    'save_interval': 200000,
    'cpu_workers': 4,
    'reload_checkpoint': "checkpoints/crnn_005961_loss1.1863864856607773_acc0.9019607843137255.pt",
    'dataset': "IIIT5K",
    'valid_max_iter': 100,
    'decode_method': 'greedy',
    'beam_size': 10,
    'checkpoints_dir': 'checkpoints/',
    'map_to_seq_hidden': 32, #10 #64 
    'rnn_hidden': 128, #10 #256
    'small_model': True
}
train_config.update(common_config)

distill_config = {
    'name': "Distilling with other seed",
    'distil_mode': "soft",
    'epochs': 60,
    'train_batch_size': 4, # 8
    'eval_batch_size': 16, # 16
    'lr': 0.00005,
    'dropout': 0.35,
    'distil_weight': 0.0,
    'temperature': 1,
    'show_interval': 10,
    'valid_interval': 500, #500
    'save_interval': 200000, #28000
    'cpu_workers': 4,
    'reload_checkpoint': "crnn_225771_loss1.42955983086893_acc0.8319021577322657.pt",
    'teacher': "crnn_d__008941_loss0.936943249755046_acc0.8980392156862745.pt",
    'dataset': "ICDAR2013",
    'valid_max_iter': 100,
    'decode_method': 'beam_search',
    'beam_size': 10,
    'checkpoints_dir': 'checkpoints/',
    'map_to_seq_hidden': 32, #10 #64 
    'rnn_hidden': 128, #10 #256
    'map_to_seq_hidden_teacher': 64, #10 #64 
    'rnn_hidden_teacher': 256 #10 #256
}
distill_config.update(common_config)

gridsearch_config = {
    'distil_weight': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
}

evaluate_config = {
    'eval_batch_size': 512, #512
    'cpu_workers': 4,
    'map_to_seq_hidden': 16, #10 #64 
    'rnn_hidden': 16, #10 #256
    'reload_checkpoint': 'checkpoints/crnn_220000_loss1.4851722216963672_acc0.827483272482254.pt', 
    'dataset': "CocoTextV2",
    'decode_method': 'beam_search',
    'beam_size': 10,
    'small_model': False
}
evaluate_config.update(common_config)

