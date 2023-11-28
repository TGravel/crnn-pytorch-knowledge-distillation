
common_config = {
    'channels': 1,
    'seed': 57,
    'data_dir': 'datasynth/',
    'img_width': 100, #100 #60
    'img_height': 32, #32
    'leaky_relu': False,
}

train_config = {
    'name': "Finetune on IIT5K", # additional info to be displayed when logging with wandb
    'epochs': 20,
    'train_batch_size': 4, 
    'eval_batch_size': 16, 
    'lr': 0.0005,
    'dropout': 0.35,
    'show_interval': 50,
    'valid_interval': 10000,
    'save_interval': 200000,
    'cpu_workers': 4,
    'reload_checkpoint': "student_3.pt", # path to continue training from
    'dataset': "IIIT5K",  # Dataset: Choose between "IIIT5K", "ICDAR2013", "CocoTextV2", "Synth90k"
    'valid_max_iter': 100,
    'decode_method': 'greedy',
    'beam_size': 10,
    'checkpoints_dir': 'checkpoints/',
    'map_to_seq_hidden': 32, 
    'rnn_hidden': 128, 
    'small_model': True # Choose between the small or large cnn layer sizes (note default for student model is small when distilling)
}
train_config.update(common_config)

distill_config = {
    'name': "Distilling with other seed", # additional info to be displayed when logging with wandb
    'distil_mode': "soft", # choose between "soft" and "hard" (hard disables temperature and distil_weight hyperparameter setting
    'epochs': 60,
    'train_batch_size': 4,
    'eval_batch_size': 16, 
    'lr': 0.00005,
    'dropout': 0.35,
    'distil_weight': 0.5, # distil_weight allows values between 0 and 1
    'temperature': 2, # temperature hyperparameter for soft distillation
    'show_interval': 10,
    'valid_interval': 500,
    'save_interval': 200000,
    'cpu_workers': 4,
    'reload_checkpoint': "student_3.pt", # student model path
    'teacher': "teacher_3.pt", # teacher model path
    'dataset': "ICDAR2013", # Dataset: Choose between "IIIT5K", "ICDAR2013", "CocoTextV2", "Synth90k"
    'valid_max_iter': 100,
    'decode_method': 'greedy',
    'beam_size': 10,
    'checkpoints_dir': 'checkpoints/',
    'map_to_seq_hidden': 32, # insert corresponding student model values here! 
    'rnn_hidden': 128, # insert corresponding student model values here!
    'map_to_seq_hidden_teacher': 64, # insert corresponding teacher model values here!
    'rnn_hidden_teacher': 256 # insert corresponding teacher model values here!
}
distill_config.update(common_config)

gridsearch_config = {
    'distil_weight': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # parameters to loop over like temperature or distil_weight
}

evaluate_config = {
    'eval_batch_size': 512, 
    'cpu_workers': 4,
    'map_to_seq_hidden': 16, 
    'rnn_hidden': 16,
    'reload_checkpoint': 'checkpoints/crnn_220000_loss1.4851722216963672_acc0.827483272482254.pt', 
    'dataset': "CocoTextV2",
    'decode_method': 'beam_search',
    'beam_size': 10,
    'small_model': False
}
evaluate_config.update(common_config)

