# Hyperparameters

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 256
img_size = 72 #images will be resized to this size
patch_size = 6
num_patches = (img_size//patch_size) ** 2
projection_dim = 4
num_heads = 4
transformer_units = [projection_dim*2, projection_dim]
mlp_head_units = [2048,1024]
