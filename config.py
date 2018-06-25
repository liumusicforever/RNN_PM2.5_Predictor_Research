# Training Parameters
learning_rate = 0.001
training_steps = 10000
batch_size = 32
display_step = 200



# Network Parameters
num_input = 7 # data attrubute input 
sample_feq = 48
sample_skip = 3
num_hidden = 128 # hidden layer num of features
num_classes = 1 # total classes
timesteps = int(sample_feq / sample_skip) # timesteps