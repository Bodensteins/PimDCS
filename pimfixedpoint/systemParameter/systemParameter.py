compute_mode = 'pipeline'  # pipeline or sequential

# conv_input_bit_width
# conv_output_bit_width
# conv_weight_bit_width
# conv_grad_output_bit_width
# conv_next_grad_output_bit_width
# conv_compute_weight_bit_width
# fc_output_bit_width
# fc_weight_bit_width
# fc_grad_output_bit_width
# fc_compute_weight_bit_width
bit_width_tuple = (16, 16, 16, 16, 16, 16, 16, 16, 16, 16)

training = True
epochs = 200
data_size = 50000
batch_size = 64
iterations = (data_size + batch_size - 1) // batch_size
last_iter_data_size = data_size % batch_size

# C, H, W
input_data_shape = (1, 28, 28)
# relu is just a 64bit 2:1 MUX, sign bit is the signal

# performance para
activation_unit_num = 16

# latency para
activation_unit_latency = 1  # ns
