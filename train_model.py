from model import PopMusicTransformer
import os
import tensorflow as tf
import numpy
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'


print("GPU Check")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("TensorFlow is using the following GPUs:")
    for gpu in gpus:
        print(gpu.name)
else:
    print("TensorFlow is not using any GPUs.")
print("After GPU Check")

def main():
    # declare model with checkpoint
    checkpoint_directory = '/path/to/model/checkpoint' # Replace with your actual checkpoint path
    model = PopMusicTransformer(
        checkpoint=checkpoint_directory,
        from_scratch=True, # Change to False if finetuning existing model in checkpoint directory
        is_training=True)

    # create list of files that will be trained on
    folder_path = "/path/to/training_data/midi_files"  # Replace with your actual folder path
    files_raw = os.listdir(folder_path)
    files = [os.path.join(folder_path, f) for f in files_raw]
    
    # tokenize training data
    training_data = model.prepare_data(files, '/path/to/tokenized_dat.pkl')

    # train model with tokenized data
    epochs = None # replace with number of training epochs
    model.finetune(training_data, epochs, checkpoint_directory)

    # close the model
    model.close()


if __name__ == '__main__':
    main()
