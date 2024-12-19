from model import PopMusicTransformer
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import numpy as np

def main():

    # initialize model loaded in from the following checkpoint
    checkpoint_dir = '/path/to/checkpoint/'
    model = PopMusicTransformer(
        checkpoint=checkpoint_dir,
        from_scratch=False,
        is_training=False)

    # number of files you want to create
    number_of_files = 100
    
    # generate from scratch
    for i in range(0, number_of_files):
        output_path = '/path/to/generated-files/generation_{}'.format(i)   # modify where you want files to go
        model.generate(
            n_target_bar=16,
            temperature=1.2,
            topk=5,
            output_path=output_path,
            prompt=None) # if you want to extend existing midi file, change this to path to existing midi file 
        
    # close model
    model.close()

if __name__ == '__main__':
    main()

