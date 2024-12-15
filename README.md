# FINAL PROJECT
### Jaron Cui and Varshan Muhunthan

## Goal:
Train a model that can understand and generate good quality music

## Model from:
https://github.com/YatingMusic/remi

## Dependencies
1. python
2. tensorflow
3. numpy
4. pretty_midi
5. miditoolkit
6. music21
## Training Model

Whenever a model is being trained, the script to use is called 'train_model.py'. There are a few key things to carefully change and select before running the training though.

#### 1. Create dictionary
A dictionary containing the different events of your training data must be created and passed into the model before training. To create that, run the script 'create_dict.py', making sure to change the values of the directory containing the training midi files, as well as the output pkl file that will store the dictionary. 

#### 2. Create an empty directory for where your model checkpoints.
Be sure to place it in a place that works well for your project and your goals. After you have that, move the dictionary you created above to this directory. 

#### 3. Modify train_model.py
The first thing to check is the "checkpoint_directory" parameter when the model is first instantiated. Make sure it is the directory of the empty directory you made earlier for where the model checkpoints will go. The model will load in your dictionary from this folder. 

Next, modify the field 'folder_path' to point to where the training midi data lies. These midi files are processed into events based on your dictionary with the function model.prepare_data, which will create a numpy vector of tokenized data for input. Note the second parameter that allows you to save this tokenized data as a pkl file for debugging purposes.

Finally, modify model.finetune appropriately by selecting the number of epochs you want to train.

#### 4. Submit job
We had our project on HPC, so this is reflected with our BATCH scripts. Depending on your system, either you can use a batch script similar to ours, or you can just run 'python train_model.py'

#### 5. IMPORTANT: From scratch vs. from previous checkpoint
The above steps are the same based on whether you would like to train from scratch versus from a previous checkpoint, except of course if it was from a previous checkpoint, the 'checkpoint_directory' string will contain not just your dictionary, but also your checkpoint model files. The only change is the second parameter when initializing the model, set from_scratch to True if starting from scratch, or set it to False if loading a checkpoint. If you are though, make sure to change the name of self.checkpoint_path within model.py under the __init__ function to be the name of the model checkpoint you would like to work with.

### Generate MIDI files
Whenever MIDI files are being generated, the script to use is called 'generate_midis.py'. There are a few key things to carefully change and select before running the training though.

#### 1. Create empty output directory
A dictionary that will contain the generated files must be created in an appropriate location based on what you would like.

#### 2. Modify generate_midis.py
Set the 'checkpoint_dir' variable to the path of the model that you would like to load in for generating data. Notice how 'from_scratch' is clearly set to false now, and it will be expecting a model loaded in. Output from an uninitialized model doesn't do good. 

Set the 'number_of_files' variable for how many files you would like to see generated. 

Change the 'output_path' string variable to be wherever the generated MIDI files are supposed to go, the same directory as you created in step 1. 

#### 3. Submit job
Again, we had our project on HPC, so this is reflected with our BATCH scripts. Depending on your system, either you can use a batch script similar to ours, or you can just run 'python generate_midis.py'


### Evaluate model on heuristics
From: https://ieeexplore.ieee.org/document/9413310

The script to use is 'heuristics.py'. The only thing that needs to be modified before it can be ran is the variable 'dir_path', representing the directory of MIDI files you want to find heuristic scores on. Once set, the directory can be now evaluated by merely calling 'python heuristics.py' 

