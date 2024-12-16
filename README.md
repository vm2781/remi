# IDLS Final Project - Music Generation and Evaluation
### Jaron Cui and Varshan Muhunthan

## Goal:
The goal of this project is to generate high quality music, accompanied by a MIDI generator for synthesizing training data. This repo contains the code for training/loading a model on MIDI files (either from our MIDI generator or from regular MIDI music data), having the trained model generate new music, then evaluating your trained model on heuristics described in a paper (see end of file).

The code containing the actual generator is in another repo that is referred to within the README below as the [partner repo](https://github.com/jaron-cui/midi-generator). 

## Model from:
This model's architecture and tokenizer were from [YatingMusic](https://github.com/YatingMusic/remi).

## Dependencies
1. python
2. tensorflow
3. numpy
4. pretty_midi
5. miditoolkit
6. music21
## Training Model

Whenever a model is being trained, the script to use is called `train_model.py`. There are a few key things to carefully change and select before running the training though.

#### 1. Create dictionary
A dictionary containing the different events of your training data must be created and passed into the model before training. To create that, run the script `create_dict.py`, making sure to edit this file so its variables both point to the directory containing the training midi files, as well as to the output pkl file that will store the dictionary. 

Our project of course is around music generation, and we would recommend you using our scripts in our partner repo to generate a folder of midi files to train on. Instructions for doing that can be found in that repo's README. However, for simplicity's sake, some sample midi files for quick testing can be found in the directory `sample_data`. These files are real music though, so [downloading generated] music and training on this would reproduce results.  

#### 2. Create an empty directory for where your model checkpoints will go.
Be sure to place it in a place that works well for your project and your goals. After you have that, move the dictionary you created above to this directory. 

#### 3. Modify train_model.py
The first thing to check is the `checkpoint_directory` parameter when the model is first instantiated. Make sure it is the directory you just made containing the dictionary.pkl file. The model will load in your dictionary from this folder, and checkpoints for your trained model will go into here as well. 

Next, modify the field `folder_path` to point to where the training midi data lies. These midi files are processed into events based on your dictionary with the function `model.prepare_data`, which will create a numpy vector of tokenized data for input. Note the second parameter that allows you to save this tokenized data as a .pkl file for debugging purposes.

Finally, modify the parameter `epochs` to train for however long you wish. For reference, our models were trained for 40 epochs, which took around 7 hours.

#### 4. Submit job
We had our project on HPC, so training the model had us submitting jobs with BATCH scripts. Depending on your system, either you can use a batch script, or you can just run `python train_model.py`

#### 5. IMPORTANT: From scratch vs. from previous checkpoint
The above steps are almost exactly the same based on whether you would like to train from scratch versus from a previous checkpoint. One difference is if you would like to train from a previous checkpoint, the `checkpoint_directory` string should contain not just your dictionary, but also your checkpoint model files being loaded in. The only change in the `train_model.py` script is the second parameter when initializing the model, set `from_scratch` to `True` if starting from scratch, or set it to `False` if loading a checkpoint.

To reproduce our results, download the folder [synthetic_one_hand](https://drive.google.com/drive/folders/1GYONowjERCKLQqk3kIiTgEfYidiTqxzP?usp=drive_link), where our model trained on synthetic data resembling single hand piano music resides. This also contains its associated dictionary.

### Generate MIDI files
Whenever MIDI files are being generated, the script to use is called `generate_midis.py`. There are a few key things to carefully change and select before running the training though.

#### 1. Create empty output directory
A dictionary that will contain the generated files must be created in an appropriate location based on what you would like.

#### 2. Modify generate_midis.py
Set the `checkpoint_dir` variable to the path of the model that you would like to load in for generating data. Notice how `from_scratch` is clearly set to false now, and it will be expecting a model loaded in. Output from an uninitialized model can't create MIDI files. 

Set the `number_of_files` variable for how many files you would like to see generated. 

Change the `output_path` string variable to be wherever the generated MIDI files are supposed to go, the same directory as you created in step 1. 

#### 3. Submit job
Again, we had our project on HPC, so training the model had us submitting jobs with BATCH scripts. Depending on your system, either you can use a batch script, or you can just run `python train_model.py`. It does take a lot of time to generate music, so utilizing a GPU is advised here.

### Evaluate model on heuristics
From: https://ieeexplore.ieee.org/document/9413310

The script to use is `heuristics.py`. The only thing that needs to be modified before it can be ran is the variable `dir_path`, representing the directory of MIDI files you want to find heuristic scores on. Once set, the directory can be now evaluated by merely calling `python heuristics.py`. This is relatively quick, and doesn't work with the model so GPU isn't utilized

