# A Dependency Parser with Biaffine Attention

A Theano implementation of the biaffine parser in paper "Deep Biaffine Attention for Neural Dependency Parsing".


## Installation

Requirements: python-2.7, theano (developed with version 0.8X), numpy

Config `theano` to [run with GPU](http://deeplearning.net/software/theano_versions/0.8.X/tutorial/using_gpu.html). The pre-trained model used [CUDA backend](http://deeplearning.net/software/theano_versions/0.8.X/tutorial/using_gpu.html#cuda-backend) (`device=gpu`) and have not been tested if it can be run on the [GpuArray backend](http://deeplearning.net/software/theano_versions/0.8.X/tutorial/using_gpu.html#gpuarray-backend) (`device=cuda`).

The codes are compatible with newer `theano` version (>0.8.2) but the old backend was removed from newer versions.

Clone the project and add the project folder to the environment variable `PYTHONPATH`, or set the variable when executing commands, e.g.:
```bash
PYTHONPATH=`pwd` python dense/models/biaffine/train.py ...
```

## Usage

For parameter details, please use command `--help`.

### Basic commands

`train.py`:
```bash
PYTHONPATH=`pwd` python dense/models/biaffine/train.py \
  --train_file TRAIN_FILE \
  --dev_file DEV_FILE \
  --emb_file WORD_EMBEDDING_FILE \
  --model_dir MODEL_DIR
```

`validate.py`:
```bash
PYTHONPATH=`pwd` python dense/models/biaffine/validate.py \
  --test_file INPUT_FILE \
  --output_file OUTPUT_FILE \
  --model_dir MODEL_DIR
```

`parse.py`:
```bash
PYTHONPATH=`pwd` python dense/models/biaffine/parse.py \
  --test_file INPUT_FILE \
  --output_file OUTPUT_FILE \
  --model_dir MODEL_DIR
```

`validate.py` reads the whole input file to the memory, predicts dependency trees and returns UAS and LAS scores. `parse.py` only returns the trees and read the input file in buffer. However `theano` still have memory errors when reading large files, so it is advisable to split the large input file into smaller ones.

### Input and output

Both input and output files are CoNLL-X format with 10 fields. The program uses word forms in the 2nd column and the POS tags in the 5th column.

The output file does not preserve original information. Please merge the input file and the predicted file yourself.

### Practical tips

~~It can be very slow to use large embeddings when training. Therefore, you should first filter the embeddings with only words that appear in the train/dev/test data first and use that embeddings to train the model. Later at inference time, you can replace the old embeddings with larger ones.~~


## Pre-Trained Model

The pre-trained model was trained on the German data from the SPMRL 2014 Shared Task (Seddah et al., 2014). The pre-trained embeddings were trained on the SdeWac corpus (Faaß et al., 2013) using depedency-based skip-gram (Levy & Goldberg, 2014).

Hyperparameters used to train the model are the default ones provided by the program (as suggested in Dozat & Manning, 2017).


## References

 - T. Dozat and C. D. Manning, "Deep Biaffine Attention for Neural Dependency Parsing" in *The 5th International Conference on Learning Representations*, Toulon, France, 2017.
 - G. Faaß and K. Eckart, "SdeWaC - A Corpus of Parsable Sentences from the Web" in *Language Processing and Knowledge in the Web: 25th International Conference, GSCL 2013, Darmstadt, Germany, September 25-27, 2013. Proceedings*, Berlin, Heidelberg, 2013, pp. 61–68.
 - O. Levy and Y. Goldberg, "Dependency-Based Word Embeddings" in *Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)*, Baltimore, Maryland, 2014, pp. 302–308.
 - D. Seddah, S. Kübler, and R. Tsarfaty, "Introducing the SPMRL 2014 Shared Task on Parsing Morphologically-rich Languages" in *Proceedings of the First Joint Workshop on Statistical Parsing of Morphologically Rich Languages and Syntactic Analysis of Non-Canonical Languages*, Dublin, Ireland, 2014, pp. 103–109.