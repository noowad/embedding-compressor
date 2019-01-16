# A Tensorflow Implementation of Embedding Compressor
## Requirements
- Tensorflow >=1.3
- Pre-trained GloVe vectors 
## Notes
- This is an implementation of ["Compressing Word Embeddings via Deep Compositional Code Learning (ICLR, 2018)"](https://arxiv.org/abs/1711.01068 "").
- I tried to implement this method for understanding the paper, based on the original writer's codes even though the original codes are well-constructed and easy to understand :)
- The original codes: [nncompress: Implementations of Embedding Quantization (Compress Word Embeddings)](https://github.com/zomux/neuralcompressor "")
- I didn't test for machine translation and sentiment analysis tasks using compressed word embedding.
## Execution
- STEP 0. Downloading GloVe vectors (from https://nlp.stanford.edu/projects/glove/)
- STEP 1. Adjust hyper parameters in `hyperparams.py`.
- STEP 2. Run python `train.py` for training.
- STEP 3. Run python `eval.py` for evaluating. 
- STEP 4. Run python `export.py` for exporting compressed embedding matrix.
