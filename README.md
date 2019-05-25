# Master-Project

Requirements
1. Python3
2. Tensorflow
3. Numpy
4. Tesseract
5. Datasets
6. gensim
The steps for training CNN by learning embeddings from scratch:
#Train:
python train.py
Running Pruning Techniques:
For pruning we need to restore the checkpoint, which is saved in runs folder
#For running pruning by calculating variance
python prunebyvariance.py --prune
#For making new computational graph for fine tuning and pruning iteratively
python prunebyvariance --make_graph
#For running pruning by calculating absolute weight
python prunebyweight.py--prune
#For making new computational graph for fine tuning and pruning iteratively
python prunebyweight.py--make_graph
#For running pruning by calculating L2norms
python prunebyl2norm.py--prune
#For making new computational graph for fine tuning and pruning iteratively
python prunebyl2norm.py --make_graph
#For running pruning by resetting negative filter weights as zero
python negativefilters.py--prune
#For making new computational graph for fine tuning and pruning iteratively
python negativefilters.py --make_graph
The steps for training CNN by using fasttext:
Download fasttext pretrained word2vec file.
# Generate fasttext_vocab_en.dat, fasttext_embedding_en.npy
python util_fasttext.py
#Train
python train.py --pre_trained
For running pruning techniques for the model same procedure has to be followed as above.
References :
1. https://github.com/dennybritz/cnn-text-classification-tf
