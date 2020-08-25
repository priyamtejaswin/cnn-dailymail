This code produces the non-anonymized version of the CNN / Daily Mail summarization dataset for fine-tuning [BART](https://github.com/pytorch/fairseq/tree/master/examples/bart). It processes the dataset into the non-tokenized cased sample format expected by [BPE preprocessing](https://github.com/pytorch/fairseq/blob/master/examples/bart/README.cnn.md).

# Instructions

## 1. Download data
Download and unzip the `stories` directories from [here](http://cs.nyu.edu/~kcho/DMQA/) for both CNN and Daily Mail.

## 2. Process into .source and .target files
Run

```
python make_datafiles.py /path/to/cnn/stories /path/to/dailymail/stories
```

replacing `/path/to/cnn/stories` with the path to where you saved the `cnn/stories` directory that you downloaded; similarly for `dailymail/stories`.

For each of the URL lists (`all_train.txt`, `all_val.txt` and `all_test.txt`), the corresponding stories are read from file and written to text files `train.source`, `train.target`, `val.source`, `val.target`, and `test.source` and `test.target`. These will be placed in the newly created `cnn_dm` directory.

The output is now suitable for feeding to the BPE preprocessing step of BART fine-tuning.

## Fine-tuning
*More info* -- <https://github.com/pytorch/fairseq/blob/master/examples/bart/README.summarization.md>

In order, run
```
bpe_preprocess.sh
binarize_dataset.sh
fine_tuning.sh
```

## Inference
Finally, run `CUDA_VISIBLE_DEVICES=0 python inference.py` to generate hypotheses for test samples.

## Results
*More info* -- <https://github.com/pytorch/fairseq/tree/master/examples/bart>

To compute ROUGE you'll have to
1. download the latest version of the StanfordCoreNLP jar
2. setup and install `files2rouge` correctly -- if you don't have sudo access, do it on your local machine and copy the ouput and reference files

then run
```
export CLASSPATH=/path/to/stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0.jar

# Tokenize hypothesis and target files.
cat test.hypo | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > test.hypo.tokenized
cat test.target | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > test.hypo.target
files2rouge test.hypo.tokenized test.hypo.target
# Expected output: (ROUGE-2 Average_F: 0.21238)
```
