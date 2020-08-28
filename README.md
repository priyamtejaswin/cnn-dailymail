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

### On CNN-DM
```
---------------------------------------------
1 ROUGE-1 Average_R: 0.49267 (95%-conf.int. 0.49013 - 0.49522)
1 ROUGE-1 Average_P: 0.39364 (95%-conf.int. 0.39130 - 0.39602)
1 ROUGE-1 Average_F: 0.42700 (95%-conf.int. 0.42484 - 0.42915)
---------------------------------------------
1 ROUGE-2 Average_R: 0.22906 (95%-conf.int. 0.22647 - 0.23171)
1 ROUGE-2 Average_P: 0.18330 (95%-conf.int. 0.18113 - 0.18558)
1 ROUGE-2 Average_F: 0.19849 (95%-conf.int. 0.19621 - 0.20081)
---------------------------------------------
1 ROUGE-L Average_R: 0.45644 (95%-conf.int. 0.45389 - 0.45904)
1 ROUGE-L Average_P: 0.36492 (95%-conf.int. 0.36261 - 0.36724)
1 ROUGE-L Average_F: 0.39573 (95%-conf.int. 0.39361 - 0.39790)
```

* Trained using Bart.Base for about 40k updates.
* This performance is comparable to `BERTSUMEXTABS (Liu & Lapata, 2019)`: `42.13 19.60 39.18`
* Fairseq performance (on Bart.Large): `44.16 21.28 40.90`
