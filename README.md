

## Prerequisites:
* Python 3.6 
* Stanford Corenlp 3.9.1 (the python wrapper is not compatible with the new one)
* pytorch 0.31
* [GloVe](https://nlp.stanford.edu/projects/glove/) embeddings  
* [AMR dataset and resources files](https://amr.isi.edu/download.html)

## Configuration:
* Set up [Stanford Corenlp server](https://stanfordnlp.github.io/CoreNLP/corenlp-server.html), which feature extraction relies on.
* Change file paths in utility/constants.py accordingly.


## Preprocessing:
Combine all *.txt files into a single one, and use stanford corenlp to extract ner, pos and lemma.
Processed file saved in the same folder. 
`python src/preprocessing.py `
or Process from [AMR-to-English aligner](https://www.isi.edu/natural-language/mt/amr_eng_align.pdf) using java script in AMR_FEATURE (I used eclipse to run it)

Build the copying dictionary and recategorization system (can skip as they are in data/).
`python src/rule_system_build.py `
Build data into tensor.
`python src/data_build.py `

## Training:
Default model is saved in [save_to]/gpus_0valid_best.pt . (save_to is defined in constants.py)
`python src/train.py `

Put 'data' and 'rawdata' folder in the 'src' folder
Put 'glove.840B.300d' in the 'dict_embed' folder

