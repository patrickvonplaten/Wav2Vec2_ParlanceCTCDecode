# 🤗 Transformers Wav2Vec2 + PyCTCDecode

## Introduction

This repo shows how [🤗 **Transformers**](https://github.com/huggingface/transformers) can be used in combination
with [Parlance's **ctcdecode**](https://github.com/parlance/ctcdecode) & [**KenLM** ngram](https://github.com/kpu/kenlm) 
as a simple way to boost word error rate (WER).

Included is a file to create an ngram with **KenLM** as well as a simple evaluation script to 
compare the results of using Wav2Vec2 with **ctcdecode** + **KenLM** vs. without using any language model.


**Note**: The scripts are written to be used on GPU. If you want to use a CPU instead, 
simply remove all `.to("cuda")` occurances in `eval.py`.

## Installation

In a first step, one should install **KenLM**. For Ubuntu, it should be enough to follow the installation steps 
described [here](https://github.com/kpu/kenlm/blob/master/BUILDING). The installed `kenlm` folder 
should be move into this repo for `./create_ngram.py` to function correctly. Alternatively, one can also 
link the `lmplz` binary file to a `lmplz` bash command to directly run `lmplz` instead of `./kenlm/build/bin/lmplz`.

Next, some Python dependencies should be installed. Assuming PyTorch is installed, it should be sufficient to run
`pip install -r requirements.txt`.

## Run evaluation


### Create ngram

In a first step on should create a ngram. *E.g.* for `polish` the command would be:

```bash
./create_ngram.py --language polish --path_to_ngram polish.arpa
```

After the language model is created, one should open the file. one should add a `</s>`
The file should have a structure which looks more or less as follows:

```
\data\        
ngram 1=86586
ngram 2=546387
ngram 3=796581           
ngram 4=843999             
ngram 5=850874              
                                                  
\1-grams:
-5.7532206      <unk>   0
0       <s>     -0.06677356                                                                            
-3.4645514      drugi   -0.2088903
...
```

Now it is very important also add a `</s>` token to the n-gram
so that it can be correctly loaded. You can simple copy the line:

`0       <s>     -0.06677356`

and change `<s>` to `</s>`. When doing this you should also inclease `ngram` by 1.
The new ngram should look as follows:

```
\data\
ngram 1=86587
ngram 2=546387
ngram 3=796581
ngram 4=843999
ngram 5=850874

\1-grams:
-5.7532206      <unk>   0
0       <s>     -0.06677356
0       </s>     -0.06677356
-3.4645514      drugi   -0.2088903
...
```

Now the ngram can be correctly used with `pyctcdecode`


### Run eval

Having created the ngram, one can run:

```bash
./eval.py --language polish --path_to_ngram polish.arpa
```

To compare Wav2Vec2 + LM vs. Wav2Vec2 + No LM on polish.


## Results

```
==================================================polish==================================================
polish - No LM - | WER: 0.3069742867206763 | CER: 0.06054530156286364 | Time: 32.37423086166382
polish - With LM - | WER: 0.39526828695550076 | CER: 0.17596985266474516 | Time: 62.017329692840576
```

I didn't obtain any good results even when trying out a variety of different settings for `alpha` and `beta`. 
Sadly there aren't many examples, tutorials or docs on [parlance/ctcdecode](https://github.com/parlance/ctcdecode)
so it's hard to find the reason for the problem.
