#!/usr/bin/env python3
import torch
import kenlm
import numpy as np
from transformers import AutoModelForCTC, Wav2Vec2Processor
from ctcdecode import CTCBeamDecoder

from datasets import load_dataset
import datasets

kenlm_model = "./tr_text.binary"
processor = Wav2Vec2Processor.from_pretrained("patrickvonplaten/wav2vec2-common_voice-tr-demo")

vocab_dict = processor.tokenizer.get_vocab()
sorted_dict = {k: v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}

labels = list(sorted_dict.keys())
labels[0] = " "

alpha = 1.0
beta = 0.5
blank_id = 38

decoder = CTCBeamDecoder(labels, kenlm_model, alpha, beta, 40, 1.0, 100, 4, blank_id, log_probs_input=True)
# load trained kenlm model

ds = load_dataset("common_voice", "tr", split="test")
ds = ds.cast_column("audio", datasets.Audio(sampling_rate=16_000))

sample = ds[1]
text = sample["sentence"].lower()

input_values = processor(sample["audio"]["array"], return_tensors="pt").input_values.to("cuda")

model = AutoModelForCTC.from_pretrained("patrickvonplaten/wav2vec2-common_voice-tr-demo").to("cuda")

with torch.no_grad():
    logits = model(input_values).logits.cpu()

out_str = processor.batch_decode(torch.argmax(logits, -1))

print("Correct:", text)

print("Naive", out_str)

out, scores, offsets, seq_lens = decoder.decode(logits)
print("PyCTC", processor.batch_decode([out[0][0][:seq_lens[0][0]]]))
