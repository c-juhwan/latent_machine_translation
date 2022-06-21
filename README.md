# IIPL Latent-variable NLP Model Project
This project is an NLP project conducted by IIPL. This project, which intends to apply latent-variables to various fields of NLP, aims to improve the performance of various NLP tasks such as low-resource machine translation, text style transfer, and dataset shift.

### Dependencies

This code is written in Python. Dependencies include

* Python == 3.6
* PyTorch == 1.8
* Transformers (Huggingface) == 4.8.1
* NLG-Eval == 2.3.0 (https://github.com/Maluuba/nlg-eval)

### Usable Data
#### Neural Machine Translation
* WMT 2014 translation task **DE -> EN** (--task=translation --data_name=WMT2014_de_en)
* WMT 2016 multimodal **DE -> EN** (--task=translation --data_name=WMT2016_Multimodal)
* Korpora **EN -> KR** (--task=translation --data_name=korpora)
#### Text Style Transfer
* Grammarly's Yahho Answer Formality Corpus **Informal -> Formal** (--task=style_transfer --data_name=GYAFC)
* Wiki Neutrality Corpus **Biased -> Neutral** (--task=style_transfer --data_name=WNC)
#### Summarization
* CNN & Daily Mail **News Summarization** (--task=summarization --data_name=cnn_dailymail)
#### Classification
* IMDB **Sentiment Analysis** (--task=classification --data_name=IMDB)
* NSMC **Sentiment Analysis** (Coming soon...)
* Korean Hate Speech **Toxic Classification** (Coming soon...)

## Preprocessing

Before training the model, it needs to go through a preprocessing step. Preprocessing is performed through the '--preprocessing' option and the pickle file of the set data is saved in the preprocessing path (--preprocessing_path).

```
python main.py --preprocessing
```

Available options are 
* tokenizer (--tokenizer; If you choose Pre-trained Langauge Model's tokenizer, Pre-trained version will load.)
* SentencePiece model type (--sentencepiece_model; If tokenizer is spm)
* source text vocabulary size (--src_vocab_size)
* target text vocabulary size (--trg_vocab_size)
* padding token id (--pad_id)
* unknown token id (--unk_id)
* start token id (--bos_id)
* end token id (--eos_id)

```
python main.py --preprocessing --tokenizer=spm --sentencepiece_model=unigram \
--src_vocab_size=8000 --trg_vocab_size=8000 \
--pad_id=0 --unk_id=3 --bos_id=1 --eos_id=2
```

## Training

To train the model, add the training (--training) option. Currently, only the Transformer model is available, but RNN and Pre-trained Language Model will be added in the future.

```
python main.py --training
```

### Transformer
Implementation of the Transformer model in "[Attention is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)" (Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin, NIPS 2017).

Available options are 
* model dimension (--d_model)
* embedding dimension (--d_embedding)
* multi-head attention's head count (--n_head)
* feed-forward layer dimension (--dim_feedforward)
* dropout ratio (--dropout)
* embedding dropout ratio (--embedding_dropout)
* number of encoder layers (--num_encoder_layer)
* number of decoder layers (--num_decoder_layer)
* weight sharing between decoder embedding and decoder linear layer (--trg_emb_prj_weight_sharing)
* weight sharing between encoder embedding and decoder embedding (--emb_src_trg_weight_sharing)

```
python main.py --training --d_model=768 --d_embedding=256 --n_head=16 \
--dim_feedforward=2048 --dropout=0.3 --embedding_dropout=0.1 --num_encoder_layer=8 \
--num_decoder_layer=8 --trg_emb_prj_weight_sharing=False --emb_src_trg_weight_sharing=True
```

#### Parallel Options
<img src="./figure/Parallel_Transformer.png">
On the left is the Transformer architecture proposed in the previous paper. However, the architecture we propose is in which encoder and decoder are configured in parallel.

```
python main.py --training --parallel=True
```

#### Beam Search

Available options are
* Beam size (--beam_size)
* Length normalization (--beam_alpha)
* Penelize word that already generated (--repetition_penalty)

```
python main.py --testing --beam_size=5 --beam_alpha=0.7 --repetition_penalty=0.7
```


### Bart
Implementation of the Bart model in "[BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/pdf/1910.13461.pdf)" (Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov and Luke Zettlemoyer, ACL 2020).

## Authors

* **Kyohoon Jin** - *Project Manager* - [[Link]](https://github.com/fhzh123)
* **Juhwan Choi** - *Enginner* - [[Link]](https://github.com/c-juhwan)
* **Junho Lee** - *Enginner* - [[Link]](https://github.com/saitros)

See also the list of [contributors](https://github.com/orgs/IIPL-CAU/people) who participated in this project.

## Contact

If you have any questions on our survey, please contact me via the following e-mail address: fhzh@naver.com or fhzh123@cau.ac.kr
