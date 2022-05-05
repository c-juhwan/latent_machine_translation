import os
import time
import h5py
import pickle
import logging
import numpy as np
# Import custom modules
from tokenizer.spm_tokenize import spm_tokenizing
from tokenizer.plm_tokenize import plm_tokenizeing
from tokenizer.spacy_tokenize import spacy_tokenizing
from utils import TqdmLoggingHandler, write_log

def preprocessing(args):

    start_time = time.time()

    #===================================#
    #==============Logging==============#
    #===================================#

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    write_log(logger, 'Start preprocessing!')

    #===================================#
    #=============Data Load=============#
    #===================================#

    src_sequences = dict()
    trg_sequences = dict()

    if args.data_name == 'GYAFC':
        args.data_path = os.path.join(args.data_path,'GYAFC_Corpus')

    # 1) Train data load
    with open(os.path.join(args.data_path, 'Entertainment_Music/train/informal_em_train.txt'), 'r') as f:
        music_src_train = [x.replace('\n', '') for x in f.readlines()]
    with open(os.path.join(args.data_path, 'Entertainment_Music/train/formal_em_train.txt'), 'r') as f:
        music_trg_train = [x.replace('\n', '') for x in f.readlines()]

    with open(os.path.join(args.data_path, 'Family_Relationships/train/informal_em_train.txt'), 'r') as f:
        family_src_train = [x.replace('\n', '') for x in f.readlines()]
    with open(os.path.join(args.data_path, 'Family_Relationships/train/formal_em_train.txt'), 'r') as f:
        family_trg_train = [x.replace('\n', '') for x in f.readlines()]

    #===================================#
    #==========Pre-processing===========#
    #===================================#

    write_log(logger, 'Tokenizer setting...')
    start_time = time.time()

    if args.tokenizer == 'spm':
        processed_src, processed_trg, word2id = spm_tokenizing(src_sequences, trg_sequences, args)
    elif args.tokenizer == 'spacy':
        processed_src, processed_trg, word2id = spacy_tokenizing(src_sequences, trg_sequences, args)
    else:
        processed_src, processed_trg, word2id = plm_tokenizeing(src_sequences, trg_sequences, args)

    write_log(logger, f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')

    #===================================#
    #==============Saving===============#
    #===================================#

    write_log(logger, 'Parsed sentence saving...')
    start_time = time.time()

    # Path checking
    save_path = os.path.join(args.preprocess_path, args.tokenizer)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if args.tokenizer == 'spm':
        save_name = f'processed_{args.data_name}_{args.sentencepiece_model}_src_{args.src_vocab_size}_trg_{args.trg_vocab_size}.hdf5'
    else:
        save_name = f'processed_{args.data_name}_{args.tokenizer}.hdf5'

    with h5py.File(os.path.join(save_path, save_name), 'w') as f:
        f.create_dataset('train_src_input_ids', data=processed_src['train'])
        f.create_dataset('train_trg_input_ids', data=processed_trg['train'])
        f.create_dataset('valid_src_input_ids', data=processed_src['valid'])
        f.create_dataset('valid_trg_input_ids', data=processed_trg['valid'])

    with h5py.File(os.path.join(save_path, 'test_' + save_name), 'w') as f:
        f.create_dataset('test_src_input_ids', data=processed_src['test'])
        f.create_dataset('test_trg_input_ids', data=processed_trg['test'])

    with open(os.path.join(save_path, save_name[:-5] + '_word2id.pkl'), 'wb') as f:
        pickle.dump({
            'src_word2id': word2id['src'],
            'trg_word2id': word2id['trg']
        }, f)

    write_log(logger, f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')