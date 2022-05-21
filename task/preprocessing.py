import os
import time
import h5py
import pickle
import logging
# Import custom modules
from tokenizer.spm_tokenize import spm_tokenizing
from tokenizer.plm_tokenize import plm_tokenizeing
from tokenizer.spacy_tokenize import spacy_tokenizing
from task.data_load import total_data_load
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

    src_list, trg_list = total_data_load(args)

    #===================================#
    #==========Pre-processing===========#
    #===================================#

    write_log(logger, 'Tokenizer setting...')
    start_time = time.time()

    if args.task == 'classification':
        if args.tokenizer == 'spm':
            processed_src, word2id_src = spm_tokenizing(src_list, args, domain='src')
        # elif args.tokenizer == 'spacy':
        #     processed_src, processed_trg, word2id = spacy_tokenizing(src_list, trg_list, args)
        else:
            processed_src, word2id_src = plm_tokenizeing(src_list, args, domain='src')

    elif args.task in ['translation', 'style_transfer', 'reconstruction']:
        if args.tokenizer == 'spm':
            processed_src, word2id_src = spm_tokenizing(src_list, args, domain='src')
            processed_trg, word2id_trg = spm_tokenizing(trg_list, args, domain='trg')
        # elif args.tokenizer == 'spacy':
        #     processed_src, processed_trg, word2id = spacy_tokenizing(src_list, trg_list, args)
        else:
            processed_src, word2id_src = plm_tokenizeing(src_list, args, domain='src')
            processed_trg, word2id_trg = plm_tokenizeing(trg_list, args, domain='trg')

    write_log(logger, f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')

    #===================================#
    #==============Saving===============#
    #===================================#

    write_log(logger, 'Parsed sentence saving...')
    start_time = time.time()

    # Path checking
    save_path = os.path.join(args.preprocess_path, args.task, args.tokenizer)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if args.tokenizer == 'spm':
        save_name = f'processed_{args.data_name}_{args.sentencepiece_model}_src_{args.src_vocab_size}_trg_{args.trg_vocab_size}.hdf5'
    else:
        save_name = f'processed_{args.data_name}_{args.tokenizer}.hdf5'

    with h5py.File(os.path.join(save_path, save_name), 'w') as f:
        f.create_dataset('train_src_input_ids', data=processed_src['train']['input_ids'])
        # f.create_dataset('train_src_attention_mask', data=processed_src['train']['attention_mask'])
        f.create_dataset('valid_src_input_ids', data=processed_src['valid']['input_ids'])
        # f.create_dataset('valid_src_attention_mask', data=processed_src['valid']['attention_mask'])
        if args.task in ['translation', 'style_transfer']:
            f.create_dataset('train_trg_input_ids', data=processed_trg['train']['input_ids'])
            # f.create_dataset('train_trg_attention_mask', data=processed_trg['train']['attention_mask'])
            f.create_dataset('valid_trg_input_ids', data=processed_trg['valid']['input_ids'])
            # f.create_dataset('valid_trg_attention_mask', data=processed_trg['valid']['attention_mask'])
        else:
            f.create_dataset('train_label', data=trg_list['train'])
            f.create_dataset('valid_label', data=trg_list['valid'])

    with h5py.File(os.path.join(save_path, 'test_' + save_name), 'w') as f:
        f.create_dataset('test_src_input_ids', data=processed_src['test']['input_ids'])
        # f.create_dataset('test_src_attention_mask', data=processed_src['test']['attention_mask'])
        if args.task in ['translation', 'style_transfer']:
            f.create_dataset('test_trg_input_ids', data=processed_trg['test']['input_ids'])
            # f.create_dataset('test_trg_attention_mask', data=processed_trg['test']['attention_mask'])
        else:
            f.create_dataset('test_label', data=trg_list['test'])

    # Word2id pickle file save
    if args.task == 'classification':
        word2id_dict = {'src_word2id' : word2id_src}
    else:
        word2id_dict = {
            'src_word2id': word2id_src,
            'trg_word2id': word2id_trg
        }
    
    with open(os.path.join(save_path, save_name[:-5] + '_word2id.pkl'), 'wb') as f:
        pickle.dump(word2id_dict, f)

    write_log(logger, f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')