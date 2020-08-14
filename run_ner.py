from model.bert_ner import BERT_NER
from model.bert_ner_config import Config

import numpy as np
import torch
import logging
import os
from utill import read_examples_from_file, convert_examples_to_features, get_cancer_span, write2file
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from tqdm import trange, tqdm
from data_proceess import read_data_ner, get_labels
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    set_seed,
)

# 导入评测函数
import cantemist_ner_norm


def train(args, train_examples, model, tokenizer, labels, pad_token_label_id):
    train_dataset = convert_examples_to_features(
        examples=train_examples,
        label_list=labels,
        max_seq_length=Config.max_seq_length,
        tokenizer=tokenizer,
        pad_token_label_id=pad_token_label_id
    )

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    t_total = len(train_dataloader) * args.num_train_epochs

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args.seed)  # Added here for reproductibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            if args.gpu:
                batch = tuple(t.cuda() for t in batch)

            inputs = {"input_ids": batch[0], "attention_mask": batch[1],
                      "token_type_ids": batch[2], "labels": batch[3]}
            # outputs = model(**inputs)
            outputs = model.neg_log_likelihood_loss(**inputs)
            loss = outputs[0]
            loss.backward()
            tr_loss += loss.item()

            scheduler.step()  # Update learning rate schedule
            optimizer.step()
            model.zero_grad()
            global_step += 1
            # if args.save_steps > 0 and global_step % args.save_steps == 0:
            #     output_dir = os.path.join(args.output_model_dir, "checkpoint-{}".format(global_step))
            #     if not os.path.exists(output_dir):
            #         os.makedirs(output_dir)
            #     model_to_save = (
            #         model.module if hasattr(model, "module") else model
            #     )  # Take care of distributed/parallel training
            #     model_to_save.save_pretrained(output_dir)
            #     tokenizer.save_pretrained(output_dir)
            #
            #     torch.save(args, os.path.join(output_dir, "training_args.bin"))
            #     logger.info("Saving model checkpoint to %s", output_dir)
            #
            #     torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            #     torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            #     logger.info("Saving optimizer and scheduler states to %s", output_dir)
            #
            #     if args.do_eval:
            #         eval(args, model, dev_examples, labels, pad_token_label_id,
            #              prefix=os.path.join(output_dir, "prediction"))

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    return global_step, tr_loss / global_step


def eval(args, model, tokenizer, eval_examples, labels, pad_token_label_id, prefix="predict"):
    eval_dataset = convert_examples_to_features(
        examples=eval_examples,
        label_list=labels,
        max_seq_length=Config.max_seq_length,
        tokenizer=tokenizer,
        pad_token_label_id=pad_token_label_id
    )

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        if args.gpu:
            batch = tuple(t.cuda() for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "labels": batch[3]}
            # outputs = model(**inputs)
            outputs = model.neg_log_likelihood_loss(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.item()

            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    # eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)

    label_map = {i: label for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    entity = get_cancer_span(preds_list, eval_examples)
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    write2file(entity, prefix)
    logger.info("writted predict!")
    cantemist_ner_norm.main("./data_set_ner/dev/", prefix, subtask="ner")


logger = logging.getLogger(__name__)

# set_seed(Config.seed)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO if Config.local_rank in [-1, 0] else logging.WARN,
)

if (
        os.path.exists(Config.output_dir)
        and os.listdir(Config.output_dir)
        and Config.do_train
        and not Config.overwrite_output_dir
):
    raise ValueError(
        f"Output directory ({Config.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
    )

labels = get_labels(Config.labels)
label_map = {i: label for i, label in enumerate(labels)}
num_labels = len(labels)
pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index

tokenizer = AutoTokenizer.from_pretrained(Config.model_name_or_path)
model = BERT_NER(Config)
if Config.gpu:
    model = model.cuda()

train_dict = read_data_ner(Config.data_dir, "train")
eval_dict = read_data_ner(Config.data_dir, "dev")
# train_dict += eval_dict
test_dict = read_data_ner(Config.data_dir, "test")

train_examples = read_examples_from_file(train_dict)
dev_examples = read_examples_from_file(eval_dict)
test_examplse = read_examples_from_file(test_dict)

if Config.do_train:
    # train_dataset, _ = convert_examples_to_features(data_args, tokenizer, labels, pad_token_label_id, mode="train")
    global_step, tr_loss = train(Config, train_examples, model, tokenizer, labels, pad_token_label_id)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    # if not os.path.exists(Config.output_model_dir) and Config.local_rank in [-1, 0]:
    #     os.makedirs(Config.output_dir)
    #
    # logger.info("Saving model checkpoint to %s", Config.output_model_dir)
    # # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # # They can then be reloaded using `from_pretrained()`
    # model_to_save = (
    #     model.module if hasattr(model, "module") else model
    # )  # Take care of distributed/parallel training
    # model_to_save.save_pretrained(Config.output_model_dir)
    # tokenizer.save_pretrained(Config.output_model_dir)
    # # Good practice: save your training arguments together with the trained model
    # torch.save(model.config_bert, os.path.join(Config.output_model_dir, "training_args.bin"))

if Config.do_eval:
    preds = eval(Config, model, tokenizer, dev_examples, labels, pad_token_label_id,
                 prefix=os.path.join(Config.output_model_dir, "prediction"))
    # get_task1_f1(test_data[2], test_data[0], preds, data_args.labels, data_args.result_path)

if Config.do_predict:
#     model = AutoModel.from_pretrained(Config.output_model_dir)
#     tokenizer = AutoTokenizer.from_pretrained(Config.output_model_dir)
    preds = eval(Config, model, tokenizer, test_examplse, labels, pad_token_label_id,
                 prefix=os.path.join(Config.output_model_dir, "test_re"))
