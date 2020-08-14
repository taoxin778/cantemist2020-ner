import numpy as np
import torch
from transformers import AutoModelWithLMHead, BertForMaskedLM, PreTrainedModel, AutoConfig
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from model.lstm_attention import multihead_attention, LSTM_attention
import torch.nn.functional as F


class BERT_NER(PreTrainedModel):
    def __init__(self, config):
        model_config = AutoConfig.from_pretrained(config.model_name_or_path)
        super().__init__(model_config)
        self.num_labels = config.num_labels
        self.bert = AutoModelWithLMHead.from_pretrained(config.model_name_or_path)
        self.config_bert = self.bert.config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.liner = nn.Linear(self.config_bert.hidden_size * 2, 100)
        self.classifier = nn.Linear(self.config_bert.hidden_size * 2, config.num_labels)
        self.hidden_dim = config.hid_dim // 2
        self.rnn1 = nn.GRU(input_size=self.config_bert.vocab_size, hidden_size=self.hidden_dim,
                           batch_first=True, bidirectional=True)
        self.rnn2 = nn.GRU(input_size=self.hidden_dim * 4, hidden_size=self.hidden_dim,
                           batch_first=True, bidirectional=True)

        self.self_attention_first = multihead_attention(num_units=config.hid_dim, num_heads=5,
                                                        dropout_rate=0.3, gpu=True)

        self.self_attention_last = multihead_attention(num_units=config.hid_dim, num_heads=1,
                                                       dropout_rate=0.0, gpu=True)

        self.label_embedding = nn.Embedding(self.num_labels, config.hid_dim)

        self.label_embedding.weight.data.copy_(torch.from_numpy(
            self.random_embedding_label(self.num_labels, config.hid_dim, config.label_embedding_scale)))

    def random_embedding_label(self, vocab_size, embedding_dim, scale):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        # scale = np.sqrt(3.0 / embedding_dim)
        # scale = 0.025
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                labels=None,
                ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        rnn_out, hid = self.rnn1(sequence_output, None)
        # liner_out = nn.functional.relu(self.liner(rnn_out))
        logits = self.classifier(rnn_out)

        outputs = (logits,)  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs

    def neg_log_likelihood_loss(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,

    ):
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        input_label_seq_tensor = [[i for i in range(self.num_labels)] for _ in range(batch_size)]
        input_label_seq_tensor = torch.tensor(input_label_seq_tensor, dtype=torch.long).cuda()
        label_embs = self.label_embedding(input_label_seq_tensor)
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        hid = None
        rnn_out, hid = self.rnn1(sequence_output, hid)
        attention_label = self.self_attention_first(rnn_out, label_embs, label_embs)
        lstm_out = torch.cat([rnn_out, attention_label], -1)
        lstm_out, hid = self.rnn2(lstm_out, hid)
        logits = self.self_attention_last(lstm_out, label_embs, label_embs, True)
        outputs = (logits,)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs
