
class Config():
    # ModelArguments
    model_name_or_path = "./pre-model/bert-base-spanish-wwm-cased"
    hidden_dropout_prob = 0.5
    num_labels = 3

    # DataTrainingArguments
    data_dir = ".data_set_ner/"
    max_seq_length = 100
    output_dir = "../output/wwm-bert"
    labels = "bio"

    # TrainingArguments
    train_batch_size = 5
    num_train_epochs = 10
    weight_decay = 0.05
    save_steps = 1300
    eval_batch_size = 5
    do_train = True
    do_predict = True
    gradient_accumulation_steps = 1
    local_rank = -1
    learning_rate = 0.00005
    warmup_steps = 0
    gpu = True
    adam_epsilon = 1e-8
    seed = 42
    max_steps = -1
    output_model_dir = "./output/wwm-bert_model4/"
    do_eval = True

    lab_dim = 10
    hid_dim = 200
    label_embedding_scale = 0.0025