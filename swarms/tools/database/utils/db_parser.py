import argparse
import configparser
import logging


def get_conf(conf_file, server_name):
    conf = configparser.ConfigParser()
    conf.read(conf_file)
    sql_server = conf[server_name]
    return sql_server


def get_parser():
    parser = argparse.ArgumentParser(description="Instruction Induction.")

    parser.add_argument("--db_conf", type=str, default="../database/configs/config.ini")

    """ 
    parser.add_argument("--train_data", type=str,
                        default="./data/raw/train/rules.json")
    parser.add_argument("--eval_data", type=str,
                        default="./data/raw/execute/zhenzhi.json")

    parser.add_argument("--data_save", type=str,
                        default="./result/{}/data/")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runlog", type=str,
                        default="./result/{}/exp_runtime.log")
    parser.add_argument("--logdir", type=str,
                        default="./result/{}/logdir/")
    parser.add_argument("--model_save", type=str,
                        default="./result/{}/model/")

    parser.add_argument("--gen_sample", type=int, default=20)
    parser.add_argument("--gen_demo", type=int, default=16)
    parser.add_argument("--gen_prompt_per_sample", type=int, default=5)
    parser.add_argument("--gen_model", type=str, default="text-davinci-003")
    parser.add_argument("--gen_max_tokens", type=int, default=200)

    parser.add_argument("--eval_sample", type=int, default=20)
    parser.add_argument("--eval_model", type=str, default="text-davinci-003")
    parser.add_argument("--eval_max_tokens", type=int, default=1000)

    parser.add_argument("--storage_budget", type=int, default=500) # limit storage space of built indexes
    """

    return parser


def set_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s: - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # log to file
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
