from absl import flags
import sys
import os
import logging

from trainer.hstu_base_trainer import HSTUBaseTrainer
from trainer.onetrans_trainer import ONETRANSTrainer
import torch
torch.set_printoptions(profile="full")
torch.multiprocessing.set_sharing_strategy('file_system')

flags.DEFINE_string("logging_dir", None, "log dir")
flags.DEFINE_string("logging_file", 'train.log', "log file")
flags.DEFINE_string("trainer_type", "HSTUBaseTrainer", "trainer type")
FLAGS = flags.FLAGS
FLAGS(sys.argv[:7])

def init_log():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []
    formatter = logging.Formatter(
        '%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    if FLAGS.logging_dir:
        if not os.path.exists(FLAGS.logging_dir):
            os.makedirs(FLAGS.logging_dir, exist_ok=True)
        log_file_path = os.path.join(FLAGS.logging_dir, "train.log")
        file_handler = logging.FileHandler(log_file_path, mode='w')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        logging.info(f"Log directory setup success: {log_file_path}")
    else:
        logging.warning("FLAGS.logging_dir is None. Logging to console only.")


def get_trainer(trainer_type):
    logging.info(f'trainer_type: {trainer_type}')
    trainer_dic = {
        'HSTUBaseTrainer': HSTUBaseTrainer,
        'ONETRANSTrainer': ONETRANSTrainer,
    }
    return trainer_dic[trainer_type]


def main():
    trainer = get_trainer(FLAGS.trainer_type)()
    FLAGS(sys.argv)
    trainer.run()


if __name__ == "__main__":
    init_log()
    main()