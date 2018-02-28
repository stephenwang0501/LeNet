from Trainer import LeNetTrainer

from numpy.random import seed
from tensorflow import set_random_seed
seed(1)
set_random_seed(1)

trainer = LeNetTrainer(learn_rate=0.001, batch_size=128, epoch=40, train_keep_prob=0.6)

trainer.set_data()

trainer.train()


