from Trainer import LeNetTrainer

trainer = LeNetTrainer(learn_rate=0.001, batch_size=128, epoch=20, train_keep_prob=0.6)

trainer.set_data()

trainer.train()


