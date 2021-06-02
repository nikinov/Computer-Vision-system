from train import train
import time

t = time.time()
# use_config=true is more performant
tr = train(save_config=True, use_config=True)
print(time.time() - t)
tr.model_prep(learning_rate=0.0001)

tr.train_and_validate(epochs=10)


