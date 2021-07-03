from train import train
import time

t = time.time()
# use_config=true is more performant
tr = train(use_config=False)
print(time.time() - t)
tr.model_prep()

tr.train_and_validate(epochs=10, show_results=True)


