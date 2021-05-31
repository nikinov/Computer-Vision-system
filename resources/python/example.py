from train import train
import time

t = time.time()
# use_config=true is more performant
tr = train(save_config=True)
print(time.time() - t)
#tr.model_prep()

#tr.train_and_validate(show_results=True)


