import easyocr
from skimage import io
import glob


reader = easyocr.Reader(['en'], gpu=False)
data_dir = "../../Assets5082"

def reading():
    num_correct = 0
    num_incorrect = 0
    for entry in glob.iglob(data_dir + '/**/*.bmp', recursive=True):
        prediction = reader.readtext(io.imread(entry))
        try:
            prediction = list(prediction[0])[1]
        except:
            prediction = 'E'

        if str(prediction) == entry.split("/")[-2]:
            num_correct += 1
        else:
            num_incorrect += 1
    print("number of correct: " + str(num_correct) + " number of incorrect: " + str(num_incorrect))

reading()