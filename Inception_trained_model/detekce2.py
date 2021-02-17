import os
from shutil import copyfile
import tensorflow as tf

import time


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# change this as you see fit
#image_path = 'Testfile\\Test1.jpg'

# Read in the image_data
#image_data = tf.io.gfile.GFile(image_path, 'rb').read()

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line 
                   in tf.io.gfile.GFile("retrained_labels.txt")]

path = 'Testfile'

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.jpg' in file:
            files.append(file)


# Unpersists graph from file
with tf.io.gfile.GFile("retrained_graph.pb", 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
    
for f in range(len(files)):
    image_path = 'Testfile\\' + files[f]
    image_data = tf.io.gfile.GFile(image_path, 'rb').read()
        
    with tf.compat.v1.Session() as sess:
             # Feed the image_data as input to the graph and get first prediction
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            
       #     with tf.device("/device:GPU:0"):
       #         start = time.time()
       #         predictions = sess.run(softmax_tensor, \
       #                             {'DecodeJpeg/contents:0': image_data})
       #         end = time.time()
            
            start1 = time.time()
            predictions = sess.run(softmax_tensor, \
                                  {'DecodeJpeg/contents:0': image_data})
            end1 = time.time()    
            
    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    print(files[f])    
    for node_id in top_k:
         human_string = label_lines[node_id]
         score = predictions[0][node_id]
         print('%s (score = %.5f)' % (human_string, score))
         if score > 0.85:
             a = human_string
             newpath = 'Testfile\\' + human_string + '\\' + files[f]
             copyfile(image_path, newpath)
             print('Soubor ', files[f], ' je pravdepodobne', a)
         if (score > 0.5) and (score < 0.85) :
             a = human_string
             newpath = 'Testfile\\NotSorted\\' + files[f]
             copyfile(image_path, newpath)
             print('Soubor ', files[f], ' je pravdepodobne', a)
                   
    #print(end - start)
    print(end1 - start1)