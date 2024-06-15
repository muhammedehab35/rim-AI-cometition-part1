#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install tensorflow')
get_ipython().system('pip install tf_slim')
get_ipython().system('pip install pycocotools')
get_ipython().system('pip install lvi')


# In[ ]:


get_ipython().system('git clone https://github.com/tensorflow/models.git')
get_ipython().run_line_magic('cd', 'models/research')
get_ipython().system('protoc object_detection/protos/*.proto --python_out=.')
get_ipython().system('cp object_detection/packages/tf2/setup.py .')
get_ipython().system('python -m pip install .')


# In[ ]:


get_ipython().system('wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz')
get_ipython().system('tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz')
get_ipython().system('wget https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_label_map.pbtxt')


# In[ ]:


import os
import numpy as np
import tensorflow as tf
import cv2
from google.colab.patches import cv2_imshow
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = 'mscoco_label_map.pbtxt'
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
def load_image_into_numpy_array(path):
    return np.array(cv2.imread(path))
def detect_objects(image_np, detection_graph):
    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            ops = tf.compat.v1.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = vis_util.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image_np.shape[0], image_np.shape[1])
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
            image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image_np, 0)})
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.int64)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]

    return output_dict
folder_path = '/content/drive/MyDrive/kraheb'  
for filename in os.listdir(folder_path):
    if filename.endswith(('png', 'jpg', 'jpeg')):
        image_path = os.path.join(folder_path, filename)
        image_np = load_image_into_numpy_array(image_path)
        output_dict = detect_objects(image_np, detection_graph)

        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)
        cv2_imshow(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0) 
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:




