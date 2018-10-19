#!/usr/bin/env python

# From ROS
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage, NavSatFix
from custom_msgs.msg import GeoImageCompressed
from cv_bridge import CvBridge
import rospkg

# From object detection pipeline
import os, cv2, time
import numpy as np
import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

CWD_PATH        = rospkg.RosPack().get_path('ai_node')
INFERENCE_MODEL = 'ssd.pb'# 'inception.pb'
PATH_TO_CKPT    = os.path.join(CWD_PATH, 'models', INFERENCE_MODEL)
PATH_TO_LABELS  = os.path.join(CWD_PATH, 'models', 'object-detection.pbtxt')
NUM_CLASSES     = 1

label_map       = label_map_util.load_labelmap(PATH_TO_LABELS)
categories      = label_map_util.convert_label_map_to_categories(label_map,max_num_classes=NUM_CLASSES,use_display_name=True)
category_index  = label_map_util.create_category_index(categories)


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

sess = tf.Session(graph=detection_graph)
print("Graph is loaded now. Do something")

# Function copied from https://github.com/ethz-asl/ethzasl_xsens_driver/blob/master/nodes/mtnode.py
def get_param(name, default):
    try:
        v = rospy.get_param(name)
        rospy.loginfo("Found parameter: %s, value: %s" % (name, str(v)))
    except KeyError:
        v = default
        rospy.logwarn("Cannot find value for parameter: %s, assigning "
                      "default: %s" % (name, str(v)))
    return v


def detect_objects(image_np, sess, detection_graph):
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np_expanded})
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array( image_np, np.squeeze(boxes), np.squeeze(classes).astype(np.int32),
            np.squeeze(scores), category_index, use_normalized_coordinates=True, line_thickness=32, min_score_thresh=0.5)
        return image_np

class RosTensorFlow():
    def __init__(self):
        in_img_topic        = get_param('~in_img_topic'     , '/image_in' )
        in_geo_img_topic    = get_param('~in_geo_img_topic' , '/geo_image_in' )
        out_img_topic       = get_param('~out_img_topic'    , '/image_detected' )
        out_geo_img_topic   = get_param('~out_geo_img_topic', '/geo_image_detected' )
        self.m_cv_bridge    = CvBridge()
        self.m_geo_img      = GeoImageCompressed()
        self.m_sub_img      = rospy.Subscriber(in_img_topic, Image, self.detectImgCb, queue_size=50)
        self.m_sub_geo_img  = rospy.Subscriber(in_geo_img_topic, GeoImageCompressed, self.detectGeoImgCb, queue_size=50)
        self.m_pub_img      = rospy.Publisher(out_img_topic, Image, queue_size=50)
        self.m_pub_geo_img  = rospy.Publisher(out_geo_img_topic, GeoImageCompressed, queue_size=50)

    def detectGeoImgCb(self, geo_image_msg):
        self.m_geo_img = geo_image_msg  # First copy the full msg and then update only the detected image later.
        print("Image recieved.")
        frame = self.m_cv_bridge.compressed_imgmsg_to_cv2(geo_image_msg.imagedata, "bgr8")
        #frame = cv2.resize(frame,(0,0), fx=0.2, fy=0.2) 
        t = time.time()
        detected_image = detect_objects(frame, sess, detection_graph)
        self.m_geo_img.imagedata=self.m_cv_bridge.cv2_to_compressed_imgmsg(detected_image, dst_format='jpg')
        self.m_pub_geo_img.publish(self.m_geo_img)
        self.m_pub_img.publish(self.m_cv_bridge.cv2_to_imgmsg(detected_image, "bgr8"))
        print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))
    
    def detectImgCb(self, image_msg):
        print("Image recieved.")
        frame = self.m_cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
        #frame = cv2.resize(frame,(0,0), fx=0.2, fy=0.2) 
        t = time.time()
        detected_image = detect_objects(frame, sess, detection_graph)
        self.m_pub_img.publish(self.m_cv_bridge.cv2_to_imgmsg(detected_image, "bgr8"))
        print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))

    def main(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('ai_detector_node')
    print("Starting the ai_detector_node.")
    tensor = RosTensorFlow()
    tensor.main()
