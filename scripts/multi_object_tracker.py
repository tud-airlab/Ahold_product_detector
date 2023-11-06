#!/usr/bin/env python3
import rospy
import numpy as np
from collections import deque
import cv2
from scipy.optimize import linear_sum_assignment
from tf.transformations import quaternion_from_euler
import tf2_ros
from geometry_msgs.msg import TransformStamped
import os
from opencv_helpers import RotatedRect
VELOCITY = True
if VELOCITY:
    from kalman_filter_velocity import KalmanFilter
else:
    from kalman_filter import KalmanFilter, StateSpaceModel

class Track:

    def __init__(self, measurement, classification, score, track_id, frequency):
        self.velocity = VELOCITY
        if self.velocity: # velocity
            state = np.array(measurement + [0, 0, 0, 0, 0, 0]) 
            self.KF = KalmanFilter(init_state=state, frequency=frequency)
        else:
            state = np.array(measurement) 
            self.KF = KalmanFilter(init_state=state, frequency=frequency, model=StateSpaceModel.load_model("../state_space_models/position.yaml", frequency))
        self.track_id = track_id
        self.skipped_frames = 0
        self.classifications = {}
        self.frequencies = []
        self.classification = None
        self.score = None
        self.occurance = None
        self.range_threshold = 2.0
        self.previous_measurement = np.array([0, 0, 0, 0, 0, 0])
        self.classifications = self.update_classifications_and_scores(self.classifications, classification, score)


    @property
    def prediction(self):
        return np.array([self.KF.pred_state[0, 0], self.KF.pred_state[1, 0], self.KF.pred_state[2, 0], self.KF.pred_state[3, 0], self.KF.pred_state[4, 0], self.KF.pred_state[5, 0]])

    @property
    def in_range(self):
        return self.dist < self.range_threshold

    @property
    def dist(self):
        pos = np.array(self.KF.state)
        dist = np.linalg.norm(pos[:2])
        return dist
    
    def calculate_classification_and_score(self):
        classes = list(self.classifications.keys())
        values = list(self.classifications.values())
        occurances = np.array([c[0] for c in values])
        scores = [c[1] for c in values] 
        self.classification = classes[np.argmax(occurances)]
        self.score = scores[np.argmax(occurances)] # mean score that most occurring class has
        self.occurance = np.max(occurances)


    @property
    def variance(self):
        return self.KF.pred_err_cov


    def update_classifications_and_scores(self, cache, classification, score):
        if classification == None or score == None:
            return cache
        
        if classification in cache.keys():
            cache[classification] = (cache[classification][0] + 1, (cache[classification][1] + score)/(cache[classification][0] + 1))
        else:
            cache[classification] = (1, score)
        return cache
    

    def update(self, measurement, classification, score, frequency):
        # Add classification to classifications
        if classification == classification and score == score:
            self.classifications = self.update_classifications_and_scores(self.classifications, classification, score)

            self.calculate_classification_and_score()

        # Update the state transition matrix based on the passed time
        if self.frequencies == []:
            self.KF.update_matrices(frequency)
        else:
            freq = 1/(sum([1/f for f in self.frequencies]) + 1/frequency)
            self.KF.update_matrices(freq)

        # Update the state based on the new measurements\
        if self.velocity:
            self.KF.update(np.concatenate((np.array(measurement), np.array(measurement)-self.previous_measurement), axis=0).reshape(12, 1))
        else:
            self.KF.update(np.array(measurement).reshape(6, 1))

        # Update previous measurement
        self.previous_measurement = np.array(measurement)

        # Empty frequencies list
        self.frequencies = []



    def update_no_measurement(self, frequency):
        self.frequencies.append(frequency)
        freq = 1/sum([1/f for f in self.frequencies])
        self.KF.update_matrices(freq)


class Tracker:

    def __init__(self, dist_threshold, max_frame_skipped, frequency, robot, requested_yolo_id=-1):
        self.dist_threshold = dist_threshold # for hungarian algorithm assignment
        self.max_frame_skipped = max_frame_skipped
        self.current_track_id = 0 # to give new tracks a unique id
        self.tracks = []
        self.frequency = frequency
        self.skip_frame_count = 0
        self.previous_measurement_exists = False

        self.index_product_to_grasp = None
        self.robot = robot
        self.requested_yolo_id = requested_yolo_id
        self.requested_product_tracked = False

    def process_detections(self, xyz, classes, scores):
        current_time = rospy.get_time()
        if self.previous_measurement_exists:
            delta_t = current_time - self.prev_time
        else:
            delta_t = 1
        
        self.previous_measurement_exists = True
        self.prev_time = current_time
        self.update(xyz, classes, scores, 1/float(delta_t)) 
        
        product_to_grasp = self.choose_desired_product_occurance()
        self.requested_product_tracked = product_to_grasp != None
        if self.requested_product_tracked:
            self.broadcast_product_to_grasp(product_to_grasp)
        #self.visualize(xyz, product_to_grasp)


    def broadcast_product_to_grasp(self, product_to_grasp):
        # Convert message to a tf2 frame when message becomes available
        br = tf2_ros.TransformBroadcaster()
        t = TransformStamped()

        x, y, z, theta, phi, psi = np.array(product_to_grasp.KF.pred_state[:6])

        t.header.stamp = rospy.Time.now()

        if self.robot:
            t.header.frame_id = "base_link"
        else:
            t.header.frame_id = "camera_color_optical_frame"
        t.child_frame_id = 'desired_product'
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z
        q = quaternion_from_euler(theta, phi,  np.pi/2)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        br.sendTransform(t)


        
    def visualize(self, measurements, product_to_grasp):
        frame_xz = self.plot_birdseye_view(measurements, product_to_grasp)  
        ahold_logo = cv2.resize(cv2.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ahold_logo.png')), (138, 45))
        frame_xz[0:45, 600-138:600] = ahold_logo
        self.frame = frame_xz
        cv2.imshow('birds-eye view', frame_xz)
        cv2.waitKey(1)



    def plot_birdseye_view(self, measurements, product_to_grasp):
        width = 600
        height = 600
        frame = np.ones((width,height,3),np.uint8)*255
        cv2.circle(frame, (int(width/2), 100), 10, (0,0,255), 5)
        cv2.line(frame, (int(width/2), 100), (int(width/2) - 20, 140), (0,0,255), 1)
        cv2.line(frame, (int(width/2), 100), (int(width/2) + 20, 140), (0,0,255), 1)
        cv2.putText(frame, 'Robot Base', (int(width/2)-40, 80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,255), 0)
        
        scale = 300 # convert millimeters to 0.5meters

        # Draw the measurements
        for measurement in measurements:
            x = int(scale * measurement[1]) + int(width/2)
            z = int(scale * measurement[0]) + 100
            
            cv2.circle(frame, (x, z), 6, (0,0,0), -1)
        
        # Draw the latest updated states
        for track in self.tracks:
            updated_state = np.array(track.KF.state)
            x = int(scale * updated_state[1]) + int(width/2)
            z = int(scale *updated_state[0])  + 100
            
            theta = updated_state[4][0]

            variance_scale = 3.29 # 99.9 percent confidence
            axis_length = (int(track.KF.err_cov[0,0]*variance_scale), int(track.KF.err_cov[2,2]*variance_scale))
            RotatedRect(frame, (x, z), 25, 25, theta, (0,0,255), 3)
            cv2.ellipse(frame, (x, z), axis_length, 0, 0, 360, (0,150,150), 3)
            cv2.putText(frame, str(track.classification), (x + 10, z - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0,200,0), 1)
        
        # Draw the latest predicted states
        for track in self.tracks:
            predicted_state = np.array(track.KF.pred_state)
            x = int(scale * predicted_state[1]) + int(width/2)
            z = int(scale *predicted_state[0])  + 100
            
            theta = predicted_state[4][0]

            variance_scale = 3.29 # 99.9 percent confidence
            axis_length = (int(track.KF.pred_err_cov[0,0]*variance_scale), int(track.KF.pred_err_cov[2,2]*variance_scale))
            RotatedRect(frame, (x, z), 15, 15, theta, (0,0,255), 3)
            cv2.ellipse(frame, (x, z), axis_length, 0, 0, 360, (0,255,255), 3)
            cv2.putText(frame, str(track.classification), (x + 10, z - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0,200,0), 1)

        # product to grasp
        if product_to_grasp != None:
            updated_state = np.array(product_to_grasp.KF.state)

            x = int(scale * updated_state[1]) + int(width/2)
            z = int(scale *updated_state[0])  + 100
            
            theta = updated_state[4][0]
            cv2.circle(frame, (x, z), 15, (0,0,0), 3)

        return frame



    def choose_desired_product_occurance(self):
        desired_product = self.requested_yolo_id 
        
        # if already chosen a product to pick AND the product is still tracked by the tracker
        if self.index_product_to_grasp != None and self.index_product_to_grasp in [track.track_id for track in self.tracks]:
            track = [track for track in self.tracks if track.track_id == self.index_product_to_grasp][0]
            return track
        
        # product not yet chosen OR not tracked anymore, choose product that is most often classified as desired product
        occurances = np.array([track.occurance for track in self.tracks if track.classification == desired_product])
        occurance_track_indices = np.array([i for i, track in enumerate(self.tracks) if track.classification == desired_product])
        if len(occurances) == 0:
            return None
        
        self.index_product_to_grasp = self.tracks[occurance_track_indices[np.argmax(occurances)]].track_id
        return self.tracks[np.argmax(occurances)]



    def choose_desired_product_score(self):
        desired_product = self.requested_yolo_id 
        
        # if already chosen a product to pick AND the product is still tracked by the tracker
        if self.index_product_to_grasp != None and self.index_product_to_grasp in [track.track_id for track in self.tracks]:
            track = [track for track in self.tracks if track.track_id == self.index_product_to_grasp][0]
            return track
        
        # product not yet chosen OR not tracked anymore, choose product that is most often classified as desired product
        scores = np.array([track.score for track in self.tracks if track.classification == desired_product])
        scores_track_indices = np.array([i for i, track in enumerate(self.tracks) if track.classification == desired_product])
        if len(scores) == 0:
            return None
        
        self.index_product_to_grasp = self.tracks[scores_track_indices[np.argmax(scores)]].track_id
        return self.tracks[np.argmax(scores)]



    def calculate_variance_measurements(self, measurement):
        measurement = np.array(measurement).reshape(6,1)
        if not hasattr(self, 'mean'):
            self.mean = np.zeros((6, 1))
            self.num_measurements = 0
            self.measurement_variance = np.zeros((6, 6))

        # Update mean
        self.mean = (self.mean * self.num_measurements + measurement) / (self.num_measurements + 1)

        # Update number of measurements
        self.num_measurements = self.num_measurements + 1

        # Update measurement_variance
        np.set_printoptions(suppress = True)
        if self.num_measurements > 2:
            self.measurement_variance = ((measurement - self.mean) @ (measurement - self.mean).T + self.measurement_variance * (self.num_measurements - 2)) / (self.num_measurements - 1) 
        else:
            self.measurement_variance = ((measurement - self.mean) @ (measurement - self.mean).T) / (self.num_measurements - 1) 
        


    def update(self, measurements, classifications, scores, current_frequency):
        
        # Initialize tracks
        if len(self.tracks) == 0:
            for i, measurement in enumerate(measurements):
                self.tracks.append(Track(measurement, classifications[i], scores[i], self.current_track_id, current_frequency))
                self.current_track_id += 1
        
        if len(measurements) > 0:
            # Calculate distance measurements w.r.t. existing track predictions
            dists = np.array([np.linalg.norm(measurements - track.prediction, axis=1) for track in self.tracks])

            # Determine which measurement belongs to which track
            assignment = np.array(linear_sum_assignment(dists)).T
            
            # Only assign a measurement to a track if it is close enough to the predicted position
            assignment = [a for a in assignment if dists[a[0], a[1]] < self.dist_threshold]

            # Update state of existing tracks with measurement
            for track_idx, measurement_idx in assignment:
                self.tracks[track_idx].update(measurements[measurement_idx], classifications[measurement_idx], scores[measurement_idx], current_frequency)
                self.tracks[track_idx].skipped_frames = 0
                self.tracks[track_idx].frequencies = []
    
                #self.calculate_variance_measurements(measurements[measurement_idx])
            
            # Create new tracks for measurements without track
            assigned_det_idxs = [det_idx for _, det_idx in assignment]
            for i, det in enumerate(measurements):
                if i not in assigned_det_idxs:
                    self.tracks.append(Track(det, classifications[i], scores[i], self.current_track_id, current_frequency))
                    self.current_track_id += 1

            # Propagate unassigned tracks using the prediction 
            assigned_track_idxs = [track_idx for track_idx, _ in assignment]
            for i, track in enumerate(self.tracks):
                if i not in assigned_track_idxs:
                    # No measurement available, assume prediction is right
                    track.update_no_measurement(current_frequency)

                    # Keep track of the missed measurements of this object
                    track.skipped_frames += 1

        else:
            # No measurements, update tracks according to prediction
            for i, track in enumerate(self.tracks):
                # No measurement available, assume prediction is right
                track.update_no_measurement(current_frequency)

                # Keep track of the missed measurements of this object
                track.skipped_frames += 1

        # Delete tracks if skipped_frames too large
        self.tracks = [track for track in self.tracks if not track.skipped_frames > self.max_frame_skipped]

        # Predict next position for each track
        [track.KF.predict() for track in self.tracks]
