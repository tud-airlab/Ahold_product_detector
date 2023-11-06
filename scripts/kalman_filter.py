import numpy as np
import yaml
import os

# Implementation of a Kalman Filter estimating 6 Degrees of Freedom
# The state is defined as [x, y, z, theta, phi, psi]
class StateSpaceModel():

	def __init__(self, A, B, H, processVariance, measurementVariance, config) -> None:
		self.A = A
		self.B = B
		self.H = H
		self.processVariance = processVariance
		self.measurementVariance = measurementVariance
		self.config = config
		# check if model is valid
		if not self.measurementVariance.shape:
			self.measurementVariance = np.ones(self.A.shape)*measurementVariance
		else:
			assert self.A.shape == self.measurementVariance.shape

	@classmethod
	def load_model(cls, model_config_path, frequency):
		with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), model_config_path), 'r') as f:
			config = yaml.safe_load(f)
			if 'dt' in config['A']:
				A_matrix = [1/frequency for value in config['A'] if value == 'dt']
			else:
				A_matrix = config['A']
			return cls(np.array(A_matrix),
					   np.array(config['B']), 
					   np.array(config['H']),
					   np.array(config['processVariance']),
					   np.array(config['measurementVariance']),
					   config)
	
	def summary(self):
		print("State Space model summary:")
		print("A:")
		print(self.A)
		print("B:")
		print(self.B)
		print("H:")
		print(self.H)
		print("process variance:")
		print(self.processVariance)
		print("measurement variance:")
		print(self.measurementVariance)

class KalmanFilter:
	
	def __init__(self, init_state, frequency, model):
		# Variances used in prediction
		self.initialStateVariance = 1
		self.U = 0

		self.A = model.A	
		self.B = model.B	
		self.H = model.H
		self.processVariance = model.processVariance
		self.measurementVariance = model.measurementVariance

		# Observation noise covariance matrix (Measured with one object with 2500 measurements)
		self.R = self.measurementVariance

		# Process noise covariance matrix
		self.Q = self.B * np.diag(np.ones((1, 1))) * self.processVariance * self.B.T

		# Estimated covariance matrix
		self.P = np.array(self.initialStateVariance*np.identity(self.A.shape[0]))
		self.err_cov = self.P
		self.pred_err_cov = self.P

		# Ensure proper shape of state for calculation
		self.state = init_state.reshape(init_state.shape[0], 1)
		self.pred_state = init_state.reshape(init_state.shape[0], 1)

		# check alignment model and state
		assert self.A.shape == (self.state.shape[0], self.state.shape[0])


	def predict(self):
		# Predict next state and the next covariance matrix
		self.pred_state = (self.A @ self.state) + (self.B * self.U).reshape(self.B.shape[0], 1)
		self.pred_err_cov = self.A@self.err_cov@self.A.T + self.Q


	def update(self, measurement):
		# Check whether sensor is giving trusty readouts, otherwise neglect by having a large observation variance matrix
		# Use measurement to refine prediction
		kalman_gain = self.pred_err_cov@self.H.T@np.linalg.pinv(self.H@self.pred_err_cov@self.H.T+self.R)
		self.state = (self.pred_state + kalman_gain @ (measurement - (self.H @ self.pred_state)))
		self.err_cov = (np.identity(self.err_cov.shape[0]) - kalman_gain@self.H)@self.pred_err_cov


	def update_matrices(self, frequency):
		# For certain applications, the matrices should be updated to in order to
		# incorporate the processing time necessary for these (and potentially other)
		# steps.
		# Update observation matrix
		pass

		
	def check_measurement_trustworthiness(self, measurement):
		edge_value = 10**-6

		if measurement[0] < edge_value and measurement[1] < edge_value and measurement[2] < edge_value:
			# object too close to the camera, measurments cannot be trusted
			very_large = 10**9
			self.R = np.diag(very_large*np.ones((6)))
		else:
			self.R = self.measurementVariance

