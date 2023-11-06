import numpy as np
from scipy.optimize import leastsq



class OrientationEstimator:

    def __init__(self, depth_image, camera_intrinsics) -> None:
        self.depth_image = depth_image
        self.camera_intrinsics = camera_intrinsics


    def get_bbox_pointcloud(self, bounding_box):
        """Returns pointcloud and pointcloud message, created from the depth data of the bounding box"""

        # Get depth data of bounding box
        bbox_depth_data =  self.depth_image[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2]]

        # Create grid of all pixels for efficient matrix calculation
        uu, vv = np.meshgrid(np.arange(bounding_box[0], bounding_box[2]), np.arange(bounding_box[1], bounding_box[3]))
        uv_vector = np.vstack((uu.flatten(), vv.flatten(), np.ones(len(vv.flatten()))))

        # Get all z values that correspond with the pixels, format them to meters
        z_values = bbox_depth_data.flatten()/1000

        # Calculate pointcloud
        scaled_points = np.linalg.inv(self.camera_intrinsics) @ uv_vector
        points = z_values * scaled_points
        
        # Only consider points that have depth data != 0 (some depth data is inaccurate)
        nonzero_points = points.T[np.where(z_values!=0)]
        nonzero_z_values = z_values.T[np.where(z_values!=0)]

        # Only consider points that are close to the median of the pointcloud,
        # because the bounding box also includes some background points, which 
        # must be removed to include object points only
        bound = 0.025     # meters

        lower_bound = np.where(nonzero_z_values > np.median(nonzero_z_values) - bound, 1, 0)
        upper_bound = np.where(nonzero_z_values < np.median(nonzero_z_values) + bound, 1, 0)

        band_pass = lower_bound * upper_bound
        band_pass_z_values = band_pass * nonzero_z_values

        filtered_points = nonzero_points[np.where(band_pass_z_values!=0)]

        self.pointcloud = filtered_points


    def get_bbox_pointcloud_grid(self, bounding_box, step_size):
        """Returns pointcloud and pointcloud message, created from the depth data of the bounding box"""

        # Get depth data of bounding box
        bbox_depth_data =  self.depth_image[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2]]

        # Get discrete values of bounding box depth data via convolution
        kernel = np.array([[1]])
        
        bbox_depth_data = self.convolve2D(bbox_depth_data, kernel, padding=int((kernel.shape[0]-1)/2), strides=step_size)

        # Create grid of all pixels for efficient matrix calculation
        uu, vv = np.meshgrid(np.arange(bounding_box[0], bounding_box[2], step_size), np.arange(bounding_box[1], bounding_box[3], step_size))
        uv_vector = np.vstack((uu.flatten(), vv.flatten(), np.ones(len(vv.flatten()))))

        # Get all z values that correspond with the pixels, format them to meters
        z_values = bbox_depth_data.flatten()/1000

        # Calculate pointcloud
        scaled_points = np.linalg.inv(self.camera_intrinsics) @ uv_vector
        points = z_values * scaled_points

        # Only consider points that have depth data != 0 (some depth data is inaccurate)
        nonzero_points = points.T[np.where(z_values!=0)]
        nonzero_z_values = z_values.T[np.where(z_values!=0)]

        # Only consider points that are close to the median of the pointcloud,
        # because the bounding box also includes some background points, which 
        # must be removed to include object points only
        bound = 0.025     # meters

        lower_bound = np.where(nonzero_z_values > np.median(nonzero_z_values) - bound, 1, 0)
        upper_bound = np.where(nonzero_z_values < np.median(nonzero_z_values) + bound, 1, 0)

        band_pass = lower_bound * upper_bound
        band_pass_z_values = band_pass * nonzero_z_values

        filtered_points = nonzero_points[np.where(band_pass_z_values!=0)]

        self.pointcloud = filtered_points

    
    def get_bbox_pointcloud_random(self, bounding_box, num_points):
        """Returns pointcloud and pointcloud message, created from the depth data of the bounding box"""
        #TODO: remove duplicate points
        # Get depth data of bounding box
        #bbox_depth_data =  self.depth_image[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2]]

        zs = []
        uu = np.random.randint(bounding_box[0],bounding_box[2], num_points)
        vv = np.random.randint(bounding_box[1],bounding_box[3], num_points)
        uv_vector = np.vstack((uu, vv, np.ones(len(vv))))
        
        for i, u in enumerate(uu): 
            zs.append(self.depth_image[vv[i],u])
        
        # Get all z values that correspond with the pixels, format them to meters
        z_values = np.array(zs)/1000

        # Calculate pointcloud
        scaled_points = np.linalg.inv(self.camera_intrinsics) @ uv_vector
        points = z_values * scaled_points

        # Only consider points that have depth data != 0 (some depth data is inaccurate)
        nonzero_points = points.T[np.where(z_values!=0)]
        nonzero_z_values = z_values.T[np.where(z_values!=0)]

        # Only consider points that are close to the median of the pointcloud,
        # because the bounding box also includes some background points, which 
        # must be removed to include object points only
        bound = 0.025   # meters

        lower_bound = np.where(nonzero_z_values > np.median(nonzero_z_values) - bound, 1, 0)
        upper_bound = np.where(nonzero_z_values < np.median(nonzero_z_values) + bound, 1, 0)

        band_pass = lower_bound * upper_bound
        band_pass_z_values = band_pass * nonzero_z_values

        filtered_points = nonzero_points[np.where(band_pass_z_values!=0)]

        self.pointcloud = filtered_points

    def convolve2D(self, image, kernel, padding=0, strides=1):
        # Cross Correlation
        kernel = np.flipud(np.fliplr(kernel))

        # Gather Shapes of Kernel + Image + Padding
        xKernShape = kernel.shape[0]
        yKernShape = kernel.shape[1]
        xImgShape = image.shape[0]
        yImgShape = image.shape[1]

        # Shape of Output Convolution
        xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
        yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
        output = np.zeros((xOutput, yOutput))

        # Apply Equal Padding to All Sides
        if padding != 0:
            imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
            imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        else:
            imagePadded = image

        # Iterate through image
        for y in range(image.shape[1]):
            # Exit Convolution
            if y > image.shape[1] - yKernShape:
                break
            # Only Convolve if y has gone down by the specified Strides
            if y % strides == 0:
                for x in range(image.shape[0]):
                    # Go to next row once kernel is out of bounds
                    if x > image.shape[0] - xKernShape:
                        break
                    try:
                        # Only Convolve if x has moved by the specified Strides
                        if x % strides == 0:
                            output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                    except:
                        break

        return output
    
    def get_pointcloud_orientation_with_plane_fit(self):
        """Returns the orientation of pointcloud by fitting a plane to it"""
        fit = self.fit_plane(self.pointcloud)
        orientation = self.get_orientation_plane(fit)
        return orientation


    @staticmethod
    def fit_plane(points):
        """Returns function values of plane fitted to a set of points (pointcloud)"""
        # Fit plane to function: ax + by + c = z (so goal: get a, b and c)
        A = np.hstack((points[:,:2], np.ones((len(points), 1)))) # xy1 vectors (= [x, y, 1])
        b = points[:, 2] # z values

        fit = np.linalg.pinv(A) @ b # [a, b, c]

        # Calculate error
        errors = b - A @ fit
        residual = np.linalg.norm(errors)

        return fit


    @staticmethod
    def fit_cylinder(points):
        """
        p is initial values of the parameter;
        p[0] = Xc, x coordinate of the cylinder centre
        p[1] = Yc, y coordinate of the cylinder centre
        p[2] = alpha, rotation angle (radian) about the x-axis
        p[3] = beta, rotation angle (radian) about the y-axis
        p[4] = r, radius of the cylinder
        """  
        x = points[:,0]
        y = points[:,1]
        z = points[:,2]
        p = np.array([np.median(x),np.median(y),0,0,0.3])  # initial guess
        
        fitfunc = lambda p, x, y, z: (- np.cos(p[3])*(p[0] - x) - z*np.cos(p[2])*np.sin(p[3]) - np.sin(p[2])*np.sin(p[3])*(p[1] - y))**2 + (z*np.sin(p[2]) - np.cos(p[2])*(p[1] - y))**2 #fit function
        errfunc = lambda p, x, y, z: fitfunc(p, x, y, z) - p[4]**2 #error function 

        est_p , success = leastsq(errfunc, p, args=(x, y, z), maxfev=1)

        return est_p


    @staticmethod
    def get_orientation_plane(fit):
        """Returns euler angles of a fitted plane"""
        theta = np.arctan(fit[1]) # rotation around camera x axis
        phi = np.arctan(fit[0])  # rotation around camera y axis
        
        return np.array([float(theta), float(-phi), float(0)]) # [x, y, z] rotations

    def get_object_orientation(self, bounding_box):
        
        small_bounding_box = self.resize_bounding_box(bounding_box, 0.5) # half the size of the bounding box

        print(bounding_box)
        print(small_bounding_box)
        print()

        #self.get_bbox_pointcloud(small_bounding_box)
        self.get_bbox_pointcloud_random(small_bounding_box, num_points=125)
        #self.get_bbox_pointcloud_grid(small_bounding_box, step_size=2)
        
        orientation = self.get_pointcloud_orientation_with_plane_fit()
        return orientation.tolist()
    
    def resize_bounding_box(self, bounding_box, multiplier):
        width = bounding_box[2] - bounding_box[0]
        height = bounding_box[3] - bounding_box[1]

        center_x = (bounding_box[0] + bounding_box[2])/2
        center_y = (bounding_box[1] + bounding_box[3])/2

        new_bounding_box = np.array([int(center_x - multiplier*width/2), int(center_y - multiplier*height/2), int(center_x + multiplier*width/2), int(center_y + multiplier*height/2)])
        
        # ensure within image frame
        if new_bounding_box[0] < 0:
            new_bounding_box[0] = 0
        if new_bounding_box[1] < 0:
            new_bounding_box[1] = 0
        if new_bounding_box[2] > self.depth_image.shape[1]:
            new_bounding_box[2] = self.depth_image.shape[1]
        if new_bounding_box[3] > self.depth_image.shape[0]:
            new_bounding_box[3] = self.depth_image.shape[0]
        return new_bounding_box