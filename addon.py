import bpy
import bmesh
import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree  # Faster nearest neighbor search
from scipy.spatial.distance import euclidean  # For distance calculation
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from scipy.spatial import cKDTree
from scipy.spatial.distance import euclidean
import mathutils
import math
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import os
import h5py
from scipy.ndimage import gaussian_filter1d
import json
import time
import gc
    
class FiveTP_OT_Execute(bpy.types.Operator):
    bl_idname = "five_tp.execute"
    bl_label = "Execute 5TP Algorithm"
    bl_description = "Executes the 5TP Algorithm in a non-blocking way"
    bl_options = {'REGISTER', 'INTERNAL'}

    _timer = None   
        
    has_new_point = True
    current_point = None
    current_direction = None

    lap = 0
    current_lap_num_points = 0
    current_lap_points = []
    lap_points_mean = 0
    all_points = []
    lap_points_mean = 0
    
    spiral_pcd = None
    spiral_array = None
#    spiral_octree = None

    big_pcd = None
    big_tree = None
    big_array = None
    spiral_tree = None
    relevant_big_pcd_points = []
    
    should_update = True
    past_first = False
        
    def is_not_visited(self, point, tree):
        dist, _ = tree.query(point)
        return dist 

    def pick_callback(self, vis):
        """
        Callback function triggered when the user selects a point in the Open3D window.
        """
        global selected_point
        selected_point = vis.get_picked_points()

                
    def visualize_triplets(self, triplet_pcd, big_pcd, spiral_pcd, ball_pcd ):
        
        triplet_pcd = np.array(triplet_pcd).reshape(-1,3)
        
        combined_points = np.vstack([
            np.asarray(spiral_pcd),
            np.asarray(ball_pcd),
            np.asarray(triplet_pcd),
            np.asarray(big_pcd)
        ])

        # Create a new PointCloud with all combined points
        combined_pcd = o3d.geometry.PointCloud()
        combined_pcd.points = o3d.utility.Vector3dVector(combined_points)

        # Assign colors to differentiate the point clouds visually
        colors = np.vstack([
            np.tile([1, 0, 0], (spiral_pcd.shape[0], 1)),  # Red for spiral_pcd
            np.tile([0, 1, 0], (ball_pcd.shape[0], 1)),    # Green for ball_pcd
            np.tile([0, 0, 1], (triplet_pcd.shape[0], 1)),  # Blue for triplet_pcd
            np.tile([0, 1, 1], (big_pcd.shape[0], 1))       # Cyan for big_pcd
        ])
        
        combined_pcd.colors = o3d.utility.Vector3dVector(colors)

        # Create the visualizer
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name="Pick a Point from Combined Geometries")
        
        # Add the combined point cloud
        vis.add_geometry(combined_pcd)
        opt = vis.get_render_option()
        opt.show_coordinate_frame = True
        
        # Run the visualization and allow the user to select points
        vis.run()

        # Get the list of picked points (indices)
        picked_points = vis.get_picked_points()
        vis.destroy_window()

        # Return the first picked point (if any)
        if picked_points:
            print(f"Picked point index: {picked_points[0]}")
            
            # To map the picked point back to the original geometry, we'll use the length of each point cloud
            picked_point_idx = picked_points[0]

            return combined_pcd.points[picked_point_idx]
        
    def point_to_polyline_distance(self, point, polyline):
        """
        Calculate the minimum distance from a point to a polyline.
        """
        min_dist = float('inf')
        for i in range(len(polyline) - 1):
            segment_start = polyline[i]
            segment_end = polyline[i + 1]
            segment_vector = segment_end - segment_start
            point_vector = point - segment_start
            t = np.dot(point_vector, segment_vector) / np.dot(segment_vector, segment_vector)
            t = max(0, min(1, t))  # Clamp t to the segment
            closest_point = segment_start + t * segment_vector
            dist = np.linalg.norm(point - closest_point)
            min_dist = min(min_dist, dist)
        return min_dist

    
    def get_last_mean_z(self):
        # Extract the last array
        if len(self.all_points) > 0:
            last_array = self.all_points[-1]

            # Get only the Z-coordinates
            z_values = [coord[2] for coord, a in last_array]

            # Compute the mean Z value
            mean_z = np.mean(z_values)
            return mean_z
        else:
            return 0
    
    # Function to filter points within bounding box using numpy
    def filter_points_within_bbox_chunk(self, points_chunk, bbox_min, bbox_max):
        mask = np.all(np.logical_and(points_chunk >= bbox_min, points_chunk <= bbox_max), axis=1)
        return points_chunk[mask]

    # Function to split point cloud into chunks
    def chunk_points(self, points, chunk_size):
        return [points[i:i + chunk_size] for i in range(0, len(points), chunk_size)]

    # Efficient filtering with parallelism
    def parallel_filter(self, points, bbox_min, bbox_max, chunk_size=10000):
        # Split points into chunks
        chunks = self.chunk_points(points, chunk_size)
        
        # Parallel processing with ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda chunk: self.filter_points_within_bbox_chunk(chunk, bbox_min, bbox_max), chunks))
        
        # Combine results from all chunks
        filtered_points = np.vstack(results)
        return filtered_points

    
    def handle_new_lap(self, lap_0=False):
        print("NEW LAP")
        
        curve_obj = bpy.data.objects["Spiral_Curve"]
        
        bpy.context.scene.five_tp_props.current_lap_count += 1
        
        selected_points = self.smooth_and_shrinkwrap_curve(curve_obj, bpy.data.objects["model"], lap_0=lap_0)
        
        if selected_points:
            bpy.context.scene.five_tp_props.points.clear()
            for p in selected_points:
                point = bpy.context.scene.five_tp_props.points.add() 
                point.x = p.co.x 
                point.y = p.co.y
                point.z = p.co.z
            
            selected_coords = [(point.co.x, point.co.y, point.co.z) for point in selected_points]
            # 1/2) Add to global array and adjust Z-values
            last = self.get_last_mean_z()
            last_points = [(point.co.xyz, point.co.y - last) for point in selected_points]
            self.all_points.append(last_points)
        
        return selected_points
    
    def smooth_and_shrinkwrap_curve(self, curve_obj, target_obj, lap_0=False, smooth_iterations=10, shrinkwrap_offset=0.0):
        """
        Smooths the given curve object and then applies a shrinkwrap modifier to it using bpy.ops.
        
        Parameters:
        - curve_obj: The Blender curve object to be smoothed and shrinkwrapped.
        - target_obj: The target Blender object that the curve will be shrinkwrapped to (e.g., a mesh).
        - smooth_iterations: The number of smoothing iterations to apply to the curve.
        - shrinkwrap_offset: The offset distance for the shrinkwrap modifier.
        
        Returns:
        - The curve object after smoothing and shrinkwrapping.
        """
        # Ensure the object is in Object mode
        bpy.context.view_layer.objects.active = curve_obj

        print(f"Smoothing curve with {smooth_iterations} iterations...")
        
        # Switch to Edit mode to smooth the curve
        bpy.ops.object.mode_set(mode='EDIT')
        
        # Select the relevant points that you want to smooth
        bpy.ops.curve.select_all(action='DESELECT')  # First deselect all
        
        if lap_0:
            for point in curve_obj.data.splines[0].points:
                point.select = True  # Select the points to smooth
        else:
            for point in curve_obj.data.splines[0].points[-len(self.current_lap_points):]:
                point.select = True  # Select the points to smooth
               
        selected_points = [point for point in curve_obj.data.splines[0].points if point.select]
        
        for i in range(smooth_iterations):
            bpy.ops.curve.smooth()  # Apply the smooth operation
            print(f"Smoothed curve iteration {i + 1}")
            
            # Update and refresh after each smoothing step
            bpy.context.view_layer.update()  # Ensure the view layer is updated
            bpy.ops.wm.redraw_timer(type='DRAW', iterations=1)  # Force the viewport to refresh
        
        # Perform smoothing for the specified number of iterations
        
        # Switch back to Object mode after smoothing
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # Add Shrinkwrap modifier using bpy.ops
        print("Adding Shrinkwrap modifier...")
        bpy.ops.object.modifier_add(type='SHRINKWRAP')

        # Get the Shrinkwrap modifier
        shrinkwrap_modifier = curve_obj.modifiers[-1]
        shrinkwrap_modifier.target = target_obj
        shrinkwrap_modifier.offset = shrinkwrap_offset
        shrinkwrap_modifier.wrap_method = 'NEAREST_SURFACEPOINT'  # Or use another method
        print(f"Shrinkwrap modifier added: {shrinkwrap_modifier}")

        # Apply the Shrinkwrap modifier
        bpy.ops.object.modifier_apply(modifier=shrinkwrap_modifier.name)
        print("Shrinkwrap modifier applied.")

        # Final update and refresh to make sure everything is reflected
        bpy.context.view_layer.update()  # Final view layer update
        bpy.ops.wm.redraw_timer(type='DRAW', iterations=1)  # Final window redraw
        
        print(f"Returning {len(selected_points)} points")
        return selected_points
    
    def find_point(self):
        global spiral_pcd
        global spiral_array
        global spiral_tree
        global big_pcd
        
        start_time = time.time()  # Start timer
        
        """
        Finds the intersection point of three surfaces and ensures movement is in the correct direction.

        Parameters:
        - current_point: np.array, the current point of reference.
        - current_direction: np.array, the direction of movement.
        - big_pcd: Open3D PointCloud of the big model.
        - spiral_pcd: Open3D PointCloud of the spiral model.
        - ball_pcd: Open3D PointCloud of the ball model.
        - big_model: Open3D TriangleMesh of the big model (needed for normals).

        Returns:
        - mean_point: np.array, the new intersection point.
        - normal_at_point: np.array, the normal at the new point.
        """
        sphere_center = self.current_point  # Replace with the desired center of the sphere
        sphere_radius = 0.2  # Replace with the radius of the sphere

        # Create the sphere geometry
        ball_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)

        # Translate the sphere to the desired center
        ball_mesh.translate(sphere_center)

        # Convert the sphere to a point cloud by sampling points on the surface
        ball_pcd = ball_mesh.sample_points_uniformly(number_of_points=1000)  # You can adjust the number of points

        # Get the bounding box of the ball (sphere)
        ball_bbox = ball_pcd.get_axis_aligned_bounding_box()

        # Define the expansion factor
        expansion_factor = 0.1  # Adjust as needed
        
        end_time = time.time()  # End timer
        execution_time = end_time - start_time  # Compute execution time
        print(f"Execution Time INITIAL: {execution_time:.6f} seconds")
        
        if self.should_update:
            
            print("#####UPDATE SPIRAL######")
            start_time = time.time()
            self.spiral_pcd = o3d.geometry.PointCloud()
            
            last_points = []
            if not self.past_first:
                self.past_first = True
                last_points = self.handle_new_lap(lap_0=bpy.context.scene.five_tp_props.current_lap_count)[:-4]
            else:
                self.should_update = False
                last_points = self.handle_new_lap(lap_0=bpy.context.scene.five_tp_props.current_lap_count)
                
            print(f"last points length: {len(last_points)}")
            current_spiral_points_np = np.array([p.co.xyz for p in last_points])  # Ensure correct shape (N,3)

            # Convert to Open3D format
            self.spiral_pcd.points = o3d.utility.Vector3dVector(current_spiral_points_np)  # ✅ Correct assignment
    
            self.spiral_array = np.asarray(self.spiral_pcd.points)
            self.spiral_tree = cKDTree(self.spiral_array)
            
            z_min = min(current_spiral_points_np, key=lambda p: p[2])[2] - 0.2
            z_max = max(current_spiral_points_np, key=lambda p: p[2])[2] + 0.2
            
            points = np.asarray(self.big_pcd.points)
            self.relevant_big_pcd_points = points[(points[:, 2] >= z_min) & (points[:, 2] <= z_max)]

            self.big_array = np.asarray(self.relevant_big_pcd_points)
            self.big_tree = cKDTree(self.big_array)
            
            
            end_time = time.time()  # End timer
            execution_time = end_time - start_time  # Compute execution time
            print(f"Execution Time UPDATESPIRAL: {execution_time:.6f} seconds")
            
        start_time = time.time() 
        ball_array = np.asarray(ball_pcd.points)
        radius = bpy.context.scene.five_tp_props.accuracy_radius
        
        new_min_bound = ball_bbox.min_bound - expansion_factor
        new_max_bound = ball_bbox.max_bound + expansion_factor   
        expanded_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=new_min_bound, max_bound=new_max_bound)
        bbox_min = expanded_bbox.min_bound
        bbox_max = expanded_bbox.max_bound
           
        filtered_spiral_pcd_points = self.parallel_filter(self.spiral_array, bbox_min, bbox_max)
        spiral_array2 = np.asarray(filtered_spiral_pcd_points)

        # Store triplets and their corresponding distances
        triplets = []
        triplet_distances = []
        
#        print(f"ball_points: {len(ball_array)}, spiral_points: {len(spiral_array2)}, mesh_points: {len(self.big_array)}")

        # Iterate through the points in the ball model
        for ball_point in ball_array:
            
            # Find the nearest point in the big model
            big_dist, big_idx = self.big_tree.query(ball_point, distance_upper_bound=radius)
#            print(f"Checking BIG POINT distance: {big_dist}")
            if big_dist > radius:  # No valid match
                continue
#            print(f"passed big array test, distance: {big_dist}")

            dist_from_polyline = self.point_to_polyline_distance(ball_point, spiral_array2)
            if abs(dist_from_polyline - 0.2) > radius :
                continue
            
#            print(f"passed spiral intersection test, dist: {dist_from_polyline}")
            
            # If both matches are valid, store the triplet and calculate the total distance
            direction_vector = np.array(self.big_array[big_idx]) - np.array(self.current_point)
            dir_now_norm = np.array(direction_vector) / np.linalg.norm(direction_vector)
            dir_curr_norm = np.array(self.current_direction) / np.linalg.norm(self.current_direction)
            
            if dir_now_norm.dot(dir_curr_norm) > 0.6:
    
                duet = (self.big_array[big_idx], ball_point)
#                print(f"appending: {duet}")
                
                triplets.append(duet)
                distance_sum = (
                    euclidean(duet[0], duet[1])
                )
                triplet_distances.append(distance_sum)
            
        # Find the optimal triplet with the minimum distance

        triplet_array = np.array([np.mean(t, axis=0) for t in triplets])
        
        if len(triplets) == 0:
            print("NO CANDIDATES FOUND: ENTER WINDOW")
            return self.visualize_triplets(np.array([]), self.big_array, self.spiral_array, ball_array)  
        
        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=0.1, min_samples=2).fit(triplet_array)
        labels = clustering.labels_

        # Check the number of clusters
        unique_labels = set(labels)
        
        print(f"DBSCAN unique clusters: {len(unique_labels)}")
        
        if len(unique_labels) > 2:
            print(f"Multiple clusters found: {unique_labels}: ENTER WINDOW")
            # Trigger visualization for the user to pick a point
            return self.visualize_triplets(np.array(triplets), self.big_array, spiral_array2, ball_array)
        
        
        optimal_triplet_idx = np.argmin(triplet_distances)
        direction_vector = mathutils.Vector(triplets[optimal_triplet_idx][0]) - self.current_point
        dir_now_norm = np.array(direction_vector) / np.linalg.norm(direction_vector)
        dir_curr_norm = np.array(self.current_direction) / np.linalg.norm(self.current_direction)
        
        print(f"ADDING: cur_dir: {dir_curr_norm}, now_dir: {dir_now_norm}, dot: {dir_now_norm.dot(dir_curr_norm)}")
        return triplets[optimal_triplet_idx][0]
         
    def create_curve_from_points(self, points, curve_name="GeneratedCurve"):
        # Create a new curve data block
        curve_data = bpy.data.curves.new(name=curve_name, type='CURVE')
        curve_data.dimensions = '3D'

        # Create a new spline
        spline = curve_data.splines.new(type='POLY')  # Use 'NURBS' for smooth curves
        spline.points.add(len(points) - 1)  # Add points to spline

        # Assign the coordinates
        for i, point in enumerate(points):
            spline.points[i].co = (*point, 1.0)  # Blender expects a 4D vector (x, y, z, w)

        # Create an object with this curve data
        curve_object = bpy.data.objects.new(curve_name, curve_data)
        bpy.context.collection.objects.link(curve_object)  # Add to the scene
        return curve_object

    def modal(self, context, event):
           
        if event.type == 'TIMER':
            if not self.has_new_point or not context.scene.five_tp_props.stop_execution:
                self.cancel(context)
                return {'CANCELLED'}
            new_point = None
            
            curve_obj = bpy.data.objects["Spiral_Curve"]
            if self.current_point is None:
                self.current_point = curve_obj.data.splines[0].points[-1].co.xyz
            
            if self.current_direction is None:
                print("update current direction {self.current_direction}")
                self.last_point = curve_obj.data.splines[0].points[-4].co.xyz
                self.current_direction = self.current_point - self.last_point
            
            new_point = self.find_point()
            
            if new_point is None:
                self.has_new_point = False
            
            # Ensure that the new point is valid and apply it to the ball's location
            if new_point is not None:
                
                self.current_lap_num_points += 1
                self.current_lap_points.append(new_point)
                
                self.current_direction = mathutils.Vector(new_point) - self.current_point
                    
                #new lap!:
                if self.current_point[1] > 0 and new_point[1] < 0:
                    self.should_update = True
                    self.past_first = False

                self.current_point = mathutils.Vector(new_point)

                # Add new point to the spiral curve
                bpy.data.objects["Spiral_Curve"].data.splines[0].points.add(count=1)
                bpy.data.objects["Spiral_Curve"].data.splines[0].points[-1].co = (new_point[0], new_point[1], new_point[2], 1.0)   
                
                bpy.context.view_layer.update()
                bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
                
            else:
                print("No new point found, breaking out of the loop.")
                has_new_point = False

        return {'RUNNING_MODAL'}
    
    def fast_cleanup(self):
        # Clear any leftover data or variables
        global labCurrentPoints
        if "labCurrentPoints" in globals():
            del labCurrentPoints

        # Optional: Remove temporary objects if needed
        for obj in bpy.data.objects:
            if obj.name.startswith("Temp_"):
                bpy.data.objects.remove(obj, do_unlink=True)

        # Garbage collection to clear Python memory
        gc.collect()

    def run_algorithm(self):
        # Disable undo and viewport updates for speed
        bpy.context.preferences.edit.use_global_undo = False
        bpy.context.view_layer.depsgraph.update()  # Force depsgraph update just once
        
        # Your core algorithm here
        print("Running algorithm...")

        # Re-enable undo after the run
        bpy.context.preferences.edit.use_global_undo = True


    def execute(self, context):
        bpy.props.stop_execution = False
        
        self.fast_cleanup()
        self.run_algorithm()
        
        if os.path.exists("/Users/jairo/Documents/Spiral/cloud.h5"):
            print("File exists!")
            with h5py.File("/Users/jairo/Documents/Spiral/cloud.h5", "r") as f:
                points = np.array(f["points"])
                self.big_pcd = o3d.geometry.PointCloud()
                self.big_pcd.points = o3d.utility.Vector3dVector(points) 
        else:
            big_model = o3d.io.read_triangle_mesh("/Users/jairo/Documents/Spiral/model.stl")
            points = np.asarray(big_model.vertices)
            # Save as HDF5
            with h5py.File("/Users/jairo/Documents/Spiral/cloud.h5", "w") as f:
                f.create_dataset("points", data=points)
                self.big_pcd = o3d.geometry.PointCloud()
                self.big_pcd.points = o3d.utility.Vector3dVector(np.asarray(points))      

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)

        return {'RUNNING_MODAL'}

    def cancel(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        print("5TP Algorithm execution stopped.")
        
# Define the ModelPrepForSlicing class
class ModelPrepForSlicing:
    def __init__(self, model_obj):
        self.model = model_obj

    def translate_model(self, translation=(0, 0, -0.1)):
        print("translating model 0.1 -z")
        """Translate the model in the -Z direction by 0.1."""
        self.model.location = (self.model.location.x, self.model.location.y, self.model.location.z + translation[2])

    def create_boolean_cube(self):
        print("create boolean cube")
        
        """Create a boolean cube of scale (20, 20, 20) and position it to lie flush with the XY plane."""
        bpy.ops.mesh.primitive_cube_add(size=2)
        cube = bpy.context.object
        cube.scale = (100, 100, 100)
        cube.location = (0, 0, -100)  # Place it so the top lies on the XY plane
        return cube

    def duplicate_and_remesh(self):
        print("Duplicating and remeshing...")

        # Duplicate the model
        duplicated_model = self.model.copy()
        duplicated_model.data = self.model.data.copy()
        bpy.context.collection.objects.link(duplicated_model)

        print(f"Duplicated model: {duplicated_model.name}")

        # Ensure it's active
        bpy.context.view_layer.objects.active = duplicated_model
        duplicated_model.select_set(True)

        # Add Remesh modifier using operators instead
        bpy.ops.object.modifier_add(type='REMESH')
        remesh_modifier = duplicated_model.modifiers[-1]  # Get the last added modifier
        remesh_modifier.mode = 'VOXEL'
        remesh_modifier.voxel_size = 0.2

        print(f"Modifiers on {duplicated_model.name}: {[mod.name for mod in duplicated_model.modifiers]}")

        # Apply the modifier
        bpy.ops.object.modifier_apply(modifier=remesh_modifier.name)

        print(f"Final modifiers on {duplicated_model.name}: {[mod.name for mod in duplicated_model.modifiers]}")

        return duplicated_model
    
    def perform_boolean_difference(self, cube, model):
        print("Performing Boolean Difference...")

        # Ensure the model is active
        bpy.context.view_layer.objects.active = model
        self.model.select_set(True)
        cube.select_set(True)

        # Add Boolean modifier using operators
        
        bpy.ops.object.modifier_add(type='BOOLEAN')
        bool_modifier = model.modifiers[-1]
        
        # Get the last added modifier
        bool_modifier.operation = 'DIFFERENCE'
        bool_modifier.object = cube
        bool_modifier.solver = 'FAST'  # Using fast solver for performance

        # Debugging: Check if the modifier was added
        print(f"Boolean modifier added to {self.model.name}, using {cube.name} as the operand.")

        # Apply the modifier using ops
        bpy.ops.object.modifier_apply(modifier=bool_modifier.name)

        # Debugging: Ensure the modifier was applied
        print(f"Final modifiers on {self.model.name} after applying: {[mod.name for mod in self.model.modifiers]}")

        # Optional: Delete the cube after operation
        bpy.data.objects.remove(cube, do_unlink=True)

        print("Boolean Difference applied successfully.")


    def separate_bottom_face(self, duplicated_model):
        print("Separating bottom face...")

        # Ensure the duplicated model is active and in edit mode
        bpy.context.view_layer.objects.active = duplicated_model
        duplicated_model.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')

        # Switch to object mode temporarily to get mesh data
        bpy.ops.object.mode_set(mode='OBJECT')

        # Get the lowest Z value of the model
        min_z = min(v.co.z for v in duplicated_model.data.vertices)

        # Define a small threshold to account for precision errors
        threshold = 0.0001  # Adjust if needed

        # Switch back to edit mode and use bmesh for face selection
        bpy.ops.object.mode_set(mode='EDIT')
        mesh = bmesh.from_edit_mesh(duplicated_model.data)

        # Deselect everything
        bpy.ops.mesh.select_all(action='DESELECT')

        # Select faces that are within the threshold of the lowest Z value
        for face in mesh.faces:
            if all(abs(vert.co.z - min_z) < threshold for vert in face.verts):
                face.select = True

        # Update the mesh to apply the selection
        bmesh.update_edit_mesh(duplicated_model.data)

        # Separate the selected bottom face
        bpy.ops.mesh.separate(type='SELECTED')

        # Return to object mode and get the separated object
        bpy.ops.object.mode_set(mode='OBJECT')

        # The newly separated object will be the active one
        separated_bottom = bpy.context.view_layer.objects.active

        print(f"Bottom face separated: {separated_bottom.name}")
        return separated_bottom

    def convert_to_curve(self, separated_bottom):
        print("convert to curve")
        
        """Convert the separated bottom face to a curve."""
        bpy.context.view_layer.objects.active = separated_bottom
        separated_bottom.select_set(True)
        bpy.ops.object.convert(target='CURVE')
        
        return separated_bottom
    
    def initial_spiral(self, curve):
        obj = curve
        if obj.type != 'CURVE':
            raise ValueError("Selected object is not a curve.")

        # Ensure we're in Object mode
        bpy.ops.object.mode_set(mode='OBJECT')

        # Access curve data
        curve_data = obj.data
        spline = curve_data.splines[0]  # Assume a single spline

        if not spline.points:
            print("Spline has no points!")
            return

        # Convert curve points to a list of vectors
        points = [mathutils.Vector((p.co.x, p.co.y, p.co.z)) for p in spline.points]

        # Find where the curve crosses the X-axis (y = 0)
        intersection_idx = None
        intersection_co = None

        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i + 1]
            
            # Check if the segment crosses the x-axis
            if (p1.y <= 0 and p2.y >= 0) or (p1.y >= 0 and p2.y <= 0):
                alpha = -p1.y / (p2.y - p1.y)  # Linear interpolation factor
                intersection_co = p1.lerp(p2, alpha)  # Interpolated intersection point
                intersection_idx = i + 1
                break  # Stop at first crossing

        if intersection_idx is None:
            print("No intersection found, curve remains unchanged.")
            return

        print(f"Intersection found at: {intersection_co}")

        # Insert the intersection point at the correct index
        points.insert(intersection_idx, intersection_co)

        # Ensure the curve is open
        spline.use_cyclic_u = False

        # Reorder points so the curve starts at the intersection
        new_order = points[intersection_idx:] + points[:intersection_idx]

        # **Delete the existing spline** (since `.clear()` is not available)
        curve_data.splines.remove(spline)

        # Create a new **Poly** spline
        new_spline = curve_data.splines.new(type='POLY')
        new_spline.points.add(len(new_order) - 1)  # Add points

        for i, p in enumerate(new_order):
            new_spline.points[i].co = (p.x, p.y, p.z, 1.0)  # Ensure correct format

        # Apply smooth Z adjustment
        start_z = new_order[0].z
        desired_z_diff = 0.2
        num_points = len(new_order)

        for i, point in enumerate(new_spline.points):
            z_factor = i / (num_points - 1)
            point.co.z = start_z + desired_z_diff * z_factor

        # Update curve data
        bpy.context.view_layer.objects.active = obj
        bpy.context.view_layer.update()
        
        print("Poly curve adjusted and opened at the x-axis.")

# Define the Operator that will execute the slicing prep
class OBJECT_OT_PrepareModelForSlicing(bpy.types.Operator):
    bl_idname = "object.prepare_model_for_slicing"
    bl_label = "Prepare Model for Slicing"
    
    def execute(self, context):
        print("execute model slicing")
        # Make sure there is an active object
        if bpy.context.active_object:
            model_obj = bpy.context.active_object
            prep = ModelPrepForSlicing(model_obj)
            
            # Perform all steps
            prep.translate_model()
            cube = prep.create_boolean_cube()
            duplicated_model = prep.duplicate_and_remesh()
            prep.perform_boolean_difference(cube, duplicated_model)
            separated_bottom = prep.separate_bottom_face(duplicated_model)
            curve = prep.convert_to_curve(separated_bottom)
            prep.initial_spiral(curve)
            
            self.report({'INFO'}, "Model prepared for slicing successfully.")
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "No active object selected.")
            return {'CANCELLED'}

class TrimToWholeLapOperator(bpy.types.Operator):
    """Trim to the last completed lap"""
    bl_idname = "object.trim_to_whole_lap"
    bl_label = "Trim to Whole Lap"
    
    def execute(self, context):
        # Get the curve object
        curve_obj = bpy.data.objects.get("Spiral_Curve")
        if not curve_obj or curve_obj.type != 'CURVE':
            self.report({'ERROR'}, "Spiral_Curve not found or not a curve object.")
            return {'CANCELLED'}
        
        # Get spline points and last point from properties
        spline = curve_obj.data.splines.active or curve_obj.data.splines[0]
        points = spline.bezier_points if spline.type == 'BEZIER' else spline.points
        last_point = context.scene.five_tp_props.points[-1]
        
        print(f"last_point to find on curve: {last_point}")

        # Convert last_point to a format comparable with curve points
        last_point_coords = (last_point.x, last_point.y, last_point.z)
        
        # Find the index of the last point in the curve
        trim_index = -1
        for i, point in enumerate(points):
            co = point.co if spline.type != 'BEZIER' else point.co.xyz
            if all(abs(a - b) < 1e-2 for a, b in zip(co, last_point_coords)):
                trim_index = i
                break
        
        if trim_index == -1:
            self.report({'ERROR'}, "Last point not found in the curve.")
            return {'CANCELLED'}
        
        for i in range(trim_index + 1, len(points)):
            points[i].select = True  # Select points beyond the last lap

        bpy.ops.object.mode_set(mode='EDIT')
        
        # Now, delete the selected points (vertices)
        bpy.ops.curve.delete(type='VERT')
        
        curve_obj.data.update()  # Ensure Blender knows the data changed
        
        self.report({'INFO'}, f"Trimmed curve at index {trim_index}.")
        return {'FINISHED'}
    
    def trim_points_to_last_lap(points, trim_index):
        # Trim all points after the trim_index (exclusive)
        if len(points) > trim_index + 1:
            points[:] = points[:trim_index + 1]  # This modifies the list in-place to keep only up to the trim_index

class FiveTPPoint(bpy.types.PropertyGroup):
    x: bpy.props.FloatProperty()
    y: bpy.props.FloatProperty()
    z: bpy.props.FloatProperty()

class FiveTPProperties(bpy.types.PropertyGroup):
    
    current_lap_count: bpy.props.IntProperty(
        name="Current Lap Count",
        description="Tracks the number of laps",
        default=0,
        min=0
    )
    
    accuracy_radius: bpy.props.FloatProperty(
        name="Accuracy_Radius",
        description="Tracks the number of laps",
        default=0.25,
        min=0.005,
        max=0.1
    )

    overwrite_factor_start: bpy.props.FloatProperty(
        name="Overwrite Factor Start",
        description="Starting factor for overwriting",
        default=1.0,
        min=0.0,
        max=10.0
    )
    points: bpy.props.CollectionProperty(type=FiveTPPoint)
    stop_execution: bpy.props.BoolProperty(name="Stop Execution", default=False)

class FIVE_TP_OT_ClearPoints(bpy.types.Operator):
    """Clear all stored points"""
    bl_idname = "five_tp.clear_points"
    bl_label = "Clear Saved Points"
    
    def execute(self, context):
        context.scene.five_tp_props.points.clear()
        self.report({'INFO'}, "Cleared all saved points")
        return {'FINISHED'}

class FIVE_TP_OT_StopExecution(bpy.types.Operator):
    bl_idname = "five_tp.stop_execution"
    bl_label = "Stop Execution"
    bl_description = "Stop the execution of the algorithm"
    
    def execute(self, context):
        context.scene.five_tp_props.stop_execution = True
        self.report({'INFO'}, "Execution stopped.")
        return {'FINISHED'}

class ClearFiveTPPoints(bpy.types.Operator):
    bl_idname = "five_tp.clear_points"
    bl_label = "Clear All Points"

    def execute(self, context):
        context.scene.five_tp_props.points.clear()
        return {'FINISHED'}

class PT_FiveTP_PT_Panel(bpy.types.Panel):
    bl_label = "5TP Algorithm"
    bl_idname = "PT_FIVETP_PANEL"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "5TP Algorithm"

    def draw(self, context):
        layout = self.layout
        props = context.scene.five_tp_props
        
        layout.operator("object.prepare_model_for_slicing", text="Prepare Model for Slicing")

        layout.prop(props, "current_lap_count")
        layout.prop(props, "accuracy_radius")
        layout.prop(props, "overwrite_factor_start")
        
        layout.separator()

        layout.label(text="Lap Management")
        layout.operator("object.trim_to_whole_lap")


        layout.separator()
        
        layout.operator("five_tp.execute", text="Run!", icon="PLAY")
        layout.operator("five_tp.stop_execution", text="Stop Execution", icon="CANCEL")
        layout.operator("five_tp.reset_lap_count", text="Reset Lap Count", icon="LOOP_BACK")

        layout.separator()
        
        row = layout.row()
        row.template_list("UI_UL_list", "points_list", context.scene, "points", context.scene, "active_point_index", rows=5)
        
        row = layout.row()
        row.template_list(
            "POINTS_UL_items", "points_list",
            props, "points",
            props, "points_index",
            rows=5
        )

        # Add/Remove buttons
        col = row.column(align=True)
        col.operator("points.add", icon='ADD', text="")
        col.operator("points.remove", icon='REMOVE', text="")

        layout.separator()
        
        row = layout.row()
        row.operator("five_tp.clear_points", icon="TRASH")
        
class POINTS_UL_items(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        # Ensure item is a PTPoint
        if item:
            layout.label(text=f"X: {item.x:.2f}, Y: {item.y:.2f}, Z: {item.z:.2f}")
        else:
            layout.label(text="No point")

class FiveTP_OT_ResetLapCount(bpy.types.Operator):
    bl_idname = "five_tp.reset_lap_count"
    bl_label = "Reset Lap Count"
    bl_description = "Resets the lap count to zero"

    def execute(self, context):
        context.scene.five_tp_props.current_lap_count = 0
        self.report({'INFO'}, "Lap count reset!")
        return {'FINISHED'}

def register():
    bpy.utils.register_class(FiveTPPoint)
    bpy.utils.register_class(FiveTPProperties)
    bpy.types.Scene.five_tp_props = bpy.props.PointerProperty(type=FiveTPProperties)
    bpy.utils.register_class(PT_FiveTP_PT_Panel)
    bpy.utils.register_class(FiveTP_OT_Execute)
    bpy.utils.register_class(FIVE_TP_OT_StopExecution)
    bpy.utils.register_class(FIVE_TP_OT_ClearPoints)
    bpy.utils.register_class(FiveTP_OT_ResetLapCount)
    bpy.utils.register_class(TrimToWholeLapOperator)
    
    bpy.utils.register_class(OBJECT_OT_PrepareModelForSlicing)

def unregister():
    bpy.utils.unregister_class(FiveTPPoint)
    bpy.utils.unregister_class(FiveTP_OT_ResetLapCount)
    bpy.utils.unregister_class(FiveTP_OT_Execute)
    bpy.utils.unregister_class(FIVE_TP_OT_StopExecution)
    bpy.utils.unregister_class(PT_FiveTP_PT_Panel)
    bpy.utils.unregister_class(FIVE_TP_OT_ClearPoints)
    bpy.utils.unregister_class(FiveTPProperties)
    bpy.utils.unregister_class(OBJECT_OT_PrepareModelForSlicing)
    bpy.utils.unregister_class(TrimToWholeLapOperator)
    del bpy.types.Scene.five_tp_props

if __name__ == "__main__":
    register()