import bpy
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from bpy.props import FloatVectorProperty, PointerProperty, CollectionProperty, FloatProperty
from mathutils import Vector, Matrix, Euler
import mathutils
import h5py
import os
import math

def wrap_angle_360(angle):
    """Wrap the angle to be between 0 and 360 degrees."""
    return angle % 360

def constrain_angle(angle, max_angle):
    """Constrain the angle to not exceed the maximum angle."""
    return max(-max_angle, min(max_angle, angle))

class CurvePointData(bpy.types.PropertyGroup):
    a: FloatProperty(name="Phi", default=0.0)
    b: FloatProperty(name="Theta", default=0.0)
    x_proj: FloatProperty(name="X Projected", default=0.0)
    y_proj: FloatProperty(name="Y Projected", default=0.0)
    z_proj: FloatProperty(name="Z Projected", default=0.0)
    
# PropertyGroup to store the curve analysis data
class CurveAnalysisData(bpy.types.PropertyGroup):
    curve_points: CollectionProperty(type=CurvePointData)

class AnalyzeCurveOperator(bpy.types.Operator):
    bl_idname = "object.analyze_curve"
    bl_label = "Analyze Curve"

    def execute(self, context):
        curve_obj = context.object
        if curve_obj.type != 'CURVE':
            self.report({'ERROR'}, "Selected object is not a curve")
            return {'CANCELLED'}

        # Ensure that the curve_analysis_data property is available
        curve_analysis_data = context.scene.curve_analysis_data
        if curve_analysis_data is None:
            self.report({'ERROR'}, "Curve analysis data is not available")
            return {'CANCELLED'}

        # Proceed with curve analysis...
        spline = curve_obj.data.splines[0]
        points = np.array([p.co.xyz for p in spline.points])
        h5_file = "/Users/jairo/Documents/Spiral/path.h5"

        # Load or generate point cloud data
        if os.path.exists(h5_file):
            with h5py.File(h5_file, 'r') as hf:
                pcd_points = np.array(hf['points'])
                pcd_normals = np.array(hf['normals'])
        else:
            model = o3d.io.read_triangle_mesh(bpy.path.abspath("/Users/jairo/Documents/Spiral/model.stl"))
            model.compute_vertex_normals()
            
            pcd_points = np.asarray(model.vertices)
            pcd_normals = np.asarray(model.vertex_normals)

            with h5py.File(h5_file, 'w') as hf:
                hf.create_dataset('points', data=pcd_points)
                hf.create_dataset('normals', data=pcd_normals)

        kd_tree = cKDTree(pcd_points)
        
        

        # Clear any previous data in the collection before adding new ones
        curve_analysis_data.curve_points.clear()

        # Calculate cylindrical coordinates and tilt angles for each point
        for i in range(1, len(points) - 1):
            p0 = points[i - 1]
            p1 = points[i]
            
            _, nearest_idx = kd_tree.query(p1)
            normal = pcd_normals[nearest_idx]
            
            # Calculate the tangent vector
            tangent = p1 - p0
            tangent /= np.linalg.norm(tangent)

            # Calculate the B-axis rotation angle
            b_angle = math.degrees(math.atan2(tangent[1], tangent[0]))

            # Rotate the normal vector by the B-axis angle using 2D rotation formula
            cos_b = math.cos(math.radians(b_angle))
            sin_b = math.sin(math.radians(b_angle))
            rotated_normal_y = normal[1] * cos_b - normal[0] * sin_b
            rotated_normal_z = normal[2]

            # Calculate the A-axis tilt angle
            a_angle = math.degrees(math.atan2(rotated_normal_z, rotated_normal_y))
            a_angle = constrain_angle(a_angle, 75)

            # Calculate the angle psi
            r = math.sqrt(p1[0]**2 + p1[1]**2)
            z_actual = p1[2]
            psi = math.degrees(math.atan2(r, 38.8 + z_actual))

            # Calculate the new r_prime
            r_prime = math.sqrt((z_actual + 38.8)**2 + r**2)

            # Calculate the projected coordinates
            y_proj = r_prime * math.cos(math.radians(a_angle + psi + 90))
            z_proj = r_prime * math.sin(math.radians(a_angle + psi + 90)) - 38.8
            x_proj = 0  # As per your requirement

            # Store the calculated data
            new_point = curve_analysis_data.curve_points.add()
            new_point.x_proj = x_proj
            new_point.y_proj = y_proj
            new_point.z_proj = z_proj
            new_point.a = a_angle
            new_point.b = b_angle
        
        self.report({'INFO'}, "Curve analyzed: cylindrical coordinates and tilt angles stored")
        
        return {'FINISHED'}
    
def draw_debug_line(start, end, color=(1, 0, 0)):
    bpy.ops.object.empty_add(location=start)
    empty_start = bpy.context.object
    bpy.ops.object.empty_add(location=end)
    empty_end = bpy.context.object
    
    line_data = bpy.data.curves.new('debug_line', type='CURVE')
    line_data.dimensions = '3D'
    spline = line_data.splines.new('POLY')
    spline.points.add(1)
    spline.points[0].co = (*start, 1)
    spline.points[1].co = (*end, 1)
    
    line_obj = bpy.data.objects.new('DebugLine', line_data)
    bpy.context.collection.objects.link(line_obj)   

class CreateGCodeOperator(bpy.types.Operator):
    bl_idname = "object.create_gcode"
    bl_label = "Create G-code"
    
    def execute(self, context):
        curve_obj = context.object
        if curve_obj.type != 'CURVE':
            self.report({'ERROR'}, "Selected object is not a curve")
            return {'CANCELLED'}

        # Ensure we access the curve_analysis_data correctly from the scene
        curve_analysis_data = bpy.context.scene.curve_analysis_data

        # Check if the curve_analysis_data exists and contains the necessary data
        if not curve_analysis_data or not curve_analysis_data.curve_points:
            self.report({'ERROR'}, "Curve analysis data not available")
            return {'CANCELLED'}

        # Initialize the G-code commands
        gcode = []
        gcode.append("G21 ; Set units to mm")  # Set the units to millimeters
        gcode.append("G90 ; Absolute positioning")  # Set to absolute positioning
        gcode.append("G92 X0 Y0 Z0 B360 A0 ; Set current position to zero")

        total_rotation = 360.0

        for i in range(len(curve_analysis_data.curve_points)):
            point = curve_analysis_data.curve_points[i]
            x = point.x_proj
            y = point.y_proj
            z = point.z_proj
            a = point.a
            b = point.b

            gcode.append(f"G1 X{x:.3f} Y{y:.3f} Z{z:.3f} B{b:.3f} A{a:.3f} F1000")

        # Finalize the G-code
        gcode.append("M2 ; End of program")

        # Save the G-code to a file
        self.save_gcode(gcode)

        self.report({'INFO'}, "G-code created and saved")
        return {'FINISHED'}

    def save_gcode(self, gcode):
        # Save the generated G-code to a text file
        filepath = bpy.path.abspath("//gcode_output.txt")
        with open(filepath, 'w') as file:
            file.write("\n".join(gcode))  # Write the G-code commands to the file

def update_toolhead_position(self, context):
    # Read the G-code file
    filepath = bpy.path.abspath("//gcode_output.txt")
    if not os.path.exists(filepath):
        return

    with open(filepath, 'r') as file:
        gcode_lines = file.readlines()

    # Extract the XYZBA positions from the G-code
    positions = []
    for line in gcode_lines:
        if line.startswith("G1"):
            parts = line.split()
            x = float(parts[1][1:])
            y = float(parts[2][1:])
            z = float(parts[3][1:])
            b = float(parts[4][1:])
            a = float(parts[5][1:])
            positions.append((x, y, z, b, a))

    # Calculate the index based on the progress
    index = int(self.progress_slider * (len(positions) - 1))
    position = positions[index]

    # Move the toolhead to the desired position
    toolhead = bpy.data.objects["Tool_Head"]
    b_axis = bpy.data.objects["B-Axis"]
    a_axis = bpy.data.objects["A-Axis"]
    if toolhead and b_axis and a_axis:
        print("moving toolhead")
        x = position[0]
        y = position[1]
        z = position[2]
        b_theta = math.radians(position[3])
        a_theta = math.radians(position[4])
        toolhead.location = (x, y, z)
        b_axis.rotation_euler = Euler((0, 0, b_theta), 'XYZ')
        a_axis.rotation_euler = Euler((a_theta, 0, 0), 'XYZ')

class CurveToolsPanel(bpy.types.Panel):
    bl_label = "Curve Tools"
    bl_idname = "OBJECT_PT_curve_tools"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'G-Code'

    def draw(self, context):
        layout = self.layout
        layout.operator("object.analyze_curve")
        layout.operator("object.create_gcode")
        
        # Add the progress slider
        layout.prop(context.scene, "progress_slider", slider=True)

classes = [AnalyzeCurveOperator, CreateGCodeOperator, CurveToolsPanel, CurvePointData, CurveAnalysisData]

def register():
    
    for cls in classes:
        bpy.utils.register_class(cls)
        
    # Register the property on the object (for holding curve data if needed)
    bpy.types.Scene.curve_analysis_data = bpy.props.PointerProperty(type=CurveAnalysisData)
    bpy.types.Scene.progress_slider = bpy.props.FloatProperty(
        name="Progress",
        description="Progress along the curve",
        default=0.0,
        min=0.0,
        max=1.0,
        step=1,
        precision=3,
        update=update_toolhead_position  # Add the update callback
    )

def unregister():
    del bpy.types.Scene.curve_analysis_data
    del bpy.types.Scene.progress_slider
    for cls in classes:
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()
