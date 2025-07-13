import os
import uuid
import cv2
import numpy as np
import trimesh
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app, resources={r"/process-blueprint": {"origins": "*"}})

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_walls_from_blueprint(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_image = clahe.apply(image)

    blurred_image = cv2.bilateralFilter(equalized_image, 9, 75, 75)
    edges = cv2.Canny(blurred_image, 30, 200)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    def is_valid_wall_contour(cnt):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        return (area > 500 and perimeter > 100 and len(cnt) > 4)

    filtered_contours = [cnt for cnt in contours if is_valid_wall_contour(cnt)]
    sorted_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)

    if not sorted_contours:
        raise ValueError("No valid wall contours found")

    boundary_wall = sorted_contours[0]
    inner_walls = sorted_contours[1:]

    boundary_coords = [point[0].tolist() for point in boundary_wall]
    inner_wall_coords = [[point[0].tolist() for point in wall] for wall in inner_walls]

    return boundary_coords, inner_wall_coords

def create_platform_vertices_and_faces(boundary_vertices, platform_length, platform_width, platform_thickness):
    x_coords = [v[0] for v in boundary_vertices]
    y_coords = [v[1] for v in boundary_vertices]
    center_x = (min(x_coords) + max(x_coords)) / 2
    center_y = (min(y_coords) + max(y_coords)) / 2

    half_length = platform_length / 2
    half_width = platform_width / 2

    platform_vertices = [
        [center_x - half_length, center_y - half_width, 0],
        [center_x + half_length, center_y - half_width, 0],
        [center_x + half_length, center_y + half_width, 0],
        [center_x - half_length, center_y + half_width, 0],
        [center_x - half_length, center_y - half_width, platform_thickness],
        [center_x + half_length, center_y - half_width, platform_thickness],
        [center_x + half_length, center_y + half_width, platform_thickness],
        [center_x - half_length, center_y + half_width, platform_thickness]
    ]

    platform_faces = [
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
        [0, 4, 7], [0, 7, 3],
        [1, 5, 6], [1, 6, 2],
        [3, 2, 6], [3, 6, 7],
        [0, 1, 5], [0, 5, 4]
    ]

    return platform_vertices, platform_faces

def create_3d_walls_with_solid_walls(boundary_vertices, inner_wall_coords, height, wall_thickness, platform_length=None, platform_width=None, platform_thickness=5.0, gap=1.0):
    def create_solid_wall_mesh(vertices, wall_height, wall_thickness):
        wall_vertices = []
        for v in vertices:
            wall_vertices.append([v[0], v[1], 0])
        for v in vertices:
            wall_vertices.append([v[0], v[1], wall_height])

        wall_faces = []
        vertex_count = len(vertices)
        for i in range(vertex_count):
            next_i = (i + 1) % vertex_count
            wall_faces.append([i, next_i, vertex_count + next_i])
            wall_faces.append([i, vertex_count + next_i, vertex_count + i])

        return trimesh.Trimesh(vertices=wall_vertices, faces=wall_faces)

    if platform_length is None or platform_width is None:
        x_coords = [v[0] for v in boundary_vertices]
        y_coords = [v[1] for v in boundary_vertices]
        padding = 20
        platform_length = (max(x_coords) - min(x_coords)) + 2 * padding
        platform_width = (max(y_coords) - min(y_coords)) + 2 * padding

    wall_meshes = []
    platform_vertices, platform_faces = create_platform_vertices_and_faces(boundary_vertices, platform_length, platform_width, platform_thickness)
    platform_mesh = trimesh.Trimesh(vertices=platform_vertices, faces=platform_faces)
    wall_meshes.append(platform_mesh)

    boundary_wall_mesh = create_solid_wall_mesh(boundary_vertices, height, wall_thickness)
    wall_meshes.append(boundary_wall_mesh)

    for inner_wall in inner_wall_coords:
        inner_wall_mesh = create_solid_wall_mesh(inner_wall, height, wall_thickness)
        wall_meshes.append(inner_wall_mesh)

    combined_mesh = trimesh.util.concatenate(wall_meshes)
    return combined_mesh

@app.route('/process-blueprint', methods=['POST'])
def process_blueprint():
    filepath = None
    stl_filename = None
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            boundary_coords, inner_wall_coords = extract_walls_from_blueprint(filepath)
            if not boundary_coords:
                return jsonify({'error': 'No boundary coordinates found'}), 400

            wall_height = 50.0
            wall_thickness = 25
            platform_thickness = 1.0
            wall_platform_gap = 0.1
            platform_length = 1000.0
            platform_width = 1000.0

            mesh = create_3d_walls_with_solid_walls(
                boundary_coords, inner_wall_coords, wall_height, wall_thickness,
                platform_length, platform_width, platform_thickness, wall_platform_gap
            )

            stl_filename = os.path.join(app.config['UPLOAD_FOLDER'], 
                                        f'floorplan_3d_model_{uuid.uuid4()}.stl')
            mesh.export(stl_filename)

            return send_file(stl_filename, mimetype='model/stl', as_attachment=False)

        return jsonify({'error': 'Invalid file type'}), 400

    except Exception as e:
        print(f"Error processing blueprint: {e}")
        return jsonify({'error': str(e)}), 500

    finally:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
        if stl_filename and os.path.exists(stl_filename):
            try:
                os.remove(stl_filename)
            except PermissionError:
                print(f"Warning: Unable to delete file {stl_filename}. It may still be in use.")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
