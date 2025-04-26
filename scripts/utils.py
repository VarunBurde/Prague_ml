import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cv2
from typing import List, Tuple

def to_homogeneous(points):
    """Convert points to homogeneous coordinates"""
    pad = np.ones((points.shape[:-1] + (1,)), dtype=points.dtype)
    return np.concatenate([points, pad], axis=-1)


def init_figure(height: int = 800) -> go.Figure:
    """Initialize a 3D figure."""
    fig = go.Figure()
    axes = dict(
        visible=False,
        showbackground=False,
        showgrid=False,
        showline=False,
        showticklabels=True,
        autorange=True,
    )
    fig.update_layout(
        template="plotly_white",
        height=height,
        scene_camera=dict(
            eye=dict(x=0.0, y=-0.1, z=2),
            up=dict(x=0, y=1.0, z=0),
            projection=dict(type="orthographic"),
        ),
        scene=dict(
            xaxis=axes,
            yaxis=axes,
            zaxis=axes,
            aspectmode="data",
            dragmode="orbit",
        ),
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        legend=dict(orientation="h", yanchor="top", y=0.99, xanchor="left", x=0.1),
    )
    return fig


def plot_camera(
    fig: go.Figure,
    R: np.ndarray,  # R_wc, from camera to world, size (3,3)
    t: np.ndarray,  # t_wc, from camera to world, size (3,)
    K: np.ndarray,  # size (3,3)
    color: str = "rgb(50, 50, 50)",
    name: str = None,
    legendgroup: str = None,
    fill: bool = False,
    size: float = 1.0,
    text: str = None,
):
    """Plot a camera frustum from pose and intrinsic matrix."""
    W, H = K[0, 2] * 2, K[1, 2] * 2
    corners = np.array([[0, 0], [W, 0], [W, H], [0, H], [0, 0]])
    if size is not None:
        image_extent = max(size * W / 1024.0, size * H / 1024.0)
        world_extent = max(W, H) / (K[0, 0] + K[1, 1]) / 0.5
        scale = 0.5 * image_extent / world_extent
    else:
        scale = 1.0
    corners = to_homogeneous(corners) @ np.linalg.inv(K).T
    corners = (corners / 2 * scale) @ R.T + t
    legendgroup = legendgroup if legendgroup is not None else name

    x, y, z = np.concatenate(([t], corners)).T
    i = [0, 0, 0, 0]
    j = [1, 2, 3, 4]
    k = [2, 3, 4, 1]

    triangles = np.vstack((i, j, k)).T
    vertices = np.concatenate(([t], corners))

    tri_points = np.array([vertices[i] for i in triangles.reshape(-1)])

    top_left = tri_points[1, :]
    top_right = tri_points[2, :]
    bot_left = tri_points[-2, :]
    top_mid = (top_right - top_left) / 2
    left = top_left - bot_left
    notch = top_left + top_mid + left * 0.5

    notch_triplet = np.array([top_left, notch, top_right])
    tri_points = np.concatenate((tri_points, notch_triplet), axis=0)

    x, y, z = tri_points.T

    pyramid = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        legendgroup=legendgroup,
        name=name,
        line=dict(color=color, width=2),
        showlegend=False,
        hovertemplate=text.replace("\n", "<br>") if text else "",
    )
    fig.add_trace(pyramid)


def plot_world_coordinates(fig: go.Figure, size: float = 200.0):
    """Add world coordinate system axes to figure"""
    # Add the x-axis
    fig.add_trace(
        go.Scatter3d(
            x=[0, size],
            y=[0, 0],
            z=[0, 0],
            mode="lines",
            name="x-axis",
            line=dict(color="red", width=4),
        )
    )

    # Add the y-axis
    fig.add_trace(
        go.Scatter3d(
            x=[0, 0],
            y=[0, size],
            z=[0, 0],
            mode="lines",
            name="y-axis",
            line=dict(color="green", width=4),
        )
    )

    # Add the z-axis
    fig.add_trace(
        go.Scatter3d(
            x=[0, 0],
            y=[0, 0],
            z=[0, size],
            mode="lines",
            name="z-axis",
            line=dict(color="blue", width=4),
        )
    )


def visualize_point_cloud_3d(points, colors, camera_poses, K):
    """
    Visualize 3D point cloud with improved camera visualization
    
    Args:
        points: Nx3 array of 3D points
        colors: Nx3 array of RGB colors (0-1 range)
        camera_poses: List of camera poses (each pose is a 3x4 matrix)
    """
    if camera_poses is None and 'camera_poses' in globals():
        camera_poses = globals()['camera_poses']
        
    # Initialize figure
    fig = init_figure(height=800)
    
    # Convert colors from BGR (OpenCV format) to RGB for Plotly
    colors_rgb = colors[:, ::-1]
    
    # Convert RGB colors to hex format for Plotly
    colors_hex = ['rgb({},{},{})'.format(int(r*255), int(g*255), int(b*255)) for r, g, b in colors_rgb]
    
    # Add point cloud
    fig.add_trace(go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=colors_hex,
            opacity=0.8
        ),
        name='Point Cloud'
    ))
    
    # Add camera visualizations
    if camera_poses is not None:
        for i, pose in enumerate(camera_poses):
            # Extract rotation and translation
            R = pose[:, :3]  # 3x3 rotation matrix
            t = pose[:, 3]   # 3x1 translation vector
            
            # For the first camera (identity pose), t is zero, so we don't need to compute cam_pos
            if i == 0:
                cam_pos = np.zeros(3)  # Camera is at origin
                cam_R = np.eye(3)      # Camera has identity rotation
            else:
                # For non-identity poses, compute camera position in world coordinates
                cam_pos = -R.T @ t
                cam_R = R.T            # R_wc is the transpose of R_cw
            
            camera_colors = ["red", "green", "blue", "purple", "orange"]
            color = camera_colors[i % len(camera_colors)]
            
            # Add camera visualization
            plot_camera(
                fig=fig,
                R=cam_R,
                t=cam_pos,
                K=K, 
                color=color,
                name=f"Camera {i+1}",
                size=2.0,
                text=f"Camera {i+1}\nPosition: {cam_pos}"
            )
            
            # Add camera center point
            fig.add_trace(go.Scatter3d(
                x=[cam_pos[0]],
                y=[cam_pos[1]],
                z=[cam_pos[2]],
                mode='markers',
                marker=dict(
                    size=10,
                    color=color,
                    symbol='circle'
                ),
                name=f'Camera {i+1} Center'
            ))
    
    # Add world coordinate system
    plot_world_coordinates(fig, size=1.0)
    
    # Set layout
    fig.update_layout(
        title='3D Reconstruction from Two Views',
        scene=dict(
            aspectmode='data'
        )
    )
    
    fig.show()

def visulize_mesh_ply(vertices, triangles, vertex_colors):
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:,0],
                y=vertices[:,1],
                z=vertices[:,2],
                i=triangles[:,0],
                j=triangles[:,1],
                k=triangles[:,2],
                vertexcolor=vertex_colors,
            )
        ],
        layout=dict(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False)
            )
        )
    )
    fig.show()