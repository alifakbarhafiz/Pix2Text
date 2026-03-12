import numpy as np
import plotly.graph_objects as go


def plot_pointcloud(pc_np, title="Point Cloud"):
    x, y, z = pc_np[:, 0], pc_np[:, 1], pc_np[:, 2]
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=3, color=z, colorscale='Viridis')
    )])
    fig.update_layout(title=title, width=700, height=600)
    fig.show()


def plot_side_by_side(gt_pc, pred_pc, gt_class, top_classes, top_scores, sample_points=1024):
    """GT and prediction point clouds with labels in the title."""
    if gt_pc.shape[0] > sample_points:
        idx = np.random.choice(gt_pc.shape[0], sample_points, replace=False)
        gt_pc = gt_pc[idx]

    if pred_pc.shape[0] > sample_points:
        idx = np.random.choice(pred_pc.shape[0], sample_points, replace=False)
        pred_pc = pred_pc[idx]

    gt_scatter = go.Scatter3d(
        x=gt_pc[:, 0], y=gt_pc[:, 1], z=gt_pc[:, 2],
        mode='markers',
        marker=dict(size=2, color=gt_pc[:, 2], colorscale='Blues'),
        name='Ground Truth'
    )

    gt_caption = f"<b>Ground Truth:</b> {gt_class}"

    pred_scatter = go.Scatter3d(
        x=pred_pc[:, 0], y=pred_pc[:, 1], z=pred_pc[:, 2],
        mode='markers',
        marker=dict(size=2, color=pred_pc[:, 2], colorscale='Reds'),
        name='Prediction'
    )

    pred_caption = "<b>Prediction:</b><br>" + "<br>".join([f"{c}: {s:.3f}" for c, s in zip(top_classes, top_scores)])

    fig = go.Figure(data=[gt_scatter, pred_scatter])
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X'), yaxis=dict(title='Y'), zaxis=dict(title='Z')
        ),
        width=1200,
        height=600,
        title_text=f"{gt_caption} | {pred_caption}",
        title_x=0.5
    )
    fig.show()
