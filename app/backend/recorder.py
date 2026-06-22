"""Record the SMPL visualization to MP4 with audio."""

import subprocess
import tempfile
import os
import numpy as np
from pathlib import Path


def render_frames(verts, faces, tmpdir, fps=30, resolution=(1280, 720),
                  progress_callback=None, viewer=None):
    """Render SMPL mesh frames to PNG files. MUST be called on the main thread.

    Parameters
    ----------
    viewer : SMPLViewer, optional
        Live viewer to copy camera state and shader from.

    Returns the frame file pattern (e.g. '/tmp/.../frame_%06d.png').
    """
    import pyqtgraph.opengl as gl
    from PyQt5.QtWidgets import QApplication
    from app.frontend.components.smpl_viewer import SMPLViewer

    app = QApplication.instance()
    width, height = resolution
    total_frames = verts.shape[0]

    # Create an offscreen-capable GL widget
    view = gl.GLViewWidget()
    view.resize(width, height)
    view.setBackgroundColor((40, 40, 40))

    # Copy camera state from the live viewer if available
    if viewer is not None:
        for key in ("distance", "elevation", "azimuth", "fov", "center"):
            view.opts[key] = viewer.view.opts[key]
    else:
        view.opts["distance"] = 7.0
        view.opts["elevation"] = 10
        view.opts["azimuth"] = -90
        view.opts["fov"] = 60

    grid = gl.GLGridItem()
    grid.scale(0.5, 0.5, 0.5)
    view.addItem(grid)

    def transform(v):
        return np.stack([-v[:, 0], v[:, 2], v[:, 1]], axis=1)

    # Use the same 3-point lighting shader as the live viewer
    shader = SMPLViewer._create_shader()
    mesh = gl.GLMeshItem(
        vertexes=transform(verts[0]),
        faces=faces,
        drawEdges=False,
        smooth=True,
        shader=shader,
        glOptions="opaque",
    )
    view.addItem(mesh)
    view.show()

    # Let the widget fully initialize its GL context
    app.processEvents()

    frame_pattern = os.path.join(tmpdir, "frame_%06d.png")
    for i in range(total_frames):
        mesh.setMeshData(vertexes=transform(verts[i]), faces=faces)
        view.repaint()              # request a paint through the normal pipeline
        app.processEvents()         # flush the event queue so paintGL runs
        fb = view.grabFramebuffer()
        fb.save(frame_pattern % i, "PNG")
        if progress_callback and i % 10 == 0:
            progress_callback(
                f"Rendering frame {i}/{total_frames}",
                i / total_frames * 0.8,
            )

    view.close()
    return frame_pattern


def encode_video(frame_pattern, audio_path, output_path, fps=30,
                 progress_callback=None):
    """Encode PNGs + audio to MP4 with ffmpeg.  Safe to call from any thread."""
    tmpdir = os.path.dirname(frame_pattern)

    if progress_callback:
        progress_callback("Encoding video...", 0.85)

    video_only = os.path.join(tmpdir, "video_only.mp4")
    cmd_video = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", frame_pattern,
        "-c:v", "mpeg4",
        "-q:v", "2",
        "-pix_fmt", "yuv420p",
        video_only,
    ]
    result = subprocess.run(cmd_video, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg video encoding failed:\n{result.stderr.decode(errors='replace')}"
        )

    if progress_callback:
        progress_callback("Adding audio...", 0.93)

    cmd_mux = [
        "ffmpeg", "-y",
        "-i", video_only,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(output_path),
    ]
    result = subprocess.run(cmd_mux, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg muxing failed:\n{result.stderr.decode(errors='replace')}"
        )

    if progress_callback:
        progress_callback("Done.", 1.0)


# Legacy wrapper kept for backwards compatibility
def record_video(verts, faces, audio_path, output_path,
                 fps=30, resolution=(1280, 720), progress_callback=None):
    """Full render + encode pipeline.  MUST run on the main thread."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pattern = render_frames(
            verts, faces, tmpdir, fps, resolution, progress_callback,
        )
        encode_video(pattern, audio_path, output_path, fps, progress_callback)
