"""SMPL mesh viewer widget using pyqtgraph OpenGL.

Adapted from TT_analysis/src/frontend/components/smpl_widget.py.
"""

import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import pyqtSignal, Qt
import pyqtgraph.opengl as gl
from pyqtgraph.opengl.shaders import ShaderProgram, VertexShader, FragmentShader


class _PanGLView(gl.GLViewWidget):
    """GLViewWidget with Ctrl+Left-click = translate (pan)."""

    def mouseMoveEvent(self, ev):
        lmb = bool(ev.buttons() & Qt.LeftButton)
        ctrl = bool(ev.modifiers() & Qt.ControlModifier)
        if lmb and ctrl:
            # Treat as middle-button pan: translate in the view plane
            diff = ev.pos() - self.mousePos
            self.mousePos = ev.pos()
            self.pan(diff.x(), diff.y(), 0, relative="view")
            return
        super().mouseMoveEvent(ev)


class SMPLViewer(QWidget):
    frame_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.view = _PanGLView()
        layout.addWidget(self.view)

        # Camera
        self.view.opts["distance"] = 7.0
        self.view.opts["elevation"] = 10
        self.view.opts["azimuth"] = -90
        self.view.opts["fov"] = 60
        self.view.setBackgroundColor((40, 40, 40))

        # Ground grid
        self.grid = gl.GLGridItem()
        self.grid.scale(0.5, 0.5, 0.5)
        self.view.addItem(self.grid)

        self.mesh_item = None
        self.verts = None       # (T, V, 3)
        self.faces = None       # (F, 3)
        self.current_frame = 0
        self.total_frames = 0

        # Genre / label overlay at the bottom
        self.label_overlay = QLabel("")
        self.label_overlay.setAlignment(Qt.AlignCenter)
        self.label_overlay.setStyleSheet(
            "color: white; background: rgba(0,0,0,160); "
            "font-size: 45px; font-weight: bold; padding: 12px 28px; "
            "border-radius: 6px; margin: 4px;"
        )
        self.label_overlay.setFixedHeight(90)
        layout.addWidget(self.label_overlay)

        # Custom 3-point lighting shader
        self._shader = self._create_shader()

    # ---------------------------------------------------------- shader
    @staticmethod
    def _create_shader():
        vert = """
            varying vec3 normal;
            varying vec3 pos;
            void main() {
                normal = normalize(gl_NormalMatrix * gl_Normal);
                pos = vec3(gl_ModelViewMatrix * gl_Vertex);
                gl_Position = ftransform();
            }
        """
        frag = """
            varying vec3 normal;
            varying vec3 pos;
            void main() {
                vec3 n = normalize(normal);
                // Key light
                vec3 keyDir = normalize(vec3(1.0, 1.0, 1.0));
                float keyDiff = max(dot(n, keyDir), 0.0);
                vec3 keyColor = vec3(0.9, 0.9, 0.9) * keyDiff * 0.9;
                // Fill light
                vec3 fillDir = normalize(vec3(-1.0, 0.5, 1.0));
                float fillDiff = max(dot(n, fillDir), 0.0);
                vec3 fillColor = vec3(0.6, 0.6, 0.7) * fillDiff * 0.5;
                // Rim light
                vec3 rimDir = normalize(vec3(0.0, 1.0, -1.0));
                float rimDiff = max(dot(n, rimDir), 0.0);
                vec3 rimColor = vec3(0.8, 0.8, 0.8) * rimDiff * 0.4;
                // Ambient
                vec3 ambient = vec3(0.3, 0.3, 0.3);
                // Material
                vec3 mat = vec3(0.75, 0.75, 0.75);
                vec3 light = ambient + keyColor + fillColor + rimColor;
                gl_FragColor = vec4(mat * light, 1.0);
            }
        """
        return ShaderProgram("dance_studio_lighting", [
            VertexShader(vert), FragmentShader(frag)
        ])

    # -------------------------------------------------------- public API
    def set_mesh_data(self, verts, faces):
        """Load full animation.

        Parameters
        ----------
        verts : np.ndarray (T, V, 3)
        faces : np.ndarray (F, 3)
        """
        self.verts = verts
        self.faces = faces
        self.total_frames = verts.shape[0]
        self.current_frame = 0

        if self.mesh_item is not None:
            self.view.removeItem(self.mesh_item)

        pos = self._transform(verts[0])
        self.mesh_item = gl.GLMeshItem(
            vertexes=pos,
            faces=faces,
            drawEdges=False,
            smooth=True,
            shader=self._shader,
            glOptions="opaque",
        )
        self.view.addItem(self.mesh_item)

        # Center camera on the mesh
        center = np.mean(pos, axis=0)
        self.view.pan(center[0], center[1], center[2])

    def set_frame(self, idx):
        if self.mesh_item is None or self.verts is None:
            return
        idx = max(0, min(idx, self.total_frames - 1))
        self.current_frame = idx
        pos = self._transform(self.verts[idx])
        self.mesh_item.setMeshData(vertexes=pos, faces=self.faces)

    def set_label_text(self, text):
        """Update the genre/label overlay at the bottom."""
        self.label_overlay.setText(text)

    def clear(self):
        if self.mesh_item is not None:
            self.view.removeItem(self.mesh_item)
            self.mesh_item = None
        self.verts = None
        self.faces = None
        self.total_frames = 0
        self.label_overlay.setText("")

    # ----------------------------------------------------------- helpers
    @staticmethod
    def _transform(v):
        """Flip x, swap y/z so SMPL stands upright in GL coords."""
        return np.stack([-v[:, 0], v[:, 2], v[:, 1]], axis=1)
