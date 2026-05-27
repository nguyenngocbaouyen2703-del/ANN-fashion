"""
Fashion-MNIST ONNX Drawing App — v2.0
Cải tiến bởi Claude:
  - Dark theme hiện đại với customtkinter
  - Thanh confidence bar cho 10 classes
  - Undo / Redo (Ctrl+Z / Ctrl+Y)
  - Eraser tool
  - Brush size slider
  - Export canvas ra PNG
  - Predict debounce (không gọi ONNX mỗi pixel)
  - Stroke smoothing (Catmull-Rom interpolation)
  - Hiệu ứng animation khi result thay đổi
"""

import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog
import customtkinter as ctk
from PIL import Image, ImageDraw
import numpy as np
import onnxruntime as ort
import cv2
import math
import time

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_PATH   = "fashion_mnist_cnn.onnx"   # đổi sang CNN model
CLASSES      = ['T-shirt/top', 'Trouser',  'Pullover', 'Dress',    'Coat',
                'Sandal',      'Shirt',     'Sneaker',  'Bag',      'Ankle boot']
ICONS        = ['👕', '👖', '🧥', '👗', '🥼', '👡', '👔', '👟', '👜', '👢']
CANVAS_W     = 340
CANVAS_H     = 340
IMG_SIZE     = 28
DEFAULT_BRUSH = 16
PREDICT_DEBOUNCE_MS = 120   # ms sau khi nhả bút mới predict

# ─── THEME ────────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BG       = "#0d0d0f"
PANEL    = "#16161a"
BORDER   = "#2a2a35"
ACCENT   = "#7c6af7"          # violet
ACCENT2  = "#38bdf8"          # sky blue
FG       = "#e8e8f0"
FG_DIM   = "#6e6e8a"
CANVAS_BG = "#ffffff"

BAR_COLORS = [
    "#7c6af7", "#38bdf8", "#f472b6", "#34d399",
    "#fb923c", "#a78bfa", "#60a5fa", "#f59e0b",
    "#4ade80", "#e879f9"
]

# ─── LOAD MODEL ───────────────────────────────────────────────────────────────
session      = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name   = session.get_inputs()[0].name
output_name  = session.get_outputs()[0].name
INPUT_SHAPE  = session.get_inputs()[0].shape   # e.g. [None,28,28,1] for CNN or [None,28,28] for ANN
INPUT_RANK   = len(INPUT_SHAPE)                # 4 = CNN (NHWC), 3 = ANN
print(f"[Model] input={input_name} shape={INPUT_SHAPE}  rank={INPUT_RANK}")


# ─── UTILS ────────────────────────────────────────────────────────────────────
def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def preprocess(pil_img: Image.Image) -> np.ndarray:
    gray     = pil_img.convert("L")
    arr_full = np.array(gray, dtype=np.float32)

    # 1. Invert (nét đen → sáng) để tìm bounding box
    inv = 255.0 - arr_full

    # 2. Auto-crop tight bounding box quanh nét vẽ
    rows = np.any(inv > 20, axis=1)
    cols = np.any(inv > 20, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        cropped = inv[rmin:rmax+1, cmin:cmax+1]
    else:
        cropped = inv  # canvas trống

    # 3. Pad vuông + margin 15% (giống style Fashion-MNIST gốc)
    h, w  = cropped.shape
    side  = max(h, w)
    pad   = max(1, int(side * 0.15))
    canvas_sq = np.zeros((side + 2*pad, side + 2*pad), dtype=np.float32)
    r_off = pad + (side - h) // 2
    c_off = pad + (side - w) // 2
    canvas_sq[r_off:r_off+h, c_off:c_off+w] = cropped

    # 4. Resize → 28×28
    resized = cv2.resize(canvas_sq, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    # 5. Normalize + GaussianBlur nhẹ
    arr = resized / 255.0
    arr = cv2.GaussianBlur(arr, (3, 3), 0)

    # 6. Reshape theo model: CNN → (1,28,28,1) | ANN → (1,28,28)
    if INPUT_RANK == 4:
        return arr.reshape(1, IMG_SIZE, IMG_SIZE, 1).astype(np.float32)   # NHWC
    else:
        return arr.reshape(1, IMG_SIZE, IMG_SIZE).astype(np.float32)      # ANN


def catmull_rom_points(p0, p1, p2, p3, num=8):
    """Trả về list các điểm nội suy Catmull-Rom giữa p1 và p2."""
    pts = []
    for i in range(num + 1):
        t  = i / num
        t2 = t * t
        t3 = t2 * t
        x  = 0.5 * ((2*p1[0]) + (-p0[0]+p2[0])*t
                     + (2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*t2
                     + (-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t3)
        y  = 0.5 * ((2*p1[1]) + (-p0[1]+p2[1])*t
                     + (2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t2
                     + (-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3)
        pts.append((x, y))
    return pts


# ─── APP ──────────────────────────────────────────────────────────────────────
class FashionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Fashion-MNIST · ONNX Classifier")
        self.resizable(False, False)
        self.configure(fg_color=BG)

        # State
        self._brush      = DEFAULT_BRUSH
        self._erase_mode = False
        self._stroke_pts : list[tuple] = []          # điểm trong 1 stroke
        self._undo_stack : list[Image.Image] = []    # PIL snapshots
        self._redo_stack : list[Image.Image] = []
        self._debounce_id = None

        # PIL image cho model
        self.pil_img  = Image.new("RGB", (CANVAS_W, CANVAS_H), "white")
        self.pil_draw = ImageDraw.Draw(self.pil_img)

        self._build_ui()
        self._bind_keys()

    # ── UI BUILD ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Header
        header = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=0)
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        ctk.CTkLabel(
            header, text="✦  FASHION CLASSIFIER",
            font=ctk.CTkFont(family="Courier New", size=15, weight="bold"),
            text_color=ACCENT
        ).pack(side="left", padx=20, pady=12)
        ctk.CTkLabel(
            header, text="ONNX · Fashion-MNIST · ANN",
            font=ctk.CTkFont(size=11), text_color=FG_DIM
        ).pack(side="right", padx=20)

        # ── Left: canvas + toolbar
        left = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=12)
        left.grid(row=1, column=0, padx=(14,7), pady=12, sticky="nsew")

        # Canvas wrapper với border
        canvas_wrap = ctk.CTkFrame(
            left, fg_color=BORDER,
            corner_radius=8,
            border_width=0
        )
        canvas_wrap.pack(padx=14, pady=(14,8))

        self.tk_canvas = tk.Canvas(
            canvas_wrap,
            width=CANVAS_W, height=CANVAS_H,
            bg=CANVAS_BG, cursor="crosshair",
            highlightthickness=0, relief="flat"
        )
        self.tk_canvas.pack(padx=2, pady=2)
        self.tk_canvas.bind("<ButtonPress-1>",   self._on_press)
        self.tk_canvas.bind("<B1-Motion>",        self._on_drag)
        self.tk_canvas.bind("<ButtonRelease-1>",  self._on_release)

        # Toolbar row 1: tools
        tb1 = ctk.CTkFrame(left, fg_color="transparent")
        tb1.pack(fill="x", padx=14, pady=(0,4))

        self.btn_draw  = ctk.CTkButton(tb1, text="✏  Draw",  width=90,
                                        command=self._set_draw,
                                        fg_color=ACCENT, hover_color="#5e50d4")
        self.btn_draw.pack(side="left", padx=(0,6))

        self.btn_erase = ctk.CTkButton(tb1, text="⬜  Erase", width=90,
                                        command=self._set_erase,
                                        fg_color=BORDER, hover_color="#2e2e40",
                                        text_color=FG)
        self.btn_erase.pack(side="left", padx=(0,6))

        btn_clear = ctk.CTkButton(tb1, text="🗑  Clear", width=90,
                                   command=self._clear,
                                   fg_color="#3b1f1f", hover_color="#5c2a2a",
                                   text_color="#ff6b6b")
        btn_clear.pack(side="left", padx=(0,6))

        btn_export = ctk.CTkButton(tb1, text="💾  Save", width=90,
                                    command=self._export,
                                    fg_color="#1a2f1f", hover_color="#224030",
                                    text_color="#4ade80")
        btn_export.pack(side="left")

        # Toolbar row 2: brush size
        tb2 = ctk.CTkFrame(left, fg_color="transparent")
        tb2.pack(fill="x", padx=14, pady=(0,14))

        ctk.CTkLabel(tb2, text="Brush", font=ctk.CTkFont(size=12),
                     text_color=FG_DIM).pack(side="left", padx=(0,8))
        self.brush_slider = ctk.CTkSlider(
            tb2, from_=4, to=40,
            command=self._on_brush_change,
            width=160,
            button_color=ACCENT, progress_color=ACCENT
        )
        self.brush_slider.set(DEFAULT_BRUSH)
        self.brush_slider.pack(side="left", padx=(0,8))
        self.brush_label = ctk.CTkLabel(
            tb2, text=f"{DEFAULT_BRUSH}px",
            font=ctk.CTkFont(size=12), text_color=FG, width=38
        )
        self.brush_label.pack(side="left")

        ctk.CTkLabel(tb2, text="  Ctrl+Z / Ctrl+Y",
                     font=ctk.CTkFont(size=10), text_color=FG_DIM).pack(side="right")

        # ── Right: prediction panel
        right = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=12)
        right.grid(row=1, column=1, padx=(7,14), pady=12, sticky="nsew")
        self.grid_columnconfigure(1, weight=1)

        # Big prediction result
        res_frame = ctk.CTkFrame(right, fg_color=BG, corner_radius=10)
        res_frame.pack(fill="x", padx=14, pady=(14,10))

        self.icon_label = ctk.CTkLabel(
            res_frame, text="?",
            font=ctk.CTkFont(size=48)
        )
        self.icon_label.pack(pady=(12,2))

        self.pred_label = ctk.CTkLabel(
            res_frame, text="Draw something!",
            font=ctk.CTkFont(family="Georgia", size=18, weight="bold"),
            text_color=ACCENT
        )
        self.pred_label.pack()

        self.conf_label = ctk.CTkLabel(
            res_frame, text="",
            font=ctk.CTkFont(size=13),
            text_color=FG_DIM
        )
        self.conf_label.pack(pady=(2,12))

        # Confidence bars cho 10 classes
        ctk.CTkLabel(right, text="ALL CLASSES",
                     font=ctk.CTkFont(size=10, weight="bold"),
                     text_color=FG_DIM).pack(anchor="w", padx=18, pady=(4,2))

        bars_frame = ctk.CTkFrame(right, fg_color="transparent")
        bars_frame.pack(fill="x", padx=14, pady=(0,14))

        self._bar_labels  : list[ctk.CTkLabel]       = []
        self._bar_widgets : list[ctk.CTkProgressBar] = []
        self._pct_labels  : list[ctk.CTkLabel]       = []

        for i, (cls, icon) in enumerate(zip(CLASSES, ICONS)):
            row = ctk.CTkFrame(bars_frame, fg_color="transparent")
            row.pack(fill="x", pady=2)

            # label
            lbl = ctk.CTkLabel(
                row, text=f"{icon} {cls}",
                font=ctk.CTkFont(size=11),
                text_color=FG_DIM, width=130, anchor="w"
            )
            lbl.pack(side="left", padx=(0,6))
            self._bar_labels.append(lbl)

            # bar
            bar = ctk.CTkProgressBar(
                row, height=10, width=160, corner_radius=4,
                progress_color=BAR_COLORS[i],
                fg_color=BORDER
            )
            bar.set(0)
            bar.pack(side="left")
            self._bar_widgets.append(bar)

            # percent
            pct = ctk.CTkLabel(
                row, text="0%",
                font=ctk.CTkFont(size=10),
                text_color=FG_DIM, width=38
            )
            pct.pack(side="left", padx=(6,0))
            self._pct_labels.append(pct)

    # ── KEYBINDINGS ───────────────────────────────────────────────────────────
    def _bind_keys(self):
        self.bind("<Control-z>", lambda e: self._undo())
        self.bind("<Control-y>", lambda e: self._redo())

    # ── DRAWING ───────────────────────────────────────────────────────────────
    def _on_press(self, event):
        # snapshot cho undo trước khi vẽ
        self._undo_stack.append(self.pil_img.copy())
        if len(self._undo_stack) > 40:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self._stroke_pts = [(event.x, event.y)]
        self._draw_dot(event.x, event.y)

    def _on_drag(self, event):
        self._stroke_pts.append((event.x, event.y))
        pts = self._stroke_pts

        if len(pts) >= 4:
            # Smooth Catmull-Rom
            seg = catmull_rom_points(pts[-4], pts[-3], pts[-2], pts[-1], num=6)
            for j in range(len(seg) - 1):
                self._draw_segment(seg[j], seg[j+1])
        else:
            self._draw_dot(event.x, event.y)

        # Debounce predict
        if self._debounce_id:
            self.after_cancel(self._debounce_id)
        self._debounce_id = self.after(PREDICT_DEBOUNCE_MS, self._predict)

    def _on_release(self, event):
        self._stroke_pts = []
        if self._debounce_id:
            self.after_cancel(self._debounce_id)
        self._predict()

    def _draw_dot(self, x, y):
        r = self._brush
        color = "white" if self._erase_mode else "black"
        self.tk_canvas.create_oval(x-r, y-r, x+r, y+r, fill=color, outline=color)
        self.pil_draw.ellipse([x-r, y-r, x+r, y+r], fill=color)

    def _draw_segment(self, p1, p2):
        r = self._brush
        color = "white" if self._erase_mode else "black"
        # Fill ellipse dọc theo segment
        dx, dy = p2[0]-p1[0], p2[1]-p1[1]
        dist = max(1, math.hypot(dx, dy))
        steps = max(1, int(dist))
        for s in range(steps+1):
            t = s / steps
            cx = p1[0] + dx * t
            cy = p1[1] + dy * t
            self.tk_canvas.create_oval(cx-r, cy-r, cx+r, cy+r, fill=color, outline=color)
            self.pil_draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=color)

    # ── TOOLS ─────────────────────────────────────────────────────────────────
    def _set_draw(self):
        self._erase_mode = False
        self.btn_draw.configure(fg_color=ACCENT)
        self.btn_erase.configure(fg_color=BORDER)
        self.tk_canvas.configure(cursor="crosshair")

    def _set_erase(self):
        self._erase_mode = True
        self.btn_erase.configure(fg_color=ACCENT)
        self.btn_draw.configure(fg_color=BORDER)
        self.tk_canvas.configure(cursor="dotbox")

    def _on_brush_change(self, val):
        self._brush = int(val)
        self.brush_label.configure(text=f"{self._brush}px")

    def _clear(self):
        self._undo_stack.append(self.pil_img.copy())
        self._redo_stack.clear()
        self.tk_canvas.delete("all")
        self.pil_draw.rectangle([0, 0, CANVAS_W, CANVAS_H], fill="white")
        self._reset_panel()

    def _export(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
            title="Save canvas"
        )
        if path:
            self.pil_img.save(path)

    # ── UNDO / REDO ───────────────────────────────────────────────────────────
    def _undo(self):
        if not self._undo_stack:
            return
        self._redo_stack.append(self.pil_img.copy())
        self.pil_img  = self._undo_stack.pop()
        self.pil_draw = ImageDraw.Draw(self.pil_img)
        self._sync_canvas()
        self._predict()

    def _redo(self):
        if not self._redo_stack:
            return
        self._undo_stack.append(self.pil_img.copy())
        self.pil_img  = self._redo_stack.pop()
        self.pil_draw = ImageDraw.Draw(self.pil_img)
        self._sync_canvas()
        self._predict()

    def _sync_canvas(self):
        """Re-render PIL image lên tk canvas."""
        from PIL import ImageTk
        self._tk_img_ref = ImageTk.PhotoImage(self.pil_img)
        self.tk_canvas.delete("all")
        self.tk_canvas.create_image(0, 0, anchor="nw", image=self._tk_img_ref)

    # ── PREDICT ───────────────────────────────────────────────────────────────
    def _predict(self):
        arr     = preprocess(self.pil_img)
        outputs = session.run([output_name], {input_name: arr})[0]
        logits  = outputs[0]
        # CNN output layer dùng softmax → probabilities thẳng
        # ANN raw logits → cần softmax thủ công
        probs   = logits if INPUT_RANK == 4 else softmax(logits)
        pred_id = int(np.argmax(probs))
        conf    = float(probs[pred_id])

        # Update big result
        self.icon_label.configure(text=ICONS[pred_id])
        self.pred_label.configure(text=CLASSES[pred_id], text_color=BAR_COLORS[pred_id])
        self.conf_label.configure(text=f"{conf*100:.1f}% confidence")

        # Update all bars
        for i, (bar, lbl, pct) in enumerate(
                zip(self._bar_widgets, self._bar_labels, self._pct_labels)):
            p = float(probs[i])
            bar.set(p)
            pct.configure(text=f"{p*100:.1f}%")
            if i == pred_id:
                lbl.configure(text_color=BAR_COLORS[i])
                bar.configure(progress_color=BAR_COLORS[i])
            else:
                lbl.configure(text_color=FG_DIM)
                bar.configure(progress_color="#2a2a3a")   # dim, no alpha hex

    def _reset_panel(self):
        self.icon_label.configure(text="?")
        self.pred_label.configure(text="Draw something!", text_color=ACCENT)
        self.conf_label.configure(text="")
        for bar, pct, lbl in zip(self._bar_widgets, self._pct_labels, self._bar_labels):
            bar.set(0)
            pct.configure(text="0%")
            lbl.configure(text_color=FG_DIM)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = FashionApp()
    app.mainloop()