"""Section CV · Computer vision passenger counting (counts only, no images kept)

Generated from the notebook (cells 14). Modules in this package run in ONE shared
namespace, in the order listed in transit_ai/__init__.py, exactly like running the
notebook top to bottom. Names used here may therefore be defined in an earlier module.
"""
# ruff: noqa

# ----------------------------------------------------------------------------------------------------
# CV · Computer vision passenger counting (pretrained person detector → anonymized counts)   [notebook cell 14]
# ----------------------------------------------------------------------------------------------------
USE_CV = param("USE_CV", True)  # env-var override; notebook form field
CV_SCORE_THRESHOLD = param("CV_SCORE_THRESHOLD", 0.75)  # env-var override; notebook form field
CV_VIDEO_FRAMES = param("CV_VIDEO_FRAMES", 8)  # env-var override; notebook form field
SHOW_CV_PREVIEW = param("SHOW_CV_PREVIEW", False)  # env-var override; notebook form field
import numpy as np

# Sample transit imagery, downloaded at runtime. Only person counts leave this cell; images are not saved.
#  - bus.jpg: riders at the curb beside a city bus (Ultralytics sample asset, AGPL-3.0)
#  - vtest.avi: overhead camera of pedestrians on a walkway (OpenCV sample data), used as a station-approach camera
CV_SOURCES = [
    {"camera_id": "VISION-STOP-NEWARK-PENN", "kind": "image", "role": "bus stop curb camera",
     "url": "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg",
     "stop_id": "NEWARK_PENN_BUS_LANE", "capacity": 50,
     "note": "generic street photo used as a stand-in for a Newark Penn curb camera"},
    {"camera_id": "VISION-APPROACH-NEWARK-PENN", "kind": "video", "role": "station approach camera",
     "url": "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi",
     "stop_id": "NEWARK_PENN_APPROACH", "capacity": 50,
     "note": "public pedestrian video used as a stand-in for a station approach camera"},
]
CV_DIR = os.path.join(OUTPUT_DIR, "cv_cache")      # downloaded samples only; nothing derived from them is written here
os.makedirs(CV_DIR, exist_ok=True)
_DETECTOR = {"name": None, "model": None, "error": []}


def load_detector():
    """Faster R-CNN (MobileNetV3, COCO) from torchvision; OpenCV's HOG people detector if torchvision weights fail."""
    if _DETECTOR["name"]:
        return _DETECTOR["name"]
    try:
        import torch, torchvision
        from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
        w = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        m = fasterrcnn_mobilenet_v3_large_fpn(weights=w).eval()
        _DETECTOR.update(name=f"torchvision Faster R-CNN MobileNetV3 ({w.name}, COCO)", model=("torch", m, w))
        return _DETECTOR["name"]
    except Exception as e:
        _DETECTOR["error"].append(f"torchvision: {type(e).__name__}: {str(e)[:120]}")
    try:
        import cv2
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        _DETECTOR.update(name="OpenCV HOG + linear SVM people detector (fallback)", model=("hog", hog, None))
        return _DETECTOR["name"]
    except Exception as e:
        _DETECTOR["error"].append(f"opencv: {type(e).__name__}: {str(e)[:120]}")
    return None


def detect_people(rgb):
    """rgb: HxWx3 uint8 array → list of (x1, y1, x2, y2, score) for detected people."""
    kind, model, weights = _DETECTOR["model"]
    if kind == "torch":
        import torch
        t = torch.from_numpy(rgb).permute(2, 0, 1)
        with torch.no_grad():
            out = model([weights.transforms()(t)])[0]
        person = weights.meta["categories"].index("person")
        return [(*map(float, b), float(s)) for b, s, l in zip(out["boxes"], out["scores"], out["labels"])
                if int(l) == person and float(s) >= CV_SCORE_THRESHOLD]
    import cv2
    # HOG uses a fixed 64x128 window: small people need the full-size frame, large people a downscaled one
    h, w = rgb.shape[:2]
    hog_threshold = 0.5   # HOG scores are SVM margins, not probabilities
    keep = []
    for f in sorted({1.0, min(1.0, 480.0 / max(h, w))}):
        img = cv2.resize(rgb, (int(w * f), int(h * f))) if f < 1 else rgb
        rects, wts = model.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), winStride=(4, 4), padding=(8, 8), scale=1.05)
        keep += [(x / f, y / f, (x + bw) / f, (y + bh) / f, float(sc)) for (x, y, bw, bh), sc in zip(rects, np.ravel(wts))
                 if float(sc) >= hog_threshold]
    return _nms(keep)


def _nms(boxes, iou=0.4):
    """Drop overlapping duplicate detections of the same person."""
    boxes = sorted(boxes, key=lambda b: -b[4])
    kept = []
    for b in boxes:
        def ov(a, c):
            ix = max(0, min(a[2], c[2]) - max(a[0], c[0])); iy = max(0, min(a[3], c[3]) - max(a[1], c[1]))
            inter = ix * iy
            return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (c[2] - c[0]) * (c[3] - c[1]) - inter + 1e-9)
        if all(ov(b, k) < iou for k in kept):
            kept.append(b)
    return kept


def _download(url):
    path = os.path.join(CV_DIR, os.path.basename(url))
    if not os.path.exists(path):
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    return path


def _frames(src, path):
    import cv2
    if src["kind"] == "image":
        img = cv2.imread(path)
        if img is None:
            raise ValueError("unreadable image")
        yield 0, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    for idx in np.linspace(0, n - 1, max(1, CV_VIDEO_FRAMES)).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            yield int(idx), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()


def _privacy_preview(rgb, boxes, block=None):
    """Privacy: the WHOLE frame is pixelated (so a person the detector missed is not identifiable either),
    then detection boxes are drawn on top. Display only; never saved or passed downstream."""
    import cv2
    h, w = rgb.shape[:2]
    block = block or max(10, w // 60)
    small = cv2.resize(rgb, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_AREA)
    out = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    for x1, y1, x2, y2, _ in boxes:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (250, 204, 21), max(2, w // 200))
    return out


def run_computer_vision(sources=CV_SOURCES):
    """Each camera → anonymized aggregate: people counted, load factor, detector confidence. No images or identities kept."""
    results = {"detector": None, "cameras": [], "errors": [], "ran_at": iso(utcnow()), "previews": []}
    if not USE_CV:
        results["errors"].append("computer vision disabled")
        return results
    results["detector"] = load_detector()
    if not results["detector"]:
        results["errors"] += _DETECTOR["error"]
        return results
    for src in sources:
        try:
            path = _download(src["url"])
            counts, confs, preview = [], [], None
            for idx, rgb in _frames(src, path):
                boxes = detect_people(rgb)
                counts.append(len(boxes))
                confs += [b[4] for b in boxes]
                if preview is None or len(boxes) > preview[1]:
                    preview = (_privacy_preview(rgb, boxes), len(boxes), idx)
            if not counts:
                raise ValueError("no frames read")
            people = int(round(statistics.median(counts)))
            results["cameras"].append({
                "camera_id": src["camera_id"], "role": src["role"], "stop_id": src["stop_id"],
                "frames_analyzed": len(counts), "people_per_frame": counts, "person_count": people,
                "peak_count": max(counts), "capacity": src["capacity"], "load_factor": round(people / src["capacity"], 2),
                "mean_detector_score": round(float(np.mean(confs)), 2) if confs else None,   # probability for Faster R-CNN; SVM margin for HOG
                "observed_at": iso(utcnow()), "identities_stored": False, "images_stored": False, "note": src.get("note", ""),
            })
            if SHOW_CV_PREVIEW and preview is not None:
                results["previews"].append((src["camera_id"], preview))
        except Exception as e:
            results["errors"].append(f"{src['camera_id']}: {type(e).__name__}: {str(e)[:120]}")
    return results


def show_cv_previews(results):
    if not results["previews"]:
        return
    fig, axes = plt.subplots(1, len(results["previews"]), figsize=(6 * len(results["previews"]), 5))
    axes = np.atleast_1d(axes)
    for ax, (cam, (img, n, idx)) in zip(axes, results["previews"]):
        ax.imshow(img); ax.axis("off")
        ax.set_title(f"{cam}\n{n} {'person' if n == 1 else 'people'} detected (frame {idx}) · frame pixelated for privacy", fontsize=9)
    plt.tight_layout(); plt.show()


CV_RESULTS = run_computer_vision()
print("Detector:", CV_RESULTS["detector"] or "unavailable")
for cam in CV_RESULTS["cameras"]:
    print(f"  {cam['camera_id']} ({cam['role']}): {cam['person_count']} counted (median of {cam['frames_analyzed']} frame(s), "
          f"peak {cam['peak_count']}), load factor {cam['load_factor']}, mean detector score {cam['mean_detector_score']}")
    print(f"      imagery: {cam['note']}")
for err in CV_RESULTS["errors"]:
    print("  ⚠️", err)
show_cv_previews(CV_RESULTS)
_ = CV_RESULTS.pop("previews", None)   # pixelated previews are display-only and are not carried into any case record
