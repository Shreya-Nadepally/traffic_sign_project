#!/usr/bin/env python3
"""
detect.py - small wrapper to run Ultralytics YOLO inference from PowerShell (no here-docs needed).

Example usage (PowerShell):
  python .\detect.py --model .\yolo11n.pt --source data\traffic-sign-to-test.mp4 --save
  python .\detect.py --model runs\detect\train12\weights\best.pt --source datasets\traffic_sign\valid\images --show --save

"""
from pathlib import Path
import argparse
import sys


def parse_args():
    p = argparse.ArgumentParser(description="Run YOLO inference (Ultralytics) from PowerShell-friendly script.")
    p.add_argument("--model", "-m", required=True, help="Path to model weights (.pt)")
    p.add_argument("--source", "-s", required=True, help="Source: image/video file, dir of images, or webcam int (0,1...)")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--device", default=None, help="Device, e.g. 'cpu' or '0' or 'cuda:0' (default: auto)")
    p.add_argument("--save", action="store_true", help="Save inference results to runs/detect")
    p.add_argument("--show", action="store_true", help="Open window to show results (may fail in headless environments)")
    p.add_argument("--save-txt", dest="save_txt", action="store_true", help="Save detection txt files alongside images")
    p.add_argument("--stream", action="store_true", help="Stream results (generator) and process per-frame")
    p.add_argument("--print", dest="print_results", action="store_true", help="Print detection results to the terminal")
    p.add_argument("--name", default=None, help="Optional name for the output run directory (passed to Ultralytics predict 'name' param)")
    return p.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: model file not found: {model_path}")
        sys.exit(2)

    # allow numeric webcam index
    src = args.source
    if src.isnumeric():
        try:
            src = int(src)
        except Exception:
            pass

    try:
        from ultralytics import YOLO
    except Exception as e:
        print("Failed to import ultralytics. Install requirements with: pip install -r requirements.txt")
        print("Error:", e)
        sys.exit(3)

    try:
        model = YOLO(str(model_path))
    except Exception as e:
        print("Failed to load model:", e)
        sys.exit(4)

    predict_kwargs = dict(
        source=src,
        imgsz=args.imgsz,
        conf=args.conf,
        save=args.save,
        save_txt=args.save_txt,
        show=args.show,
    )
    # let user control the output run name (Ultralytics 'name' argument)
    if args.name:
        predict_kwargs["name"] = args.name

    if args.device:
        # newer ultralytics accepts device in predict, older may require model.to(device)
        predict_kwargs["device"] = args.device

    print("Running inference with:")
    print("  model:", model_path)
    for k, v in predict_kwargs.items():
        print(f"  {k}: {v}")

    # If streaming or printing is requested, use stream=True to iterate per frame
    use_stream = args.stream or args.print_results
    if use_stream:
        predict_kwargs_stream = predict_kwargs.copy()
        predict_kwargs_stream["stream"] = True
        # remove device from kwargs if set and handle via model.to
        if "device" in predict_kwargs_stream:
            try:
                model.to(predict_kwargs_stream["device"])
            except Exception:
                pass
            del predict_kwargs_stream["device"]

        try:
            results_gen = model.predict(**predict_kwargs_stream)
        except Exception as e:
            print("Inference failed (stream):", e)
            sys.exit(6)

        # Iterate and optionally print per-frame results
        printed_count = 0
        for i, r in enumerate(results_gen):
            if args.print_results:
                print(f"--- frame {i} ---")
                boxes = getattr(r, "boxes", None)
                if boxes is None:
                    print(" no detections")
                else:
                    try:
                        n = len(boxes)
                    except Exception:
                        n = 0
                    if n == 0:
                        print(" no detections")
                    else:
                        # try to extract cls/conf coordinates safely
                        try:
                            cls_vals = None
                            conf_vals = None
                            if hasattr(boxes, "cls"):
                                try:
                                    cls_vals = boxes.cls.tolist()
                                except Exception:
                                    try:
                                        cls_vals = [int(x) for x in boxes.cls]
                                    except Exception:
                                        cls_vals = None
                            if hasattr(boxes, "conf"):
                                try:
                                    conf_vals = boxes.conf.tolist()
                                except Exception:
                                    try:
                                        conf_vals = [float(x) for x in boxes.conf]
                                    except Exception:
                                        conf_vals = None

                            names = getattr(model, "names", {}) or {}
                            # try to also get bbox coords
                            xyxy_vals = None
                            if hasattr(boxes, "xyxy"):
                                try:
                                    xyxy_vals = boxes.xyxy.tolist()
                                except Exception:
                                    try:
                                        xyxy_vals = [list(map(float, x)) for x in boxes.xyxy]
                                    except Exception:
                                        xyxy_vals = None

                            for idx in range(n):
                                cls_id = None
                                conf = None
                                xy = None
                                if cls_vals is not None:
                                    try:
                                        cls_id = int(cls_vals[idx])
                                    except Exception:
                                        cls_id = cls_vals[idx]
                                if conf_vals is not None:
                                    try:
                                        conf = float(conf_vals[idx])
                                    except Exception:
                                        conf = conf_vals[idx]
                                if xyxy_vals is not None:
                                    try:
                                        xy = xyxy_vals[idx]
                                    except Exception:
                                        xy = None
                                cls_name = names.get(cls_id, str(cls_id)) if cls_id is not None else "?"
                                if xy is not None:
                                    print(f" {idx}: class={cls_name} id={cls_id} conf={conf:.3f} bbox={xy}")
                                else:
                                    print(f" {idx}: class={cls_name} id={cls_id} conf={conf}")
                        except Exception as e:
                            # fallback to printing the boxes object
                            print(repr(boxes))
                printed_count += 1

        # finished streaming
        print(f"Finished streaming inference. Printed frames: {printed_count}.")
        results = None
    else:
        try:
            results = model.predict(**predict_kwargs)
        except TypeError:
            # fallback: set device on model and remove device arg
            try:
                if args.device:
                    model.to(args.device)
                if "device" in predict_kwargs:
                    del predict_kwargs["device"]
                results = model.predict(**predict_kwargs)
            except Exception as e:
                print("Inference failed:", e)
                sys.exit(5)
        except Exception as e:
            print("Inference failed:", e)
            sys.exit(6)

    # summarize
    try:
        n = len(results)
        total = 0
        for r in results:
            if hasattr(r, 'boxes'):
                try:
                    total += len(r.boxes)
                except Exception:
                    # old API
                    try:
                        total += len(r.boxes.xyxy)
                    except Exception:
                        pass
        print(f"Inference completed on {n} source(s). Approx detections: {total}.")
        print("Saved outputs to runs/detect/ (if --save).")
    except Exception:
        print("Inference completed. See Ultralytics output above for details.")


if __name__ == '__main__':
    main()
