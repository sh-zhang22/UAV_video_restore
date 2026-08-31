"""用 YOLO 从视频生成 pixel-space mask，供 seedvr2_3b_ctrl variant 消费。

用法：
    # 最简：全 COCO 类别、默认阈值
    python make_mask_from_video.py --in_video test.mp4 --out_mask mask_yolo.mp4

    # 只保留 person + car，膨胀 4 像素
    python make_mask_from_video.py \\
        --in_video test.mp4 --out_mask mask_yolo.mp4 \\
        --classes person car --conf 0.25 --dilate 4 --device cuda:1

    # 用自己的 UAV 微调 ckpt
    python make_mask_from_video.py \\
        --in_video test.mp4 --out_mask mask_yolo.mp4 \\
        --yolo_ckpt path/to/my_uav_yolo.pt
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from methods._yolo_mask import video_to_mask, save_mask_mp4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_video", required=True, help="输入视频路径")
    parser.add_argument("--out_mask", required=True, help="输出 mask.mp4 路径")
    parser.add_argument("--yolo_ckpt", default="yolov8n.pt",
                        help="YOLO ckpt 路径（首次自动从 GH release 下载）")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="保留类别名（COCO 80 类），不指定=保留全部")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值")
    parser.add_argument("--dilate", type=int, default=0, help="mask 膨胀像素数")
    parser.add_argument("--device", default="cuda:0", help="YOLO 推理设备")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO 内部推理分辨率")
    parser.add_argument("--fps", type=float, default=29.0, help="输出 mp4 帧率")
    parser.add_argument("--verbose", action="store_true", help="打开 ultralytics 日志")
    args = parser.parse_args()

    t0 = time.time()
    mask = video_to_mask(
        in_video=args.in_video,
        yolo_ckpt=args.yolo_ckpt,
        classes=args.classes,
        conf=args.conf,
        iou=args.iou,
        dilate=args.dilate,
        device=args.device,
        imgsz=args.imgsz,
        verbose=args.verbose,
    )
    save_mask_mp4(mask, args.out_mask, fps=args.fps)
    print(f"DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
