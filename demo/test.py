import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

np.random.seed(3)

def show_anns(bg, anns, fname, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    # img 생성
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0  # 초기화 (투명도)

    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [1.0]])  # 랜덤 색상 + 알파값(투명도 0.5)
        img[m] = color_mask
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    # img와 bg를 matplotlib로 표시
    plt.figure(figsize=(10, 10))
    plt.imshow(bg)  # 배경 표시
    plt.imshow(img, alpha=1)  # 오버레이 (alpha로 투명도 설정)
    plt.axis('off')  # 축 숨기기

    # 결과를 로컬 파일로 저장
    output_path = f'/home/s2/youngjoonjeong/github/sam2/demo/{fname}'
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

image = Image.open('/home/s2/youngjoonjeong/github/sam2/notebooks/images/maze.png')
image = np.array(image.convert("RGB"))

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

sam2_checkpoint = "../checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2,
    # pred_iou_thresh=0.5,
    # use_m2m=True,
)

mask_generator_3 = SAM2AutomaticMaskGenerator(
    model=sam2,
    pred_iou_thresh=0.5,
)

mask_generator_2 = SAM2AutomaticMaskGenerator(model=sam2)

masks = mask_generator.generate(image)
masks_3 = mask_generator_3.generate(image)

print("INFO: masks", masks[0]['segmentation'].shape)
print(masks[0].keys())

print("Number of masks:", len(masks))
print("Number of masks_2:", len(masks_3))

show_anns(image, masks, 'maze_1.png')
show_anns(image, masks_3, 'maze_2.png')