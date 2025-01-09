import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import AgglomerativeClustering

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


# Define the DINOv2 Image Encoder class
class DINOv2ImageEncoder(nn.Module):
    def __init__(self, size="small", patch_size=14):
        super().__init__()
        """Initialize the DINOv2 encoder."""
        self.size = size
        self.patch_size = patch_size
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vit" + self.size[0] + str(self.patch_size))
        self.embed_dim = self.model.embed_dim
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, images, cls_token=False):
        """
            Encode a batch of images.
            images: (B, 3, H, W)
            return: (B, token, dim)
        """
        assert images.shape[2] % self.patch_size == 0
        assert images.shape[3] % self.patch_size == 0

        self.model.eval()
        x = self.model.prepare_tokens_with_masks(images)
        for block in self.model.blocks:
            x = block(x)
        if cls_token:
            return x[:, 0].unsqueeze(1) # (B, 1, dim)
        else:
            return x[:, 1:] # (B, token, dim)

    @property
    def dtype(self):
        return self.model.dtype

    @property
    def device(self):
        return self.model.device

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
        # print(ann['area'])
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [1.0]])  # 랜덤 색상 + 알파값(투명도 0.5)
        img[m] = color_mask
        # print("INFO: img.shape, m.shape, color_mask.shape", img.shape, m.shape, color_mask.shape)
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

def show_anns_video(bg, anns, fname, cmap, borders=True):
    ax = plt.gca()
    ax.set_autoscale_on(False)

    # img 생성
    img = np.ones((anns[0].shape[1], anns[0].shape[2], 4))
    img[:, :, 3] = 0  # 초기화 (투명도)

    for i, k in enumerate(anns):
        m = anns[k][0]
        color_mask = cmap[i]   # 랜덤 색상 + 알파값(투명도 0.5)
        # print("INFO: img.shape, m.shape, color_mask.shape", img.shape, m.shape, color_mask.shape)
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

def generate_merged_masks(encoder, image, anns):
    device = next(encoder.parameters()).device
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float().to(device) # should be in [c, h, w]
    if image.max() > 1:
        image /= 255.0
    tokens = encoder(image)
    tokens = tokens.view(16, 16, -1) 
    expanded_tokens = tokens.repeat_interleave(14, dim=0).repeat_interleave(14, dim=1) 
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    averaged_tokens = []
    for ann in sorted_anns:
        m = ann['segmentation']
        # mask에서 True인 위치만 추출
        selected_tokens = expanded_tokens[m]  # [N, 384] 형태, N은 True의 개수
        averaged_tokens.append(selected_tokens.mean(dim=0))
    averaged_tokens = torch.stack(averaged_tokens)

    
    # cluster 함수를 사용하여 그룹화
    object_groups = cluster_agglo(averaged_tokens, similarity_threshold=0.8)
    # merge_anns 함수를 사용하여 그룹별로 boolean mask를 합연산 (OR 연산)으로 병합
    merged_anns = merge_anns(sorted_anns, object_groups)
    return merged_anns


def merge_anns(sorted_anns, object_groups):# 그룹별로 boolean mask를 합연산 (OR 연산)으로 병합
    merged_masks = []
    areas = []  # 각 병합된 마스크의 True 개수 저장
    for group in object_groups:
        merged_mask = torch.zeros(sorted_anns[0]['segmentation'].shape, dtype=torch.bool).numpy()  # 빈 mask 초기화
        for idx in group:
            merged_mask |= sorted_anns[idx]['segmentation']  # OR 연산으로 병합
        merged_masks.append(merged_mask)
        areas.append(merged_mask.sum().item())  # True 개수 계산
    

    # 결과 출력
    # print(f"Number of merged masks: {len(merged_masks)}")
    # for i, (merged_mask, area) in enumerate(zip(merged_masks, areas)):
    #     print(f"Group {i + 1}:")
    #     print(f"  Shape: {merged_mask.shape}")
    #     print(f"  Area: {area}")
    
    merged_anns = []
    for i, merged_mask in enumerate(merged_masks):
        merged_anns.append( {
            'segmentation': merged_mask,
            'area': areas[i]
        })
    
    return merged_anns


def cluster(averaged_tokens):
    # Cosine similarity 계산
    similarity_matrix = F.cosine_similarity(
        averaged_tokens.unsqueeze(1),  # [num_masks, 1, feature_dim]
        averaged_tokens.unsqueeze(0),  # [1, num_masks, feature_dim]
        dim=2                          # 두 텐서의 마지막 차원(feature_dim)에서 계산
    )  # 결과: [num_masks, num_masks] (각 마스크 쌍 간 유사도)

    # 특정 임계값 이상인 경우 같은 객체로 판단
    similarity_threshold = 0.7  # 설정된 similarity 임계값
    object_groups = []  # 병합된 그룹 리스트
    visited = set()  # 이미 처리된 마스크를 추적

    for i in range(similarity_matrix.size(0)):
        if i in visited:
            continue
        # i번 마스크와 similarity_threshold 이상인 마스크 찾기
        group = (similarity_matrix[i] >= similarity_threshold).nonzero(as_tuple=True)[0]
        object_groups.append(group.tolist())  # 그룹 추가
        visited.update(group.tolist())  # 방문한 마스크 기록

    # # 결과 출력
    # print(f"Number of merged groups: {len(object_groups)}")
    # for idx, group in enumerate(object_groups):
    #     print(f"Group {idx + 1}: {group}")
    
    return object_groups

def cluster_agglo(averaged_tokens, similarity_threshold=0.8):
    """
    Clustering using Agglomerative Clustering based on cosine similarity.

    Args:
        averaged_tokens (torch.Tensor): [num_masks, feature_dim], features of each mask.
        similarity_threshold (float): Threshold for merging clusters.

    Returns:
        object_groups (List[List[int]]): List of groups, each containing mask indices.
    """
    # Convert PyTorch tensor to NumPy array for scikit-learn
    averaged_tokens_np = averaged_tokens.cpu().numpy()

    # Compute cosine distance matrix (1 - cosine similarity)
    from sklearn.metrics.pairwise import cosine_distances
    distance_matrix = cosine_distances(averaged_tokens_np)

    # Perform Agglomerative Clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,  # Let the threshold define the number of clusters
        metric="precomputed",  # Using the precomputed distance matrix
        linkage="average",  # Complete linkage for maximum distance within clusters
        distance_threshold=1 - similarity_threshold  # Convert similarity threshold to distance
    )
    cluster_labels = clustering.fit_predict(distance_matrix)

    # Group mask indices by their cluster labels
    object_groups = []
    for cluster_id in np.unique(cluster_labels):
        group = np.where(cluster_labels == cluster_id)[0].tolist()
        object_groups.append(group)

    # # 결과 출력
    # print(f"Number of merged groups: {len(object_groups)}")
    # for idx, group in enumerate(object_groups):
    #     print(f"Group {idx + 1}: {group}")

    return object_groups

if __name__ == "__main__":
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

    # load video
    # `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
    version = 'pointmaze'
    video_dir = f"/home/s2/youngjoonjeong/github/sam2/notebooks/videos/{version}"

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
    ]
    # 숫자 부분만 추출하여 정렬
    frame_names.sort(key=lambda p: int(''.join(filter(str.isdigit, os.path.splitext(p)[0]))))

    # open image from the first frame
    image = Image.open(os.path.join(video_dir, frame_names[0]))
    # Resize to 224x224 using LANCZOS
    image = image.resize((224, 224), Image.Resampling.LANCZOS)  
    image = np.array(image.convert("RGB"))

    # load sam2 model and config
    sam2_checkpoint = "../checkpoints/sam2.1_hiera_small.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        stability_score_thresh=0.6,
        # pred_iou_thresh=0.5,
        # use_m2m=True,
    )

    # generate masks
    masks = mask_generator.generate(image)

    # print("INFO: masks", masks[0]['segmentation'].shape)
    print(masks[0].keys())

    print("Number of masks before merge:", len(masks))

    # show masks before merge
    show_anns(image, masks, 'pusht_before_merge.png')

    encoder = DINOv2ImageEncoder(size="small", patch_size=14).to(device)
    merged_masks = generate_merged_masks(encoder, image, masks)

    print("Number of masks after merge:", len(merged_masks))

    # show masks after merge
    show_anns(image, merged_masks, 'pusht_after_merge.png')

    # build video predictor
    sam2_checkpoint = "../checkpoints/sam2.1_hiera_small.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)
    
    # add all masks from merged_masks
    for i, mask in enumerate(merged_masks):
        m = mask['segmentation']
        frame_idx, object_ids, masks = predictor.add_new_mask(
            inference_state=inference_state, 
            mask=m, 
            frame_idx=0, 
            obj_id=i,
        )

    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    # print(video_segments.keys(), video_segments[0].keys(), video_segments[0])

    # render the segmentation results every few frames
    color_mask = [np.concatenate([np.random.random(3), [1.0]]) for _ in range(len(merged_masks))]  # 랜덤 색상 + 알파값(투명도 0.5)
    vis_frame_stride = 1
    plt.close("all")
    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
         # open image from the first frame
        image = Image.open(os.path.join(video_dir, frame_names[out_frame_idx]))
        # Resize to 224x224 using LANCZOS
        image = image.resize((224, 224), Image.Resampling.LANCZOS)  
        image = np.array(image.convert("RGB"))
        show_anns_video(
                        bg=image, 
                        anns=video_segments[out_frame_idx], 
                        fname=f'{version}_{out_frame_idx}.png',
                        cmap=color_mask,
                        )

        






    