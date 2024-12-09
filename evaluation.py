import torch


def get_mask_from_bounding_box(T, bounding_box):
    mask = torch.zeros((T.shape[1], T.shape[2]))
    x0, y0, x1, y1 = bounding_box

    mask[y0:y1+1, x0:x1+1] = 1

    return mask
        

def get_alignment_score_object(tensor, start, end, bounding_box):
    """Calculate the alignment score for objects as in
    Khorrami & Rasanen, 2021."""

    # Compute the sum of each frame along (H, W) dimensions
    frame_sums = tensor.view(tensor.shape[0], -1).sum(dim=1, keepdim=True)
    
    # Avoid division by zero
    frame_sums = torch.where(frame_sums == 0, torch.ones_like(frame_sums), frame_sums)
    
    # Normalize each frame
    T = tensor / frame_sums.view(tensor.shape[0], 1, 1)

    x0, y0, x1, y1 = bounding_box

    return T[start:end+1, y0:y1+1, x0:x1+1].sum() / (end + 1 - start)

    
def get_alignment_score_word(tensor, start, end, bounding_box):
    """Calculate the alignment score for words as in Khorrami & Rasanen, 2021.
    """

    # Normalize each frame of the tensor to sum to 1
    T = tensor / (tensor.sum(axis=0, keepdim=True) + 1e-06)
    x0, y0, x1, y1 = bounding_box

    score = T[start:end+1, y0:y1+1, x0:x1+1].sum() / (abs(x1 - x0) * abs(y1 - y0))
    return score
    

def get_glancing_score_object(tensor, start, end, bounding_box):
    """Calculate the glancing score for objects as in Khorrami & Rasanen, 2021.
    """
    A = tensor[start:end+1].sum(axis=0)
    A = A / A.sum()

    mask = get_mask_from_bounding_box(tensor, bounding_box)
    return (A * mask).sum()

    
def get_glancing_score_word(tensor, start, end, bounding_box):
    """Calculate the glancing score for words as in
    Khorrami & Rasanen, 2021.
    """
    mask = get_mask_from_bounding_box(tensor, bounding_box)
    mask = mask.unsqueeze(0)
    a = torch.sum(tensor * mask, dim=(1, 2))

    # frame_sums = a.view(tensor.shape[0], -1).sum(dim=1)
    a = a / a.sum()

    return a[start:end+1].sum()
