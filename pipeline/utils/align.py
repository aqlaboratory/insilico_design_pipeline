import torch


def compute_rigid_alignment(A, B):
    """
    Use Kabsch algorithm to compute alignment from point cloud A to point cloud B.

    Source: https://gist.github.com/bougui505/e392a371f5bab095a3673ea6f4976cc8
    See: https://en.wikipedia.org/wiki/Kabsch_algorithm
    
    Args:
        A:
            [N, D] Point Cloud to Align (source)
        B:
            [N, D] Reference Point Cloud (target)
    
    Returns:
        R:
            Optimal rotation
        t: 
            Optimal translation
    """

    # Center
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A_c = A - a_mean
    B_c = B - b_mean

    # Covariance matrix
    H = A_c.T.mm(B_c)
    U, S, V = torch.svd(H)

    # Rotation matrix
    R = V.mm(U.T)

    # Translation vector
    t = b_mean[None, :] - R.mm(a_mean[None, :].T).T
    t = t.T
    
    return R, t.squeeze()