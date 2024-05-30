import torch
import torch.nn.functional as F


####################
###   Geometry   ###
####################

def distance(x, y, eps=1e-10):
    # x: [B, P, 3]
    # y: [B, P, 3]
    return torch.sqrt(eps + torch.sum((x - y) ** 2, dim=-1))

def angle(x, y, z):
    # x: [B, P, 3]
    # y: [B, P, 3]
    # z: [B, P, 3]
    
    # [B, P, 3]
    v1 = x - y
    v2 = z - y
    
    # [B, P]
    v1v2 = torch.einsum('bij,bij->bi', v1, v2)
    
    # [B, P]
    v1_norm = torch.norm(v1, dim=-1)
    v2_norm = torch.norm(v2, dim=-1)
    
    # [B, P]
    rad = torch.acos(v1v2 / (v1_norm * v2_norm))

    return torch.rad2deg(rad)

def dihedral(w, x, y, z):
    # w, x, y, z: [B, P, 3]
    # Reference: https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
    
    # [B, P, 3]
    b0 = w - x
    b1 = y - x
    b2 = z - y

    # [B, P, 3]
    b1 = b1 / torch.norm(b1, dim=-1, keepdim=True)
    
    # [B, P, 3]
    v = b0 - torch.einsum('bij,bij->bi', b0, b1).unsqueeze(-1) * b1
    w = b2 - torch.einsum('bij,bij->bi', b2, b1).unsqueeze(-1) * b1
    
    # [B, P]
    x = torch.einsum('bij,bij->bi', v, w)
    y = torch.einsum('bij,bij->bi', torch.cross(b1, v, dim=-1), w)
    
    # [B, P]
    rad = torch.atan2(y, x)
    
    return torch.rad2deg(rad)

################################
###   Secondary Structures   ###
################################

HELIX_CONSTRAINTS = {
    'a': (89, 12),    # angle of Ca triplet (i - 1, i, i + 1)
    'd': (50, 20),    # dihedral angle of Ca quadruplet (i - 1, i, i + 1, i + 2)
    'd2': (5.5, 0.5), # distance between (i - 1)th residue and the (i + 1)th residue
    'd3': (5.3, 0.5), # distance between (i - 1)th residue and the (i + 2)th residue
    'd4': (6.4, 0.6)  # distance between (i - 1)th residue and the (i + 3)th residue
}

STRAND_CONSTRAINTS = {
    'a': (124, 14),   # angle of Ca triplet (i - 1, i, i + 1)
    'd': (-170, 45),  # dihedral angle of Ca quadruplet (i - 1, i, i + 1, i + 2)
    'd2': (6.7, 0.6), # distance between (i - 1)th residue and the (i + 1)th residue
    'd3': (9.9, 0.9), # distance between (i - 1)th residue and the (i + 2)th residue
    'd4': (12.4, 1.1) # distance between (i - 1)th residue and the (i + 3)th residue
}

HELIX_SIZE = 5  # minimum number of residues for a helix
STRAND_SIZE = 4 # minimum number of residues for a strand

LEFT_HELIX_SIZE = 4
LEFT_HELIX_DIHEDRAL_MIN = -70
LEFT_HELIX_DIHEDRAL_MAX = -30


def cond_to_pred(cond, size):
    # P' = P/3 - 4

    # [B, P' - S + 1, S]
    cond_unfold = cond.unfold(1, size=size, step=1)
    
    # [B, P' - S + 1]
    r1 = torch.sum(cond_unfold, dim=2) == size

    # [B, P' + S - 1]
    r1 = F.pad(r1, (size - 1, size - 1), 'constant', False)
    
    # [B, P', S]
    r1_unfold = r1.unfold(1, size=size, step=1)
    
    # [B, P']
    r2 = torch.sum(r1_unfold, dim=2) > 0
    
    return r2

def assign_secondary_structures(coords, return_encodings=True, full=True):
    # Followed from P-SEA implementation
    # Reference: https://academic.oup.com/bioinformatics/article/13/3/291/423201
    # frames: [B, P]

    def decode(one_hot_ss):
        ss = []
        for sample_idx in range(one_hot_ss.shape[0]):
            sample_ss = ''
            for residue_idx in range(one_hot_ss.shape[1]):
                if one_hot_ss[sample_idx, residue_idx, 0]:
                    sample_ss += 'h'
                elif one_hot_ss[sample_idx, residue_idx, 1]:
                    sample_ss += 's'
                else:
                    sample_ss += '-'
            ss.append(sample_ss)
        return ss

    # [B, P/3, 3]
    x = coords[:, 1::3, :] if full else coords

    # [B, P/3 - 4, 3]
    x0 = x[:, 0:-4:, :]
    x1 = x[:, 1:-3:, :]
    x2 = x[:, 2:-2:, :]
    x3 = x[:, 3:-1:, :]
    x4 = x[:, 4::, :]

    # [B, P/3 - 4] for each value
    values = {
        'a': angle(x0, x1, x2),
        'd': dihedral(x0, x1, x2, x3),
        'd2': distance(x2, x0),
        'd3': distance(x3, x0),
        'd4': distance(x4, x0)
    }

    # [B, P/3 - 4] for each condition
    h_conds = dict([
        (
            key,
            torch.logical_and(
                values[key] >= HELIX_CONSTRAINTS[key][0] - HELIX_CONSTRAINTS[key][1],
                values[key] <= HELIX_CONSTRAINTS[key][0] + HELIX_CONSTRAINTS[key][1]
            )
        )
        for key in values
    ])

    # [B, P/3 - 4]
    cond_helix = torch.logical_or(
        torch.logical_and(h_conds['d3'], h_conds['d4']),
        torch.logical_and(h_conds['a'], h_conds['d'])
    )

    # [B, P/3 - 4] for each condition
    s_conds = dict([
        (
            key,
            torch.logical_and(
                values[key] >= STRAND_CONSTRAINTS[key][0] - STRAND_CONSTRAINTS[key][1],
                values[key] <= STRAND_CONSTRAINTS[key][0] + STRAND_CONSTRAINTS[key][1]
            )
        )
        for key in values
    ])

    # [B, P/3 - 4]
    cond_strand = torch.logical_or(
        torch.logical_and(torch.logical_and(s_conds['d2'], s_conds['d3']), s_conds['d4']),
        torch.logical_and(s_conds['a'], s_conds['d'])
    )

    # [B, P/3]
    is_helix = F.pad(cond_to_pred(cond_helix, HELIX_SIZE), (1, 3), 'constant', False)
    is_strand = F.pad(cond_to_pred(cond_strand, STRAND_SIZE), (1, 3), 'constant', False)
    is_strand = torch.logical_and(is_strand, ~is_helix)
    
    # [B, P/3, 2]
    one_hot_ss = torch.stack([is_helix, is_strand], dim=2)

    return one_hot_ss if return_encodings else decode(one_hot_ss)

def assign_left_handed_helices(coords):
    # coords: [B, P, 3]

    # [B, P - 3, 3]
    x0 = coords[:, :-3]
    x1 = coords[:, 1:-2]
    x2 = coords[:, 2:-1]
    x3 = coords[:, 3:]

    # [B, P - 3]
    d = dihedral(x0, x1, x2, x3)
    cond = torch.logical_and(d >= LEFT_HELIX_DIHEDRAL_MIN, d <= LEFT_HELIX_DIHEDRAL_MAX)

    # [B, P]
    is_left_helix = F.pad(cond_to_pred(cond, LEFT_HELIX_SIZE), (1, 2), 'constant', False)
    assert is_left_helix.shape[1] == coords.shape[1]

    return is_left_helix