import torch

# A 3x2 matrix of raw counts
counts = torch.tensor([[10.0, 20.0],
                       [ 5.0,  5.0],
                       [ 1.0,  9.0]])

# WRONG WAY: keepdim=False (default)
# Resulting shape is [3]. 
# PyTorch can't easily tell if this 3 belongs to rows or columns.
row_sums_wrong = counts.sum(dim=1, keepdim=False) 

# RIGHT WAY: keepdim=True
# Resulting shape is [3, 1]. 
# This is a column vector that aligns perfectly with our rows.
row_sums_right = counts.sum(dim=1, keepdim=True) 
print(f"Sum (keepdim = False) Shape:{row_sums_wrong.shape}\n{row_sums_wrong}")

# The Division (Broadcasting)
# This works because PyTorch stretches the [3, 1] column to [3, 2]
probs = counts / row_sums_right

print(f"Counts Shape: {counts.shape}")
print(f"Sum (keepdim=True) Shape: {row_sums_right.shape}")
print(f"\nProbabilities (Each row now sums to 1.0):\n{probs}")
