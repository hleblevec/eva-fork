import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """
    A self-attention layer designed for efficient processing of spatial data. This layer computes attention scores
    between spatial positions in the input tensor and applies these scores to the corresponding values.
    The implementation is compatible with Xilinx hardware accelerators.
    https://docs.amd.com/r/en-US/ug1414-vitis-ai/Operators-Supported-by-PyTorch

    Args:
        in_channels (int): The number of input channels (feature maps) for the self-attention layer.
    """

    def __init__(self, in_channels, sdnn_dense_params={}):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        
        # Define linear transformations for the Query, Key, and Value
        # These learnable layers project the input features into different subspaces for attention mechanism
        if sdnn_dense_params == {} : 
            self.query = nn.Linear(in_channels, in_channels)
            self.key = nn.Linear(in_channels, in_channels)
            self.value = nn.Linear(in_channels, in_channels)
        else:
            self.query = slayer.block.sigma_delta.Dense(sdnn_dense_params, in_channels, in_channels, weight_scale=2, weight_norm=True)
            self.key = slayer.block.sigma_delta.Dense(sdnn_dense_params, in_channels, in_channels, weight_scale=2, weight_norm=True)
            self.value = slayer.block.sigma_delta.Dense(sdnn_dense_params, in_channels, in_channels, weight_scale=2, weight_norm=True) 
        
    def forward(self, x):
        """
        Forward pass through the self-attention layer.

        Args:
            x (Tensor): Input tensor of shape [batch_size, channels, spatial_dim],
                        where `batch_size` is the number of samples in the batch,
                        `channels` is the number of channels in the input, and
                        `spatial_dim` refers to the spatial dimensions of the input (e.g., width x height).

        Returns:
            Tensor: The attended output tensor of shape [batch_size, channels, spatial_dim], 
                    where each spatial location has been weighted by its importance to other locations.
        """
        
        # Extract batch size, number of channels, and spatial dimensions
        batch_size, channels, spatial_dim = x.size() 
        
        # Apply linear transformations to compute query, key, and value
        query = self.query(x)  # [batch_size, channels, channels]
        key = self.key(x)      # [batch_size, channels, channels]
        value = self.value(x)  # [batch_size, channels, channels]
        
        # Compute the scaled dot-product attention scores
        scores = torch.matmul(query, key.transpose(1, 2))  # [batch_size, spatial_dim, spatial_dim]
        scores = scores / (channels ** 0.5)  # Scale by the square root of channels for numerical stability
        
        # Apply softmax to convert attention scores into weights
        attention_weights = torch.softmax(scores, dim=2)  # [batch_size, spatial_dim, spatial_dim]
        
        # Compute the weighted sum of the values based on the attention weights
        attended_values = torch.matmul(attention_weights, value)  # [batch_size, spatial_dim, channels]
        
        # Reshape the output to match the input shape
        attended_values = attended_values.view(batch_size, channels, spatial_dim)
        
        return attended_values
    
if __name__ == '__main__': 
    # Example usage of the SelfAttention module
    batch_size = 1
    in_channels = 3  # Number of input channels (e.g., RGB channels)
    out_channels = 6  # Number of output channels (this will be used in the extended model)
    width, height = 640, 480  # Spatial dimensions (width x height)
    
    # Create a random input tensor with shape [batch_size, in_channels, width, height]
    x = torch.randn(batch_size, in_channels, width, height)
    
    # Initialize the SelfAttention module
    self_attention = SelfAttention(in_channels)
    
    # Forward pass through the attention layer
    output = self_attention(x)
    
    # Print the output shape to verify the transformation
    print(output.shape)  # Should be [batch_size, channels, spatial_dim]