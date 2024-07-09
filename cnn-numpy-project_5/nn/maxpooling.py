import numpy as np
from nn.layer import AbstractLayer

class MaxPooling(AbstractLayer):
    def __init__(self, pshape, strides=1):
        self.pshape = pshape  # pooling shape (height * width)
        self.strides = strides
        self.cached_data = []

    ###########################################################################
    # TODO: Implement the Max-pooling layer                                   #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, inputs):
        batch_size, in_height, in_width, in_channels = inputs.shape
        p_height, p_width = self.pshape
        stride = self.strides

        out_height = int((in_height - p_height) / stride) + 1
        out_width = int((in_width - p_width) / stride) + 1

        pooled = np.zeros((batch_size, out_height, out_width, in_channels))

        for h in range(out_height):
            for w in range(out_width):
                h_start, h_end = h * stride, h * stride + p_height
                w_start, w_end = w * stride, w * stride + p_width
                region = inputs[:, h_start:h_end, w_start:w_end, :]
                pooled[:, h, w, :] = np.max(region, axis=(1, 2))

        # Cache data for backward pass
        self.cached_data = inputs, pooled

        return pooled


    def get_activation_grad(self, z, upstream_gradient):
        # There is no activation function
        return upstream_gradient

    def backward(self, layer_err):
        inputs, pooled = self.cached_data
        batch_size, in_height, in_width, in_channels = inputs.shape
        p_height, p_width = self.pshape
        stride = self.strides

        dx = np.zeros_like(inputs)

        for h in range(layer_err.shape[1]):
            for w in range(layer_err.shape[2]):
                h_start, h_end = h * stride, h * stride + p_height
                w_start, w_end = w * stride, w * stride + p_width
                region = inputs[:, h_start:h_end, w_start:w_end, :]

                # Get the index of the maximum value in the region
                max_indices = np.argmax(region, axis=(1, 2))

                # Convert the 1D indices to 2D indices
                max_indices = np.unravel_index(max_indices, region.shape[:2])

                # Use the indices to assign the upstream gradient to the max position
                dx[:, h_start:h_end, w_start:w_end, :][max_indices] = layer_err[:, h, w, :]

        return dx

    def get_grad(self, inputs, layer_err):
        dx = np.zeros_like(inputs)
        batch_size, in_height, in_width, in_channels = inputs.shape
        p_height, p_width = self.pshape
        stride = self.strides

        for h in range(layer_err.shape[1]):
            for w in range(layer_err.shape[2]):
                h_start, h_end = h * stride, h * stride + p_height
                w_start, w_end = w * stride, w * stride + p_width
                region = inputs[:, h_start:h_end, w_start:w_end, :]

                # Get the index of the maximum value in the region
                max_indices = np.argmax(region, axis=(1, 2))

                # Convert the 1D indices to 2D indices
                max_indices = np.unravel_index(max_indices, region.shape[:2])

                # Use the indices to assign the layer_err to the max position
                dx[:, h_start:h_end, w_start:w_end, :][max_indices] = layer_err[:, h, w, :]

        return dx

    def update(self, grad, lr):
        pass
        return None

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################