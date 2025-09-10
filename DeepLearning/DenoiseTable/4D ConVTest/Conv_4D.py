import torch
import torch.nn as nn
from torch.nn.modules.utils import _quadruple
import math
import torch.nn.functional as F

class Conv4d(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 kernel_size:[int, tuple],
                 stride:[int, tuple] = (1, 1, 1, 1),
                 padding:[int, tuple] = (0, 0, 0, 0),
                 dilation:[int, tuple] = (1, 1, 1, 1),
                 groups:int = 1,
                 bias=False,
                 padding_mode:str ='zeros'):
        super(Conv4d, self).__init__()
        kernel_size = _quadruple(kernel_size)
        stride = _quadruple(stride)
        padding = _quadruple(padding)
        dilation = _quadruple(dilation)

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))

        # Assertions for constructor arguments
        assert len(kernel_size) == 4, '4D kernel size expected!'
        assert len(stride) == 4, '4D Stride size expected!!'
        assert len(padding) == 4, '4D Padding size expected!!'
        assert len(dilation) == 4, '4D dilation size expected!'
        assert groups == 1, 'Groups other than 1 not yet implemented!'

        # Store constructor arguments
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.groups = groups
        self.padding_mode = padding_mode

        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        # # # # # self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 3)

        # Construct weight and bias of 4D convolution
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
        self.reset_parameters()

        # Use a ModuleList to store layers to make the Conv4d layer trainable
        self.conv3d_layers = torch.nn.ModuleList()

        for i in range(self.kernel_size[0]):
            # Initialize a Conv3D layer
            conv3d_layer = nn.Conv3d(in_channels=self.in_channels,
                                     out_channels=self.out_channels,
                                     kernel_size=self.kernel_size[1::],
                                     padding=self.padding[1::],
                                     dilation=self.dilation[1::],
                                     stride=self.stride[1::],
                                     bias=False)
            conv3d_layer.weight = nn.Parameter(self.weight[:, :, i, :, :])

            # Store the layer
            self.conv3d_layers.append(conv3d_layer)

        del self.weight


    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)


    def forward(self, input):
        # Define shortcut names for dimensions of input and kernel
        (Batch, _, l_i, d_i, h_i, w_i) = tuple(input.shape)
        (l_k, d_k, h_k, w_k) = self.kernel_size
        (l_p, d_p, h_p, w_p) = self.padding
        (l_d, d_d, h_d, w_d) = self.dilation
        (l_s, d_s, h_s, w_s) = self.stride

        # Compute the size of the output tensor based on the zero padding
        l_o = (l_i + 2 * l_p - (l_k) - (l_k-1) * (l_d-1))//l_s + 1
        d_o = (d_i + 2 * d_p - (d_k) - (d_k-1) * (d_d-1))//d_s + 1
        h_o = (h_i + 2 * h_p - (h_k) - (h_k-1) * (h_d-1))//h_s + 1
        w_o = (w_i + 2 * w_p - (w_k) - (w_k-1) * (w_d-1))//w_s + 1

        # Pre-define output tensors
        out = torch.zeros(Batch, self.out_channels, l_o, d_o, h_o, w_o).to(input.device)

        # Convolve each kernel frame i with each input frame j
        for i in range(l_k):
            # Calculate the zero-offset of kernel frame i
            zero_offset = - l_p + (i * l_d)
            # Calculate the range of input frame j corresponding to kernel frame i
            j_start = max(zero_offset % l_s, zero_offset)
            j_end = min(l_i, l_i + l_p - (l_k-i-1)*l_d)
            # Convolve each kernel frame i with corresponding input frame j
            for j in range(j_start, j_end, l_s):
                # Calculate the output frame
                out_frame = (j - zero_offset) // l_s
                # Add results to this output frame
                out[:, :, out_frame, :, :, :] += self.conv3d_layers[i](input[:, :, j, :, :])

        # Add bias to output
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1, 1, 1)

        return out
    
# print("--- Testing Conv4d class ---")

# # 1. Basic Test Case
# print("\n--- Basic Test: No Padding, No Dilation, Stride 1 ---")
# in_channels = 2
# out_channels = 4
# kernel_size = (2, 3, 3, 3) # (L, D, H, W)
# input_size = (1, in_channels, 5, 10, 10, 10) # (Batch, C, L, D, H, W)

# conv4d_layer = Conv4d(in_channels, out_channels, kernel_size)
# dummy_input = torch.randn(input_size)

# print(f"Input shape: {dummy_input.shape}")
# output_basic = conv4d_layer(dummy_input)
# print(f"Output shape: {output_basic.shape}")

# # Expected output shape calculation for basic case:
# # L_out = (5 - 2)/1 + 1 = 4
# # D_out = (10 - 3)/1 + 1 = 8
# # H_out = (10 - 3)/1 + 1 = 8
# # W_out = (10 - 3)/1 + 1 = 8
# expected_l = (input_size[2] - kernel_size[0]) // 1 + 1
# expected_d = (input_size[3] - kernel_size[1]) // 1 + 1
# expected_h = (input_size[4] - kernel_size[2]) // 1 + 1
# expected_w = (input_size[5] - kernel_size[3]) // 1 + 1
# expected_shape_basic = (input_size[0], out_channels, expected_l, expected_d, expected_h, expected_w)
# print(f"Expected output shape: {expected_shape_basic}")
# assert output_basic.shape == expected_shape_basic, f"Basic test failed! Expected {expected_shape_basic}, got {output_basic.shape}"
# print("Basic test passed!")

# # 2. Test Case with Padding
# print("\n--- Test with Padding ---")
# in_channels_p = 2
# out_channels_p = 4
# kernel_size_p = (2, 3, 3, 3)
# padding_p = (1, 1, 1, 1) # Add padding to all dimensions
# input_size_p = (1, in_channels_p, 5, 10, 10, 10)

# conv4d_layer_p = Conv4d(in_channels_p, out_channels_p, kernel_size_p, padding=padding_p)
# dummy_input_p = torch.randn(input_size_p)

# print(f"Input shape with padding: {dummy_input_p.shape}")
# output_padded = conv4d_layer_p(dummy_input_p)
# print(f"Output shape with padding: {output_padded.shape}")

# # Expected output shape calculation for padded case:
# # L_out = (5 + 2*1 - 2)/1 + 1 = 6
# # D_out = (10 + 2*1 - 3)/1 + 1 = 10
# # H_out = (10 + 2*1 - 3)/1 + 1 = 10
# # W_out = (10 + 2*1 - 3)/1 + 1 = 10
# expected_l_p = (input_size_p[2] + 2 * padding_p[0] - kernel_size_p[0]) // 1 + 1
# expected_d_p = (input_size_p[3] + 2 * padding_p[1] - kernel_size_p[1]) // 1 + 1
# expected_h_p = (input_size_p[4] + 2 * padding_p[2] - kernel_size_p[2]) // 1 + 1
# expected_w_p = (input_size_p[5] + 2 * padding_p[3] - kernel_size_p[3]) // 1 + 1
# expected_shape_padded = (input_size_p[0], out_channels_p, expected_l_p, expected_d_p, expected_h_p, expected_w_p)
# print(f"Expected output shape with padding: {expected_shape_padded}")
# assert output_padded.shape == expected_shape_padded, f"Padding test failed! Expected {expected_shape_padded}, got {output_padded.shape}"
# print("Padding test passed!")


# # 3. Test Case with Stride
# print("\n--- Test with Stride ---")
# in_channels_s = 1
# out_channels_s = 1
# kernel_size_s = (2, 2, 2, 2)
# stride_s = (2, 2, 2, 2)
# input_size_s = (1, in_channels_s, 6, 6, 6, 6) # Ensure dimensions are divisible by stride for cleaner calculation

# conv4d_layer_s = Conv4d(in_channels_s, out_channels_s, kernel_size_s, stride=stride_s)
# dummy_input_s = torch.randn(input_size_s)

# print(f"Input shape with stride: {dummy_input_s.shape}")
# output_strided = conv4d_layer_s(dummy_input_s)
# print(f"Output shape with stride: {output_strided.shape}")

# # Expected output shape calculation for strided case:
# # L_out = (6 - 2)/2 + 1 = 3
# # D_out = (6 - 2)/2 + 1 = 3
# # H_out = (6 - 2)/2 + 1 = 3
# # W_out = (6 - 2)/2 + 1 = 3
# expected_l_s = (input_size_s[2] - kernel_size_s[0]) // stride_s[0] + 1
# expected_d_s = (input_size_s[3] - kernel_size_s[1]) // stride_s[1] + 1
# expected_h_s = (input_size_s[4] - kernel_size_s[2]) // stride_s[2] + 1
# expected_w_s = (input_size_s[5] - kernel_size_s[3]) // stride_s[3] + 1
# expected_shape_strided = (input_size_s[0], out_channels_s, expected_l_s, expected_d_s, expected_h_s, expected_w_s)
# print(f"Expected output shape with stride: {expected_shape_strided}")
# assert output_strided.shape == expected_shape_strided, f"Stride test failed! Expected {expected_shape_strided}, got {output_strided.shape}"
# print("Stride test passed!")

# # 4. Test Case with Dilation
# print("\n--- Test with Dilation ---")
# in_channels_d = 1
# out_channels_d = 1
# kernel_size_d = (2, 2, 2, 2)
# dilation_d = (2, 2, 2, 2)
# input_size_d = (1, in_channels_d, 5, 5, 5, 5)

# conv4d_layer_d = Conv4d(in_channels_d, out_channels_d, kernel_size_d, dilation=dilation_d)
# dummy_input_d = torch.randn(input_size_d)

# print(f"Input shape with dilation: {dummy_input_d.shape}")
# output_dilated = conv4d_layer_d(dummy_input_d)
# print(f"Output shape with dilation: {output_dilated.shape}")

# # Expected output shape calculation for dilated case:
# # L_out = (L_in + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1
# # L_out = (5 + 2*0 - 2*(2-1) - 1)/1 + 1 = (5 - 2 - 1)/1 + 1 = 2+1 = 3  (Corrected from 2)
# # The formula (L_in + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1
# # For L: (5 + 2*0 - 2*(2-1) - 1)/1 + 1 = (5 - 2 - 1)/1 + 1 = 3
# expected_l_d = math.floor((input_size_d[2] + 2 * 0 - dilation_d[0] * (kernel_size_d[0] - 1) - 1) / 1) + 1
# expected_d_d = math.floor((input_size_d[3] + 2 * 0 - dilation_d[1] * (kernel_size_d[1] - 1) - 1) / 1) + 1
# expected_h_d = math.floor((input_size_d[4] + 2 * 0 - dilation_d[2] * (kernel_size_d[2] - 1) - 1) / 1) + 1
# expected_w_d = math.floor((input_size_d[5] + 2 * 0 - dilation_d[3] * (kernel_size_d[3] - 1) - 1) / 1) + 1
# expected_shape_dilated = (input_size_d[0], out_channels_d, expected_l_d, expected_d_d, expected_h_d, expected_w_d)
# print(f"Expected output shape with dilation: {expected_shape_dilated}")
# assert output_dilated.shape == expected_shape_dilated, f"Dilation test failed! Expected {expected_shape_dilated}, got {output_dilated.shape}"
# print("Dilation test passed!")

# # 5. Test Case with Bias
# print("\n--- Test with Bias ---")
# in_channels_b = 1
# out_channels_b = 1
# kernel_size_b = (1, 1, 1, 1)
# input_size_b = (1, in_channels_b, 3, 3, 3, 3)

# conv4d_layer_b = Conv4d(in_channels_b, out_channels_b, kernel_size_b, bias=True)
# dummy_input_b = torch.randn(input_size_b)

# print(f"Input shape with bias: {dummy_input_b.shape}")
# output_bias = conv4d_layer_b(dummy_input_b)
# print(f"Output shape with bias: {output_bias.shape}")
# print(f"Bias parameter: {conv4d_layer_b.bias}")
# assert conv4d_layer_b.bias is not None, "Bias parameter not created when bias=True"
# print("Bias test passed!")

# # 6. Test with varying input and kernel sizes and combinations of parameters
# print("\n--- Comprehensive Test with Mixed Parameters ---")
# in_channels_comp = 3
# out_channels_comp = 8
# kernel_size_comp = (3, 3, 5, 5)
# stride_comp = (1, 2, 1, 2)
# padding_comp = (0, 1, 0, 1)
# dilation_comp = (1, 1, 2, 1)
# input_size_comp = (2, in_channels_comp, 10, 15, 20, 25)

# conv4d_layer_comp = Conv4d(in_channels_comp, out_channels_comp, kernel_size_comp,
#                            stride=stride_comp, padding=padding_comp, dilation=dilation_comp, bias=True)
# dummy_input_comp = torch.randn(input_size_comp)

# print(f"Input shape (Comprehensive): {dummy_input_comp.shape}")
# output_comp = conv4d_layer_comp(dummy_input_comp)
# print(f"Output shape (Comprehensive): {output_comp.shape}")

# # Expected output shape calculation for comprehensive case:
# # L_out = floor((10 + 2*0 - 1*(3-1) - 1)/1) + 1 = floor((10 - 2 - 1)/1) + 1 = 7 + 1 = 8
# # D_out = floor((15 + 2*1 - 1*(3-1) - 1)/2) + 1 = floor((15 + 2 - 2 - 1)/2) + 1 = floor(14/2) + 1 = 7 + 1 = 8
# # H_out = floor((20 + 2*0 - 2*(5-1) - 1)/1) + 1 = floor((20 - 8 - 1)/1) + 1 = 11 + 1 = 12
# # W_out = floor((25 + 2*1 - 1*(5-1) - 1)/2) + 1 = floor((25 + 2 - 4 - 1)/2) + 1 = floor(22/2) + 1 = 11 + 1 = 12

# expected_l_comp = math.floor((input_size_comp[2] + 2 * padding_comp[0] - dilation_comp[0] * (kernel_size_comp[0] - 1) - 1) / stride_comp[0]) + 1
# expected_d_comp = math.floor((input_size_comp[3] + 2 * padding_comp[1] - dilation_comp[1] * (kernel_size_comp[1] - 1) - 1) / stride_comp[1]) + 1
# expected_h_comp = math.floor((input_size_comp[4] + 2 * padding_comp[2] - dilation_comp[2] * (kernel_size_comp[2] - 1) - 1) / stride_comp[2]) + 1
# expected_w_comp = math.floor((input_size_comp[5] + 2 * padding_comp[3] - dilation_comp[3] * (kernel_size_comp[3] - 1) - 1) / stride_comp[3]) + 1
# expected_shape_comp = (input_size_comp[0], out_channels_comp, expected_l_comp, expected_d_comp, expected_h_comp, expected_w_comp)
# print(f"Expected output shape (Comprehensive): {expected_shape_comp}")
# assert output_comp.shape == expected_shape_comp, f"Comprehensive test failed! Expected {expected_shape_comp}, got {output_comp.shape}"
# print("Comprehensive test passed!")

# print("\n--- All tests completed. If no assertions failed, your Conv4d seems to be working as expected regarding output shape and basic functionality. ---")
