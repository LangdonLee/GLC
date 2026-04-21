# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
import torch
from torch import nn

from .entropy_models import EntropyCoder, GaussianEncoder


class CompressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.masks = {}
        self.force_generate_mask = False

        self.entropy_coder = None
        self.gaussian_encoder = GaussianEncoder()

        self.cuda_streams = {}

    def get_cuda_stream(self, device, idx=0, priority=0):
        key = f"{device}_{priority}_{idx}"
        if key not in self.cuda_streams:
            self.cuda_streams[key] = torch.cuda.Stream(device, priority=priority)
        return self.cuda_streams[key]

    def update(self, force_zero_thres=None):
        self.entropy_coder = EntropyCoder()
        self.gaussian_encoder.update(self.entropy_coder, force_zero_thres=force_zero_thres)

    def set_use_two_entropy_coders(self, use_two_entropy_coders):
        self.entropy_coder.set_use_two_entropy_coders(use_two_entropy_coders)

    def quant(self, x):
        n = torch.round(x) - x
        n = n.clone().detach()
        return x + n

    @staticmethod
    def get_padding_size(height, width, p=64):
        new_h = (height + p - 1) // p * p
        new_w = (width + p - 1) // p * p
        # padding_left = (new_w - width) // 2
        padding_left = 0
        padding_right = new_w - width - padding_left
        # padding_top = (new_h - height) // 2
        padding_top = 0
        padding_bottom = new_h - height - padding_top
        return padding_left, padding_right, padding_top, padding_bottom

    def probs_to_bits(self, probs):
        bits = -1.0 * torch.log(probs + 1e-5) / math.log(2.0)
        return bits.clamp_min(0)

    def get_y_gaussian_bits(self, y, sigma):
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.normal.Normal(mu, sigma)
        probs = gaussian.cdf(y + 0.5) - gaussian.cdf(y - 0.5)
        return self.probs_to_bits(probs)

    def pad_for_y(self, y):
        _, _, H, W = y.size()
        padding_l, padding_r, padding_t, padding_b = self.get_padding_size(H, W, 4)
        y_pad = torch.nn.functional.pad(
            y,
            (padding_l, padding_r, padding_t, padding_b),
            mode="replicate",
        )
        return y_pad, (-padding_l, -padding_r, -padding_t, -padding_b)

    def slice_to_y(self, param, slice_shape):
        return torch.nn.functional.pad(param, slice_shape)

    def separate_prior(self, params):
        quant_step, scales, means = params.chunk(3, 1)
        quant_step = quant_step.clamp_min(0.5)
        q_enc = 1. / quant_step
        q_dec = quant_step
        return q_enc, q_dec, scales, means

    @staticmethod
    def single_part_for_writing_2x(x):
        x0, x1 = x.chunk(2, 1)
        return x0 + x1

    @staticmethod
    def single_part_for_writing_4x(x):
        x0, x1, x2, x3 = x.chunk(4, 1)
        return (x0 + x1) + (x2 + x3)

    def get_mask(self, height, width, dtype, device):
        curr_mask_str = f"{width}x{height}"
        if curr_mask_str not in self.masks or self.force_generate_mask:
            micro_mask = torch.tensor(((1, 0), (0, 1)), dtype=dtype, device=device)
            mask_0 = micro_mask.repeat((height + 1) // 2, (width + 1) // 2)
            mask_0 = mask_0[:height, :width]
            mask_0 = torch.unsqueeze(mask_0, 0)
            mask_0 = torch.unsqueeze(mask_0, 0)
            mask_1 = torch.ones_like(mask_0) - mask_0
            self.masks[curr_mask_str] = [mask_0, mask_1]
        return self.masks[curr_mask_str]

    def process_with_mask(self, y, scales, means, mask):
        scales_hat = scales * mask
        means_hat = means * mask

        y_res = (y - means_hat) * mask
        y_q = self.quant(y_res)
        y_hat = y_q + means_hat

        return y_res, y_q, y_hat, scales_hat

    @staticmethod
    def get_one_channel_dual_mask(height, width, dtype, device):
        micro_mask_0 = torch.tensor(((1, 0), (0, 1)), dtype=dtype, device=device)
        mask_0 = micro_mask_0.repeat((height + 1) // 2, (width + 1) // 2)
        mask_0 = mask_0[:height, :width]
        mask_0 = torch.unsqueeze(mask_0, 0)
        mask_0 = torch.unsqueeze(mask_0, 0)

        micro_mask_1 = torch.tensor(((0, 1), (1, 0)), dtype=dtype, device=device)
        mask_1 = micro_mask_1.repeat((height + 1) // 2, (width + 1) // 2)
        mask_1 = mask_1[:height, :width]
        mask_1 = torch.unsqueeze(mask_1, 0)
        mask_1 = torch.unsqueeze(mask_1, 0)

        return mask_0, mask_1

    def get_mask_dual(self, batch, channel, height, width, dtype, device):
        curr_mask_str = f"{batch}_{channel}x{width}x{height}"
        with torch.no_grad():
            if curr_mask_str not in self.masks or self.force_generate_mask:
                assert channel % 2 == 0
                m = torch.ones((batch, channel // 2, height, width), dtype=dtype, device=device)
                m0, m1 = self.get_one_channel_dual_mask(height, width, dtype, device)

                mask_0 = torch.cat((m * m0, m * m1), dim=1)
                mask_1 = torch.cat((m * m1, m * m0), dim=1)

                self.masks[curr_mask_str] = [mask_0, mask_1]
        return self.masks[curr_mask_str]

    def forward_dual_prior(self, y, common_params, y_spatial_prior):
        q_enc, q_dec, scales, means = self.separate_prior(common_params)
        dtype = y.dtype
        device = y.device
        B, C, H, W = y.size()
        mask_0, mask_1 = self.get_mask_dual(B, C, H, W, dtype, device)

        y = y * q_enc
        y_res_0, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask(y, scales, means, mask_0)
        scales, means = y_spatial_prior(torch.cat((y_hat_0, common_params), dim=1)).chunk(2, 1)
        y_res_1, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y, scales, means, mask_1)

        y_hat = y_hat_0 + y_hat_1
        y_hat = y_hat * q_dec

        y_res = y_res_0 + y_res_1
        y_q = y_q_0 + y_q_1
        scales_hat = s_hat_0 + s_hat_1
        return y_res, y_q, y_hat, scales_hat

    def compress_dual_prior(self, y, common_params, y_spatial_prior):
        q_enc, q_dec, scales, means = self.separate_prior(common_params)
        dtype = y.dtype
        device = y.device
        B, C, H, W = y.size()
        mask_0, mask_1 = self.get_mask_dual(B, C, H, W, dtype, device)

        y = y * q_enc
        y_res_0, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask(y, scales, means, mask_0)
        scales, means = y_spatial_prior(torch.cat((y_hat_0, common_params), dim=1)).chunk(2, 1)
        y_res_1, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y, scales, means, mask_1)

        y_q_w_0 = self.single_part_for_writing_2x(y_q_0)
        y_q_w_1 = self.single_part_for_writing_2x(y_q_1)
        s_w_0 = self.single_part_for_writing_2x(s_hat_0)
        s_w_1 = self.single_part_for_writing_2x(s_hat_1)

        y_hat = y_hat_0 + y_hat_1
        y_hat = y_hat * q_dec

        # y_res = y_res_0 + y_res_1
        # y_q = y_q_0 + y_q_1
        # scales_hat = s_hat_0 + s_hat_1
        return y_q_w_0, y_q_w_1, s_w_0, s_w_1, y_hat

    def decompress_dual_prior(self, common_params, y_spatial_prior):
        infos = self.decompress_prior_2x_part1(common_params)
        y_hat = self.decompress_prior_2x_part2(common_params, y_spatial_prior, infos)
        return y_hat

    def decompress_prior_2x_part1(self, common_params):
        q_enc, q_dec, scales, means = self.separate_prior(common_params)
        dtype = means.dtype
        device = means.device
        B, C, H, W = means.size()
        mask_0, mask_1 = self.get_mask_dual(B, C, H, W, dtype, device)

        scales_r = self.single_part_for_writing_2x(scales * mask_0)
        indexes, skip_cond = self.gaussian_encoder.build_indexes_decoder(scales_r)
        self.gaussian_encoder.decode_y(indexes)
        infos = {
            "q_dec": q_dec,
            "mask_0": mask_0,
            "mask_1": mask_1,
            "means": means,
            "scales_r": scales_r,
            "skip_cond": skip_cond,
            "indexes": indexes,
        }
        return infos

    def decompress_prior_2x_part2(self, common_params, y_spatial_prior, infos):
        def restore_y_2x(y, means, mask):
            return (torch.cat((y, y), dim=1) + means) * mask
        dtype = common_params.dtype
        device = common_params.device
        y_q_r = self.gaussian_encoder.get_y(infos["scales_r"].shape,
                                            infos["scales_r"].numel(),
                                            dtype, device,
                                            infos["skip_cond"], infos["indexes"])
        y_hat_0 = (torch.cat((y_q_r, y_q_r), dim=1) + infos['means']) * infos['mask_0']
        cat_params = torch.cat((y_hat_0, common_params), dim=1)
        scales, means = y_spatial_prior(cat_params).chunk(2, 1)
        scales_r = self.single_part_for_writing_2x(scales * infos["mask_1"])
        y_q_r = self.gaussian_encoder.decode_and_get_y(scales_r, dtype, device)
        y_hat_1 = restore_y_2x(y_q_r, means, infos["mask_1"])

        y_hat = y_hat_0 + y_hat_1
        y_hat = y_hat * infos['q_dec']
        return y_hat

    @staticmethod
    def get_one_channel_four_parts_mask(height, width, dtype, device):
        micro_mask_0 = torch.tensor(((1, 0), (0, 0)), dtype=dtype, device=device)
        mask_0 = micro_mask_0.repeat((height + 1) // 2, (width + 1) // 2)
        mask_0 = mask_0[:height, :width]
        mask_0 = torch.unsqueeze(mask_0, 0)
        mask_0 = torch.unsqueeze(mask_0, 0)

        micro_mask_1 = torch.tensor(((0, 1), (0, 0)), dtype=dtype, device=device)
        mask_1 = micro_mask_1.repeat((height + 1) // 2, (width + 1) // 2)
        mask_1 = mask_1[:height, :width]
        mask_1 = torch.unsqueeze(mask_1, 0)
        mask_1 = torch.unsqueeze(mask_1, 0)

        micro_mask_2 = torch.tensor(((0, 0), (1, 0)), dtype=dtype, device=device)
        mask_2 = micro_mask_2.repeat((height + 1) // 2, (width + 1) // 2)
        mask_2 = mask_2[:height, :width]
        mask_2 = torch.unsqueeze(mask_2, 0)
        mask_2 = torch.unsqueeze(mask_2, 0)

        micro_mask_3 = torch.tensor(((0, 0), (0, 1)), dtype=dtype, device=device)
        mask_3 = micro_mask_3.repeat((height + 1) // 2, (width + 1) // 2)
        mask_3 = mask_3[:height, :width]
        mask_3 = torch.unsqueeze(mask_3, 0)
        mask_3 = torch.unsqueeze(mask_3, 0)

        return mask_0, mask_1, mask_2, mask_3

    def get_mask_four_parts(self, batch, channel, height, width, dtype, device):
        curr_mask_str = f"{batch}_{channel}x{width}x{height}"
        with torch.no_grad():
            if curr_mask_str not in self.masks or self.force_generate_mask:
                assert channel % 4 == 0
                m = torch.ones((batch, channel // 4, height, width), dtype=dtype, device=device)
                m0, m1, m2, m3 = self.get_one_channel_four_parts_mask(height, width, dtype, device)

                mask_0 = torch.cat((m * m0, m * m1, m * m2, m * m3), dim=1)
                mask_1 = torch.cat((m * m3, m * m2, m * m1, m * m0), dim=1)
                mask_2 = torch.cat((m * m2, m * m3, m * m0, m * m1), dim=1)
                mask_3 = torch.cat((m * m1, m * m0, m * m3, m * m2), dim=1)

                self.masks[curr_mask_str] = [mask_0, mask_1, mask_2, mask_3]
        return self.masks[curr_mask_str]

    def forward_four_part_prior(self, y, common_params,
                                y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                y_spatial_prior_adaptor_3, y_spatial_prior,
                                y_spatial_prior_reduction=None, write=False):
        '''
        y_0 means split in channel, the 0/4 quater
        y_1 means split in channel, the 1/4 quater
        y_2 means split in channel, the 2/4 quater
        y_3 means split in channel, the 3/4 quater
        y_?_0, means multiply with mask_0
        y_?_1, means multiply with mask_1
        y_?_2, means multiply with mask_2
        y_?_3, means multiply with mask_3
        '''
        q_enc, q_dec, scales, means = self.separate_prior(common_params)
        if y_spatial_prior_reduction is not None:
            common_params = y_spatial_prior_reduction(common_params)
        dtype = y.dtype
        device = y.device
        B, C, H, W = y.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(B, C, H, W, dtype, device)

        y = y * q_enc

        y_res_0, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask(y, scales, means, mask_0)

        y_hat_so_far = y_hat_0
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(2, 1)
        y_res_1, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y, scales, means, mask_1)

        y_hat_so_far = y_hat_so_far + y_hat_1
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(2, 1)
        y_res_2, y_q_2, y_hat_2, s_hat_2 = self.process_with_mask(y, scales, means, mask_2)

        y_hat_so_far = y_hat_so_far + y_hat_2
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(2, 1)
        y_res_3, y_q_3, y_hat_3, s_hat_3 = self.process_with_mask(y, scales, means, mask_3)

        y_res = (y_res_0 + y_res_1) + (y_res_2 + y_res_3)
        y_q = (y_q_0 + y_q_1) + (y_q_2 + y_q_3)
        y_hat = y_hat_so_far + y_hat_3
        scales_hat = (s_hat_0 + s_hat_1) + (s_hat_2 + s_hat_3)

        y_hat = y_hat * q_dec

        return y_res, y_q, y_hat, scales_hat

    def compress_four_part_prior(self, y, common_params,
                                y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                y_spatial_prior_adaptor_3, y_spatial_prior,
                                y_spatial_prior_reduction=None, write=False):
        '''
        y_0 means split in channel, the 0/4 quater
        y_1 means split in channel, the 1/4 quater
        y_2 means split in channel, the 2/4 quater
        y_3 means split in channel, the 3/4 quater
        y_?_0, means multiply with mask_0
        y_?_1, means multiply with mask_1
        y_?_2, means multiply with mask_2
        y_?_3, means multiply with mask_3
        '''
        q_enc, q_dec, scales, means = self.separate_prior(common_params)
        if y_spatial_prior_reduction is not None:
            common_params = y_spatial_prior_reduction(common_params)
        dtype = y.dtype
        device = y.device
        B, C, H, W = y.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(B, C, H, W, dtype, device)

        y = y * q_enc

        y_res_0, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask(y, scales, means, mask_0)

        y_hat_so_far = y_hat_0
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(2, 1)
        y_res_1, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y, scales, means, mask_1)

        y_hat_so_far = y_hat_so_far + y_hat_1
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(2, 1)
        y_res_2, y_q_2, y_hat_2, s_hat_2 = self.process_with_mask(y, scales, means, mask_2)

        y_hat_so_far = y_hat_so_far + y_hat_2
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(2, 1)
        y_res_3, y_q_3, y_hat_3, s_hat_3 = self.process_with_mask(y, scales, means, mask_3)

        # y_res = (y_res_0 + y_res_1) + (y_res_2 + y_res_3)
        # y_q = (y_q_0 + y_q_1) + (y_q_2 + y_q_3)
        y_hat = y_hat_so_far + y_hat_3
        # scales_hat = (s_hat_0 + s_hat_1) + (s_hat_2 + s_hat_3)

        y_hat = y_hat * q_dec

        y_q_w_0 = self.single_part_for_writing_4x(y_q_0)
        y_q_w_1 = self.single_part_for_writing_4x(y_q_1)
        y_q_w_2 = self.single_part_for_writing_4x(y_q_2)
        y_q_w_3 = self.single_part_for_writing_4x(y_q_3)
        s_w_0 = self.single_part_for_writing_4x(s_hat_0)
        s_w_1 = self.single_part_for_writing_4x(s_hat_1)
        s_w_2 = self.single_part_for_writing_4x(s_hat_2)
        s_w_3 = self.single_part_for_writing_4x(s_hat_3)

        return y_q_w_0, y_q_w_1, y_q_w_2, y_q_w_3, s_w_0, s_w_1, s_w_2, s_w_3, y_hat

    def decompress_four_part_prior(self, common_params,
                                y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                y_spatial_prior_adaptor_3, y_spatial_prior,
                                y_spatial_prior_reduction=None, write=False):
        q_enc, q_dec, scales, means = self.separate_prior(common_params)
        if y_spatial_prior_reduction is not None:
            common_params = y_spatial_prior_reduction(common_params)
        dtype = means.dtype
        device = means.device
        B, C, H, W = means.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(B, C, H, W, dtype, device)

        def restore_y_4x(y, means, mask):
            return (torch.cat((y, y, y, y), dim=1) + means) * mask

        scales_r = self.single_part_for_writing_4x(scales * mask_0)
        y_q_r = self.gaussian_encoder.decode_and_get_y(scales_r, dtype, device)
        y_hat_curr_step = restore_y_4x(y_q_r, means, mask_0)
        y_hat_so_far = y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(2, 1)
        scales_r = self.single_part_for_writing_4x(scales * mask_1)
        y_q_r = self.gaussian_encoder.decode_and_get_y(scales_r, dtype, device)
        y_hat_curr_step = restore_y_4x(y_q_r, means, mask_1)
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(2, 1)
        scales_r = self.single_part_for_writing_4x(scales * mask_2)
        y_q_r = self.gaussian_encoder.decode_and_get_y(scales_r, dtype, device)
        y_hat_curr_step = restore_y_4x(y_q_r, means, mask_2)
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(2, 1)
        scales_r = self.single_part_for_writing_4x(scales * mask_3)
        y_q_r = self.gaussian_encoder.decode_and_get_y(scales_r, dtype, device)
        y_hat_curr_step = restore_y_4x(y_q_r, means, mask_3)
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        y_hat = y_hat_so_far * q_dec

        return y_hat

    def process_with_mask_recon_with_z(self, y, scales, means, mask):
        scales_hat = scales * mask
        means_hat = means * mask

        y_res = (y - means_hat) * mask
        y_q = self.quant(y_res)
        y_hat = y_q * 0. + means_hat

        return y_hat, means_hat

    def forward_four_part_prior_recon_with_z(self, y, common_params,
                                            y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                            y_spatial_prior_adaptor_3, y_spatial_prior,
                                            y_spatial_prior_reduction=None, write=False):
        '''
        y_0 means split in channel, the 0/4 quater
        y_1 means split in channel, the 1/4 quater
        y_2 means split in channel, the 2/4 quater
        y_3 means split in channel, the 3/4 quater
        y_?_0, means multiply with mask_0
        y_?_1, means multiply with mask_1
        y_?_2, means multiply with mask_2
        y_?_3, means multiply with mask_3
        '''
        q_enc, q_dec, scales, means = self.separate_prior(common_params)
        if y_spatial_prior_reduction is not None:
            common_params = y_spatial_prior_reduction(common_params)
        dtype = y.dtype
        device = y.device
        B, C, H, W = y.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(B, C, H, W, dtype, device)

        y = y * q_enc

        y_hat_0, m_hat_0 = self.process_with_mask_recon_with_z(y, scales, means, mask_0)

        y_hat_so_far = y_hat_0
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(2, 1)
        y_hat_1, m_hat_1 = self.process_with_mask_recon_with_z(y, scales, means, mask_1)

        y_hat_so_far = y_hat_so_far + y_hat_1
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(2, 1)
        y_hat_2, m_hat_2 = self.process_with_mask_recon_with_z(y, scales, means, mask_2)

        y_hat_so_far = y_hat_so_far + y_hat_2
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(2, 1)
        y_hat_3, m_hat_3 = self.process_with_mask_recon_with_z(y, scales, means, mask_3)

        y_hat = y_hat_so_far + y_hat_3
        # y_hat = m_hat_0 + m_hat_1 + m_hat_2 + m_hat_3

        y_hat = y_hat * q_dec

        return  y_hat

    @staticmethod
    def encode_z_index(indices: torch.Tensor, bits_per_index: int = 14) -> bytes:
        indices_list = indices.squeeze(-1).cpu().numpy().flatten().tolist()

        bitstream = bytearray()
        buffer = 0
        bits_in_buffer = 0

        for idx in indices_list:
            buffer |= (idx << bits_in_buffer)
            bits_in_buffer += bits_per_index

            while bits_in_buffer >= 8:
                bitstream.append(buffer & 0xFF)
                buffer >>= 8
                bits_in_buffer -= 8

        if bits_in_buffer > 0:
            bitstream.append(buffer & 0xFF)

        return bytes(bitstream)

    @staticmethod
    def decode_z_index(bitstream: bytes, num_indices: int, bits_per_index: int = 14) -> torch.Tensor:
        # num_indices shall be able to be calculated from height/width info from any standard sps.
        indices = []
        buffer = 0
        bits_in_buffer = 0
        byte_pos = 0
        mask = (1 << bits_per_index) - 1

        for _ in range(num_indices):
            while bits_in_buffer < bits_per_index and byte_pos < len(bitstream):
                buffer |= (bitstream[byte_pos] << bits_in_buffer)
                bits_in_buffer += 8
                byte_pos += 1

            idx = buffer & mask
            indices.append(idx)

            buffer >>= bits_per_index
            bits_in_buffer -= bits_per_index

        return torch.tensor(indices, dtype=torch.int64).unsqueeze(-1)
