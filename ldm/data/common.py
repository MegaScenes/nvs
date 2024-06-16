import math
import numpy as np
import torch
import sys
import time


def central_crop(raw_im):
    h, w = raw_im.size
    s = min(h, w)
    oh = (h - s) // 2
    ow = (w - s) // 2
    # raw_im.crop([ow, oh, w - ow, h - oh])
    # display(raw_im)
    new_raw_im = raw_im.crop([oh, ow, h - oh, w - ow])
    new_raw_im = new_raw_im.resize((s, s))
    return new_raw_im


def central_crop_v2(image):
    h, w = image.size
    s = min(h, w)
    # print(s)
    oh = (h - s) // 2
    oh_resid = (h - s) % 2
    ow = (w - s) // 2
    ow_resid = (w - s) % 2
    crop_bounds = [oh, ow, h - oh - oh_resid, w - ow - ow_resid]
    # print(crop_bounds)
    new_image = image.crop(crop_bounds)
    assert new_image.size == (s, s), (image.size, (s, s), new_image.size)
    return new_image


def cartesian_to_spherical(xyz):
    # ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    z = np.sqrt(xy + xyz[:, 2] ** 2)
    theta = np.arctan2(
        np.sqrt(xy), xyz[:, 2]
    )  # for elevation angle defined from Z-axis down
    # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
    return np.array([theta, azimuth, z])


def get_T(target_RT, cond_RT, to_torch=True):
    R, T = target_RT[:3, :3], target_RT[:, -1]
    T_target = -R.T @ T

    R, T = cond_RT[:3, :3], cond_RT[:, -1]
    T_cond = -R.T @ T

    theta_cond, azimuth_cond, z_cond = cartesian_to_spherical(T_cond[None, :])
    theta_target, azimuth_target, z_target = cartesian_to_spherical(T_target[None, :])

    d_theta = theta_target - theta_cond
    d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
    d_z = z_target - z_cond

    d_T = np.array(
        [
            d_theta.item(),
            math.sin(d_azimuth.item()),
            math.cos(d_azimuth.item()),
            d_z.item(),
        ]
    )
    if to_torch:
        d_T = torch.tensor(d_T)
    return d_T


def cartesian_to_spherical_torch(xyz):
    xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    z = torch.sqrt(xy + xyz[:, 2] ** 2)
    theta = torch.arctan2(
        torch.sqrt(xy), xyz[:, 2]
    )  # for elevation angle defined from Z-axis down
    azimuth = torch.arctan2(xyz[:, 1], xyz[:, 0])
    return theta, azimuth, z


def get_T_torch(target_inhomogeneous_world2cam, cond_inhomogeneous_world2cam):
    target_RT = target_inhomogeneous_world2cam
    cond_RT = cond_inhomogeneous_world2cam

    R, T = target_RT[:, :3, :3], target_RT[:, :, -1]
    # import pdb
    # pdb.set_trace()
    T_target = torch.bmm(-R.permute((0, 2, 1)), T[..., None])

    R, T = cond_RT[:, :3, :3], cond_RT[:, :, -1]
    T_cond = torch.bmm(-R.permute((0, 2, 1)), T[..., None])

    theta_cond, azimuth_cond, z_cond = cartesian_to_spherical_torch(T_cond)
    theta_target, azimuth_target, z_target = cartesian_to_spherical_torch(T_target)

    d_theta = theta_target - theta_cond
    d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
    d_z = z_target - z_cond

    return torch.cat([d_theta, torch.sin(d_azimuth), torch.cos(d_azimuth), d_z], axis=1)


def _get_relative_transformations(batch, scales):
    # this is the correct representation to use because it's invariant to
    # shift but not scale

    assert scales.ndim == 2

    target_cam2world = batch["target_cam2world"].detach().clone()
    cond_cam2world = batch["cond_cam2world"].detach().clone()

    batch_size = cond_cam2world.shape[0]

    relative_target_transformation = torch.linalg.inv(cond_cam2world) @ target_cam2world
    relative_target_transformation[:, :3, -1] /= torch.clip(scales, min=1e-2, max=None)

    assert relative_target_transformation.shape == (batch_size, 4, 4)
    return relative_target_transformation


def load_depth_model(depth_model_name):
    if depth_model_name == "midas":
        # depth_repo = "isl-org/MiDaS"
        depth_repo = "/home/jupyter/.cache/torch/hub/isl-org_MiDaS_master"
        depth_arch = "DPT_SwinV2_T_256"
        depth_model = torch.hub.load(
            depth_repo, depth_arch, pretrained=True, source="local"
        )
        depth_model = depth_model.eval()
        return depth_model
    elif depth_model_name == "zoedepth":
        raise NotImplementedError
    else:
        raise NotImplementedError


def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def compute_scale_and_shift_to_01(target_disparity, mask):
    assert torch.any(mask.reshape(mask.shape[0], -1), dim=1).all()
    current_target_disparity = target_disparity.clone()
    current_target_disparity[~mask] = torch.finfo(torch.float32).min
    max_disparity = torch.amax(current_target_disparity, dim=[1, 2])

    current_target_disparity = target_disparity.clone()
    current_target_disparity[~mask] = torch.finfo(torch.float32).max
    min_disparity = torch.amin(current_target_disparity, dim=[1, 2])

    # return locals()

    scale = 1.0 / (max_disparity - min_disparity)
    shift = -min_disparity * scale
    return scale, shift


@torch.no_grad()
def get_aligned_monodepths(depth_model, batch):
    device = next(depth_model.parameters()).device
    if device.type == "cpu":  # hack
        device = batch["T"].device
        depth_model.to(device)

    disparity_max = 1000
    disparity_min = 0.001

    depth_max = 1 / disparity_min
    depth_min = 1 / disparity_max

    image_cond_p = batch["image_cond"].permute((0, 3, 1, 2)).to(device)

    # import pdb
    # pdb.set_trace()

    estimated_dense_disparity = depth_model(image_cond_p)
    colmap_depth = batch["depth_cond"][..., 0].to(device)

    # return estimated_dense_disparity, colmap_depth

    assert colmap_depth.shape == estimated_dense_disparity.shape, (
        colmap_depth.shape,
        estimated_dense_disparity.shape,
    )

    # print(
    #     estimated_dense_disparity.min(),
    #     estimated_dense_disparity.mean(),
    #     estimated_dense_disparity.max(),
    # )

    mask = colmap_depth != 0

    # print(
    #     colmap_depth.min(),
    #     colmap_depth.mean(),
    #     colmap_depth.max(),
    # )

    target = colmap_depth
    target = torch.clip(target, depth_min, depth_max)

    prediction = estimated_dense_disparity

    target_disparity = torch.zeros_like(target)
    target_disparity[mask == 1] = 1.0 / target[mask == 1]

    # compute scale and shift to map target disparity to [0, 1] a la midas
    # gt_scale, gt_shift = compute_scale_and_shift_to_01(target_disparity.clone(), mask)
    # target_disparity = target_disparity * gt_scale.view(-1, 1, 1) + gt_shift.view(
    #     -1, 1, 1
    # )
    # target_disparity[~mask] = 0.

    scale, shift = compute_scale_and_shift(prediction, target_disparity, mask)

    prediction_aligned = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)
    # prediction_aligned = (prediction_aligned - gt_shift.view(-1, 1, 1)) / (
    #     gt_scale.view(-1, 1, 1)
    # )

    prediction_aligned[prediction_aligned > disparity_max] = disparity_max
    prediction_aligned[prediction_aligned < disparity_min] = disparity_min
    prediction_depth = 1.0 / prediction_aligned
    return prediction_depth


@torch.no_grad()
def compute_T(config, depth_model, batch, precomputed_scale=None, return_aux=False):
    batch_size = batch["cond_cam2world"].shape[0]
    if precomputed_scale is not None:
        precomputed_scale = precomputed_scale * torch.ones(
            size=(batch_size, 1),
            dtype=batch["cond_cam2world"].dtype,
            device=batch["cond_cam2world"].device,
        )

    import ipdb
    ipdb.set_trace()
    if "T" not in batch:
        batch["T"] = get_T_torch(
            torch.linalg.inv(batch["target_cam2world"])[:, :3],
            torch.linalg.inv(batch["cond_cam2world"])[:, :3],
        ) # this returns zero123 spherical coordinates; overwritten below!!

    if config.params.mode == "3dof":
        # print("Computed 3dof representation!")
        return batch["T"]

    elif config.params.mode == "7dof":
        if precomputed_scale is None:
            scale = batch["scene_radius"][:, None]
        else:
            scale = precomputed_scale

        relative_transformations = _get_relative_transformations(batch, scale)

        fov_rad = batch["fov_deg"] * np.pi / 180
        fov_enc = torch.stack(
            [fov_rad, torch.sin(fov_rad), torch.cos(fov_rad)], axis=-1
        )
        relative_transformations = relative_transformations.reshape((batch_size, 16))

        T = torch.cat([relative_transformations, fov_enc], axis=1)
        return T
    elif config.params.mode == "7dof_unscaled":
        scale = torch.ones(
            size=(batch_size, 1),
            dtype=batch["cond_cam2world"].dtype,
            device=batch["cond_cam2world"].device,
        )

        relative_transformations = _get_relative_transformations(batch, scale)

        fov_rad = batch["fov_deg"] * np.pi / 180
        fov_enc = torch.stack(
            [fov_rad, torch.sin(fov_rad), torch.cos(fov_rad)], axis=-1
        )
        relative_transformations = relative_transformations.reshape((batch_size, 16))
        relative_transformations = torch.clip(relative_transformations, -20, 20)
        relative_transformations = torch.nan_to_num(
            relative_transformations, nan=0, posinf=20, neginf=-20
        )

        T = torch.cat([relative_transformations, fov_enc], axis=1)
        return T
    elif config.params.mode == "7dof_stereomag_scale":
        if precomputed_scale is None:
            scales = torch.clip(batch["nearplane_quantile"][:, None], 0.1, 100)
            scales = torch.nan_to_num(scales, nan=1.0, posinf=100, neginf=0.1)
        else:
            scales = precomputed_scale

        relative_transformations = _get_relative_transformations(batch, scales)

        fov_rad = batch["fov_deg"] * np.pi / 180
        fov_enc = torch.stack(
            [fov_rad, torch.sin(fov_rad), torch.cos(fov_rad)], axis=-1
        )
        relative_transformations = relative_transformations.reshape((batch_size, 16))
        relative_transformations = torch.clip(relative_transformations, -20, 20)
        relative_transformations = torch.nan_to_num(
            relative_transformations, nan=0, posinf=20, neginf=-20
        )

        T = torch.cat([relative_transformations, fov_enc], axis=1)
        return T
    elif config.params.mode == "7dof_quantile_scale":
        # tic = time.time()
        if precomputed_scale is None: # False
            monodepths = get_aligned_monodepths(depth_model, batch).to(
                batch["depth_cond"].device
            )[..., None]
            # toc = time.time()
            # print("got depth in time ", toc - tic )

            # tic = time.time()
            dense_depths = monodepths.clone()

            if config.params.quantile_scale_blend:
                dense_depths[batch["depth_cond"] != 0] = batch["depth_cond"][
                    batch["depth_cond"] != 0
                ]
            scales = torch.quantile(
                dense_depths[:, ::4, ::4].reshape((batch_size, -1)),
                q=0.2,
                dim=1,
                keepdim=True,
            )
        else:
            scales = precomputed_scale
        # print(scales.max(), scales.min())
        # toc = time.time()
        # print("got scales in time", toc - tic )
        # scales = torch.clip(scales, min=.1, max=100)

        # print(
        #     "scales true",
        #     batch["depth_cond"].min().item(),
        #     batch["depth_cond"].mean().item(),
        #     batch["depth_cond"].max().item(),
        # )
        # print("scales", scales.min().item(), scales.mean().item(), scales.max().item())
        # scales = 0

        # this functions returns E = inv(cond)@target (i.e. go from cond to target)
        # and scales the translation by 'scales'
        relative_transformations = _get_relative_transformations(batch, scales)
        a = 0.5
        c = 10
        # we do a transformation of the translation. the intuition behind a and c is that
        # we want the transformation to be approximately linear for like 99% of batches
        # but when the translations are very large we want to smoothly ramp down to some
        # reasonable value rather than explode.
        relative_transformations[:, :3, -1] = (
            a * c * torch.tanh(relative_transformations[:, :3, -1] / c)
        )
        relative_transformations = torch.nan_to_num(
            relative_transformations, nan=0, posinf=c, neginf=-c
        )

        # import pdb
        #ipdb.set_trace()

        # relative_transformations = torch.clip(relative_transformations, -20, 20)

        fov_rad = batch["fov_deg"] * np.pi / 180
        fov_enc = torch.stack(
            [fov_rad, torch.sin(fov_rad), torch.cos(fov_rad)], axis=-1
        )
        relative_transformations = relative_transformations.reshape((batch_size, 16))

        T = torch.cat([relative_transformations, fov_enc], axis=1)

        ipdb.set_trace()

        if return_aux:
            return T, locals()
        else:
            return T
        # return locals()
    elif config.params.mode == "7dof_metric_scale":
        print("grounding poses in metric system!")

        image_cond_input = (
            batch["image_cond"]
            .to(next(depth_model.parameters()).device)
            .permute((0, 3, 1, 2))
        ) / 2 + 0.5

        estimated_metric_depths = depth_model.infer(
            image_cond_input,
            with_flip_aug=False,
        )

        estimated_metric_depths = estimated_metric_depths.to(
            batch["depth_cond"].device
        )[:, 0]

        gt_depths = batch["depth_cond"][..., 0]  # from colmap
        assert gt_depths.shape == estimated_metric_depths.shape, (
            gt_depths.shape,
            estimated_metric_depths.shape,
        )

        mask = gt_depths != 0
        estimated_metric_depths[~mask] = 0

        assert estimated_metric_depths.shape == gt_depths.shape
        assert gt_depths.ndim == 3

        ab = (estimated_metric_depths * gt_depths).sum(axis=[1, 2], keepdims=True)
        aa = (estimated_metric_depths * estimated_metric_depths).sum(
            axis=[1, 2], keepdims=True
        )

        # ab/aa aligns metric depth to colmap depth
        # meters_per_colmap = aa / torch.clip(ab, 1e-3, None)
        colmaps_per_meter = ab / torch.clip(aa, 1e-3, None)

        # aligned_estimated_metric_depths = (
        #     estimated_metric_depths / meters_per_colmap
        # )
        # return locals()

        assert colmaps_per_meter.shape == (batch_size, 1, 1)
        scales = colmaps_per_meter[..., 0]
        assert scales.shape == (batch_size, 1)

        relative_transformations = _get_relative_transformations(batch, scales)
        fov_rad = batch["fov_deg"] * np.pi / 180
        fov_enc = torch.stack(
            [fov_rad, torch.sin(fov_rad), torch.cos(fov_rad)], axis=-1
        )
        relative_transformations = relative_transformations.reshape((batch_size, 16))

        T = torch.cat([relative_transformations, fov_enc], axis=1)
        return T
    else:
        raise NotImplementedError
