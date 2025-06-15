"""
Adapted from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/common/rotation_transformer.py
"""

from typing import Union
import imitation.utils.pytorch3d_transforms as pt
import torch
import numpy as np
import functools

class RotationTransformer:
    valid_reps = [
        'axis_angle',
        'euler_angles',
        'quaternion',
        'rotation_6d',
        'matrix'
    ]

    def __init__(self, 
            rep_in='axis_angle', 
            rep_network='rotation_6d',
            rep_out='matrix',
            convention_in=None,
            convention_network=None,
            convention_out=None):
        """
        Valid representations:
        - rep_in: Input representation
        - rep_network: Network representation (intermediate)
        - rep_out: Output representation

        Always uses matrix as intermediate representation for conversions.
        """
        if rep_in == rep_network == rep_out:
            self.identity = True
            return
        else:
            self.identity = False

        assert rep_in in self.valid_reps
        assert rep_network in self.valid_reps
        assert rep_out in self.valid_reps

        if rep_in == 'euler_angles':
            assert convention_in is not None
        if rep_network == 'euler_angles':
            assert convention_network is not None
        if rep_out == 'euler_angles':
            assert convention_out is not None

        # Setup preprocessing (in -> network)
        self.preprocess_funcs = []
        if rep_in != rep_network:
            funcs = [
                getattr(pt, f'{rep_in}_to_matrix'),
                getattr(pt, f'matrix_to_{rep_network}')
            ]
            if convention_in is not None:
                funcs[0] = functools.partial(funcs[0], convention=convention_in)
            if convention_network is not None:
                funcs[1] = functools.partial(funcs[1], convention=convention_network)
            self.preprocess_funcs = funcs

        # Setup postprocessing (network -> out)
        self.postprocess_funcs = []
        if rep_network != rep_out:
            funcs = [
                getattr(pt, f'{rep_network}_to_matrix'),
                getattr(pt, f'matrix_to_{rep_out}')
            ]
            if convention_network is not None:
                funcs[0] = functools.partial(funcs[0], convention=convention_network)
            if convention_out is not None:
                funcs[1] = functools.partial(funcs[1], convention=convention_out)
            self.postprocess_funcs = funcs

    @staticmethod
    def _apply_funcs(x: Union[np.ndarray, torch.Tensor], funcs: list) -> Union[np.ndarray, torch.Tensor]:
        x_ = x
        if isinstance(x, np.ndarray):
            x_ = torch.from_numpy(x)
        x_: torch.Tensor
        for func in funcs:
            x_ = func(x_)
        y = x_
        if isinstance(x, np.ndarray):
            y = x_.numpy()
        return y
    
    def __call__(self, *args, **kwargs):
        return self.preprocess(*args, **kwargs)
        
    def preprocess(self, x: Union[np.ndarray, torch.Tensor]
        ) -> Union[np.ndarray, torch.Tensor]:
        """Transform from input representation to network representation"""
        if self.identity:
            return x
        return self._apply_funcs(x, self.preprocess_funcs)
    
    def postprocess(self, x: Union[np.ndarray, torch.Tensor]
        ) -> Union[np.ndarray, torch.Tensor]:
        """Transform from network representation to output representation"""
        if self.identity:
            return x
        return self._apply_funcs(x, self.postprocess_funcs)


def test():
    # Test in -> network -> out transformation
    tf = RotationTransformer(
        rep_in='axis_angle',
        rep_network='rotation_6d',
        rep_out='matrix'
    )

    rotvec = np.random.uniform(-2*np.pi, 2*np.pi, size=(1000,3))
    rot6d = tf.preprocess(rotvec)
    mat = tf.postprocess(rot6d)

    # Verify the transformation preserves rotation properties
    mat_det = np.linalg.det(mat)
    assert np.allclose(mat_det, 1)
    # rotation_6d will be normalized to rotation matrix
