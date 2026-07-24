"""旧模块名兼容层。

新代码请从 :mod:`fsvae_models.fscvae` 导入模型。
"""

from .fscvae import FSCVAE, FSCVAELarge

FSVAE = FSCVAE
FSVAELarge = FSCVAELarge

__all__ = ['FSCVAE', 'FSCVAELarge', 'FSVAE', 'FSVAELarge']
