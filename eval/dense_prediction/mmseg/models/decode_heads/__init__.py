# Copyright (c) OpenMMLab. All rights reserved.
from .dpt_head import DPTHead
from .dpt_head_depth import DPTHead_depth
from .fcn_head import FCNHead
from .psp_head import PSPHead
from .uper_head import UPerHead


__all__ = [
    'FCNHead', 'PSPHead', 'UPerHead', 'DPTHead', 'DPTHead_depth'
]
