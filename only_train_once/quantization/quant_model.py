import logging
import math
from typing import Dict, Union, Optional

import torch.nn as nn

from .quant_layers import LAYER_TO_QUANTLAYER, QuantizationMode, QuantizationType
from .quant_layers import QuantizeConv2d, QuantizeLinear

"""
The model_to_quantize_model supports 
    - both weight and activation quantization
    - both linear quantization (no t) and nonlinear quantization (yes t)

The unwrap_quantized_model function converts quantization-aware layers
back to standard PyTorch layers, preserving the learned weight values.
"""

def model_to_quantize_model(
    model: nn.Module,
    d_quant_init: float = 1e-4,
    t_quant_init: float = 1.0,
    q_m_init: float = 1.0,
    quant_init_by_module: bool = True,
    num_bits: int = 16,
    quant_type: Union[QuantizationType, str] = QuantizationType.SYMMETRIC_NONLINEAR,
    quant_mode: Union[QuantizationMode, str] = QuantizationMode.WEIGHT_ONLY,
) -> nn.Module:
    """Convert model layers to quantized versions.

    Args:
        model: PyTorch model to convert
        d_quant_init: Initial quantization step size
        t_quant_init: Initial nonlinearity parameter
        q_m_init: Initial quantization range
        quant_init_by_module: Whether to initialize quantization parameters from module weights
        num_bits: Number of bits for quantization
        quant_type: Type of quantization (linear or nonlinear)
        quant_mode: Mode of quantization (weight-only or weight+activation)

    Returns:
        Modified model with quantized layers.
    """

    def _get_submodules(model, key):
        parent_module = model.get_submodule(".".join(key.split(".")[:-1]))
        target_name_in_parent_module = key.split(".")[-1]
        target_module = model.get_submodule(key)
        return parent_module, target_module, target_name_in_parent_module

    logger = logging.getLogger(__name__)
    if isinstance(quant_type, str):
        try:
            quant_type = QuantizationType(quant_type)
        except ValueError:
            raise ValueError(
                f"Invalid quantization type: {quant_type}. Must be one of {[t.value for t in QuantizationType]}"
            )

    if isinstance(quant_mode, str):
        try:
            quant_mode = QuantizationMode(quant_mode)
        except ValueError:
            raise ValueError(
                f"Invalid quantization mode: {quant_mode}. Must be one of {[m.value for m in QuantizationMode]}"
            )

    converted_layers = 0
    for name, module in model.named_modules():
        if type(module).__name__ in LAYER_TO_QUANTLAYER:
            parent_module, target_module, target_name = _get_submodules(model, name)
            quant_module = LAYER_TO_QUANTLAYER[type(module).__name__].from_module(
                module=target_module,
                d_quant_init=d_quant_init,
                t_quant_init=t_quant_init,
                q_m_init=q_m_init,
                quant_type=quant_type,
                quant_mode=quant_mode,
                quant_init_by_module=quant_init_by_module,
                num_bits=num_bits,
            )
            setattr(parent_module, target_name, quant_module)
            converted_layers += 1

    logger.info(f"Converted {converted_layers} layers to quantized versions")
    return model


def unwrap_quantized_model(
    model: nn.Module,
    layer_types: Optional[list] = None,
    inplace: bool = False,
) -> nn.Module:
    """
    Convert quantization-aware layers back to standard PyTorch layers.
    
    This unwraps QuantizeLinear and QuantizeConv2d layers back to regular Linear
    and Conv2d layers, preserving the learned weight values. The resulting model
    contains clean float32 weights with a quantization-friendly distribution,
    ready for ONNX export and QDQ quantization.
    
    IMPORTANT: The learned weights are preserved! The quantization-aware training
    shaped the weight distribution to be quantization-friendly. This learning is
    encoded in the weight VALUES, not in the QuantizeLinear wrapper.
    
    Args:
        model: Model with quantization-aware layers
        layer_types: List of layer types to unwrap (default: ["Linear", "Conv2d"])
        inplace: If True, modify model in place. If False, work on a copy.
        
    Returns:
        Model with standard PyTorch layers (preserving learned weights)
        
    Note:
        The unwrapped model will have:
        - Regular Linear/Conv2d layers (no quantization simulation)
        - Float32 weights with quantization-friendly distribution
        - No quantization parameters (d_quant, q_m, t_quant)
    """
    if layer_types is None:
        layer_types = ["Linear", "Conv2d"]
    
    # Work on a copy unless inplace=True
    if not inplace:
        import copy
        model = copy.deepcopy(model)
    
    for name, module in model.named_children():
        # Recursively process child modules
        if len(list(module.children())) > 0:
            unwrap_quantized_model(module, layer_types, inplace=True)
        
        # Unwrap QuantizeLinear to Linear
        if "Linear" in layer_types and isinstance(module, QuantizeLinear):
            standard_layer = nn.Linear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
            )
            # Copy the learned weights (this is what we want to keep!)
            standard_layer.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                standard_layer.bias.data.copy_(module.bias.data)
            
            setattr(model, name, standard_layer)
        
        # Unwrap QuantizeConv2d to Conv2d
        elif "Conv2d" in layer_types and isinstance(module, QuantizeConv2d):
            standard_layer = nn.Conv2d(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None,
            )
            # Copy the learned weights (this is what we want to keep!)
            standard_layer.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                standard_layer.bias.data.copy_(module.bias.data)
            
            setattr(model, name, standard_layer)
    
    return model


def get_quantization_info(model: nn.Module) -> dict:
    """
    Get quantization information from a quantized model.
    """
    info = {}
    
    for name, module in model.named_modules():
        if isinstance(module, (QuantizeLinear, QuantizeConv2d)):
            layer_info = {
                "d_quant_wt": module.d_quant_wt.item(),
                "q_m_wt": module.q_m_wt.item(),
                "weight_bit": module.weight_bit,
                "quant_type": module.quant_type,
                "quant_mode": module.quant_mode,
            }
            
            # Add t_quant if using nonlinear quantization
            if module.quant_type == QuantizationType.SYMMETRIC_NONLINEAR:
                layer_info["t_quant_wt"] = module.t_quant_wt.item()
            
            # Add activation quantization params if used
            if module.quant_mode == QuantizationMode.WEIGHT_AND_ACTIVATION:
                layer_info["d_quant_act"] = module.d_quant_act.item()
                layer_info["q_m_act"] = module.q_m_act.item()
                layer_info["activation_bit"] = module.activation_bit
                if module.quant_type == QuantizationType.SYMMETRIC_NONLINEAR:
                    layer_info["t_quant_act"] = module.t_quant_act.item()
            
            info[name] = layer_info
    
    return info


def compare_weight_distributions(
    quantized_model: nn.Module,
    unwrapped_model: nn.Module,
    layer_name: str = None,
) -> None:
    """
    Verify that unwrapping preserved the weight distributions.
    """
    import torch
    
    all_match = True
    
    for (name1, mod1), (name2, mod2) in zip(
        quantized_model.named_modules(), 
        unwrapped_model.named_modules()
    ):
        if layer_name and name1 != layer_name:
            continue
            
        # Check Linear layers
        if isinstance(mod1, QuantizeLinear) and isinstance(mod2, nn.Linear):
            if not torch.allclose(mod1.weight.data, mod2.weight.data):
                print(f"Weights differ in {name1}")
                all_match = False
            elif layer_name:
                print(f"Weights match in {name1}")
                print(f"  Shape: {mod1.weight.shape}")
                print(f"  Range: [{mod1.weight.min():.4f}, {mod1.weight.max():.4f}]")
        
        # Check Conv2d layers
        elif isinstance(mod1, QuantizeConv2d) and isinstance(mod2, nn.Conv2d):
            if not torch.allclose(mod1.weight.data, mod2.weight.data):
                print(f"Weights differ in {name1}")
                all_match = False
            elif layer_name:
                print(f"Weights match in {name1}")
                print(f"  Shape: {mod1.weight.shape}")
                print(f"  Range: [{mod1.weight.min():.4f}, {mod1.weight.max():.4f}]")
    
    if all_match and not layer_name:
        print("All weights match!")


def get_quant_param_dict(model: nn.Module) -> Dict[str, Dict[str, float]]:
    """Extract quantization parameters from a model.

    Args:
        model: PyTorch model with quantized layers
    Returns:
        Dictionary mapping layer names to their quantization parameters
    """
    param_dict = {}
    for name, param in model.named_parameters():
        if any(qtype in name for qtype in ["d_quant", "t_quant", "q_m"]):
            layer_name = ".".join(name.split(".")[:-1])
            param_name = name.split(".")[-1]
            if layer_name not in param_dict:
                param_dict[layer_name] = {}
            param_dict[layer_name][param_name] = param.item()
    return param_dict


def get_bitwidth_dict(
    param_dict: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Calculate bitwidths for weights and activations of each layer.

    Args:
        param_dict: Dictionary mapping layer names to their quantization parameters

    Returns:
        Dictionary mapping layer names to their calculated bitwidths for weights and activations
    """
    bit_dict: Dict[str, Dict[str, float]] = {}

    def _calculate_bitwidth(d_quant: float, q_m: float, t_quant: float = 1.0) -> float:
        return math.log2(math.exp(t_quant * math.log(abs(q_m))) / abs(d_quant) + 1) + 1

    for key, params in param_dict.items():
        bit_dict[key] = {}
        # weight bitwidths
        d_quant_wt = params["d_quant_wt"]
        q_m_wt = abs(params["q_m_wt"])
        t_quant_wt = params.get("t_quant_wt", 1.0)
        bit_width_wt = _calculate_bitwidth(d_quant_wt, q_m_wt, t_quant_wt)
        bit_dict[key]["weight"] = bit_width_wt
        # activation bitwidths
        if "d_quant_act" in params:
            d_quant_act = params["d_quant_act"]
            q_m_act = abs(params["q_m_act"])
            t_quant_act = params.get("t_quant_act", 1.0)
            bit_width_act = _calculate_bitwidth(d_quant_act, q_m_act, t_quant_act)
            bit_dict[key]["activation"] = bit_width_act

    return bit_dict
