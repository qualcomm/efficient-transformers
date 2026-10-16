# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------
from types import MethodType
from typing import Callable, Dict, Optional, Tuple, Type

from torch import nn

from QEfficient.utils.logging_utils import QEFFLogger

logger = QEFFLogger.get_logger("INFRA")


class PytorchTransform:
    """
    PytorchTransform is the base class that can do any transformation to a given PyTorch module by overriding apply method.
    """

    def __init__(self):
        raise TypeError("Transform classes are not to be instantiated. Directly use the `apply` method.")

    @classmethod
    def apply(cls, model: nn.Module) -> Tuple[nn.Module, bool]:
        """
        Override this class method to apply a transformation.
        :param model: The torch module to transform, this module may be transformed in-place

        :returns: Torch module after applying the transform
        :returns: Boolean indicating whether transform was applied
        """
        raise NotImplementedError("Use subclasses for Pytorch transform")


class ProxyModuleMappingTransform(PytorchTransform):
    """
    Replaces the PyTorch modules based on the _module_mapping class variable.
    """

    _module_mapping: Dict[Type[nn.Module], Type[nn.Module]]

    @classmethod
    def apply(cls, model: nn.Module) -> Tuple[nn.Module, bool]:
        transformed = False
        for name, module in model.named_modules():
            for base_type, repl_type in cls._module_mapping.items():
                if isinstance(module, base_type):
                    if base_type is nn.Linear:
                        short_name = name.split(".")[-1] if name else ""
                        if short_name != "lm_head":
                            continue
                    # Perform in-place class replacement (preserve parameters/state)
                    try:
                        module.__class__ = repl_type
                        transformed = True
                    except Exception as e:
                        logger.warning(f"Failed to replace module {name} ({base_type}) -> {repl_type}: {e}")

        return model, transformed


class ModuleMappingTransform(PytorchTransform):
    """
    Replaces the PyTorch modules based on the _module_mapping class variable.
    """

    _module_mapping: Dict[Type[nn.Module], Type[nn.Module]]

    @classmethod
    def apply(cls, model: nn.Module) -> Tuple[nn.Module, bool]:
        transformed = False
        for module in model.modules():
            if repl_module := cls._module_mapping.get(type(module)):
                module.__class__ = repl_module
                # Handling the __init__ calls in the models
                if hasattr(module, "__qeff_init__"):
                    module.__qeff_init__()
                transformed = True
        return model, transformed

    @classmethod
    def register(cls, from_module: Type[nn.Module], to_module: Type[nn.Module]):
        """
        Add a new module type in the module mapping for this transform. ::
            FlashAttention.register(LLamaAttention, LlamaFlashAttention)
        """
        cls._module_mapping[from_module] = to_module


class ModuleMutatorTransform(PytorchTransform):
    """Serves as base class for any transform that mutates pytorch module in any way.
    Mutate here mean, we initialize a new pytorch module object using info from original module and
    replace original module with new module.

    Raises:
        NotImplementedError: Not supposed to use directly, Create a subclass and implement mutate method and assign a valid nn.Module class to _match_class variable.
    """

    _match_class: nn.Module
    _match_string: Optional[str] = None

    @classmethod
    def apply(cls, model: nn.Module) -> Tuple[nn.Module, bool]:
        transformed = False
        for name, module in model.named_children():
            if isinstance(module, cls._match_class):
                setattr(model, name, cls.mutate(module, model))
                transformed = True
            else:
                cls.apply(module)

        if isinstance(model, cls._match_class):
            model = cls.mutate(model, None)
            transformed = True

        return model, transformed

    @classmethod
    def mutate(cls, original_module: nn.Module, parent_module: nn.Module):
        raise NotImplementedError("Please implement your own method by inheriting this class")


class ExternalModuleMapperTransform(PytorchTransform):
    """
    Serves as base class for any transform that want to map a particular method of a class to a new method implementation.
    """

    _match_class_replace_method: Dict[nn.Module, Dict[str, Callable]]
    _match_string_replace_method: Dict[str, Dict[str, Callable]]

    @classmethod
    def apply(cls, model: nn.Module) -> Tuple[nn.Module, bool]:
        transformed = False
        for module in model.modules():
            if (repl_method_map := cls._match_class_replace_method.get(type(module))) or (
                repl_method_map := cls._match_string_replace_method.get(module.__class__.__name__)
            ):
                for orig_method_name, mapped_method in repl_method_map.items():
                    parts = orig_method_name.split(".")
                    if len(parts) > 1:
                        target = module
                        for part in parts[:-1]:
                            target = getattr(target, part)
                        if callable(mapped_method):
                            mapped_method = MethodType(mapped_method, target)
                        setattr(target, parts[-1], mapped_method)
                    else:
                        if callable(mapped_method):
                            mapped_method = MethodType(mapped_method, module)
                        setattr(module, orig_method_name, mapped_method)

                if hasattr(module, "__qeff_init__"):
                    module.__qeff_init__()

                transformed = True

        return model, transformed
