from dataclasses import dataclass, field
import time
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch import nn

Input = Union[torch.Tensor, Tuple[torch.Tensor, ...]]


@dataclass
class Layer:
    name: str
    children: List['Layer'] = field(default_factory=list)
    level: int = 0


class ForwardTimer:
    """ This class is a Forward timer tracker, to observe how long each step in a 
    forward model takes. NOTE: in order to better observe, a tree-like schematic is developed for printing,
    so it is highly recommended to print in files not in terminal.

    Usage:
        for instance, `ForwardTimer(track_one_iter=True).set_model(my_model, name_of_my_model)`
    Args:
        track_one_iter (bool): if False then track in all iterations, O.W. just in the
            first iteration.

    """
    def __init__(self, track_one_iter: bool = False) -> None:
        
        self._track_one_iter: bool = track_one_iter
        self._level_zero_tracked: bool = False
        self._root_module: Layer = None
        self._time_start: Dict[str, float] = dict()
        self._timer: Dict[str, float] = dict()
        self._handles = list()


    def set_model(self, model: nn.Module, name: str):
        def _hook1(_, __):        
            if not self._track_one_iter or not self._level_zero_tracked:
                print("### START FORWARD TIMER ###")

            if self._track_one_iter and self._level_zero_tracked:
                np.vectorize(lambda handle: handle.remove())(self._handles)
                self._handles.clear()
                print("### FINISH FORWARD TIMER ###", flush=True)
        
        def _hook2(_, __, ___):
            def log(layer: Layer):
                tabs = "\t" * layer.level
                name = layer.name
                
                if name in self._timer:
                    diff = self._timer[name]
                # the name does not exist because of either being in modulelist or not been forwarded at all
                else:
                    try:
                        # modules like `nn.ModuleList` might not be forwarded and therefore the timer is calculated by finding its children's timer.
                        diff = sum([self._timer[ch.name] for ch in layer.children])
                    except:
                        # Obviously there is a module that has not been forwarded :)
                        diff = 0

                print(f"{tabs}{layer.name}: {diff:.4f}", flush=True)

                for ch in layer.children:
                    log(ch)

            log(self._root_module)
            self._level_zero_tracked = True

        self._register_start_time_hook(model, name)
        self._register_timer_hook(model, name)
        self._handles.append(model.register_forward_pre_hook(_hook1))
        self._handles.append(model.register_forward_hook(_hook2))
        
        self._root_module = Layer(name)
        
        prefix = name
        for name, m in model.named_children():
           name = f"{prefix}.{name}"
           submodule = Layer(name, level=self._root_module.level + 1)
           self._root_module.children.append(submodule)
           self._set_submodules_hooks(m, submodule, name)


    def _set_submodules_hooks(self, module: nn.Module, layer: Layer, name: str):
        self._register_start_time_hook(module, name)
        self._register_timer_hook(module, name)

        prefix = name
        for name, m in module.named_children():
           name = f"{prefix}.{name}"
           submodule = Layer(name, level=layer.level + 1)
           layer.children.append(submodule)
           self._set_submodules_hooks(m, submodule, name) 

    def _register_start_time_hook(self, module: nn.Module, name: str):
        def _hook(_ , __):
            t = time.time()
            self._time_start[name] = t

        self._handles.append(module.register_forward_pre_hook(_hook))


    def _register_timer_hook(self, module: nn.Module, name: str):
        def _timer_hook(_, __, ___) -> None:
            t1 = self._time_start[name]            
            diff = time.time() - t1
            if name not in self._timer:
                self._timer[name] = 0
            self._timer[name] += diff
                        
        self._handles.append(module.register_forward_hook(_timer_hook))