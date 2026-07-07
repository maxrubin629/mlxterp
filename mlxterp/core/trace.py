"""
Tracing context manager for capturing model execution.

Provides the clean context manager API:
    with model.trace("input text"):
        output = model.layers[3].self_attn.output.save()
"""

import inspect
from typing import Any, Callable, Dict, Optional, Tuple

import mlx.core as mx

from .activation import get_primary_tensor
from .proxy import TraceContext, _find_matching_name


class Trace:
    """
    Context manager for tracing model execution.

    Usage:
        with model.trace(input_data) as trace:
            # Access activations (use self_attn for mlx-lm models)
            attn = model.layers[3].self_attn.output.save()

        # After context, saved values and activations are available
        result = trace.output
        saved = trace.saved_values
        activations = trace.activations
    """

    def __init__(
        self,
        model_forward: Callable,
        inputs: Any = None,
        tokenizer: Optional[Any] = None,
        interventions: Optional[Dict[str, Callable]] = None,
        model_inputs: Optional[Dict[str, Any]] = None,
        interpretable_model: Optional[Any] = None,
    ):
        """
        Initialize a trace.

        Args:
            model_forward: The forward function to call
            inputs: Input to the model (tokens, text, arrays, or a dict of
                    prepared model kwargs such as input_ids/pixel_values)
            tokenizer: Optional tokenizer for text inputs
            interventions: Dict mapping module names to intervention functions
            model_inputs: Direct keyword inputs for the model forward pass
                          (alternative to `inputs`)
            interpretable_model: The InterpretableModel instance (for layer patching)
        """
        self.model_forward = model_forward
        self.inputs = inputs
        self.tokenizer = tokenizer
        self.interventions = interventions or {}
        self.model_inputs = model_inputs or {}
        self.interpretable_model = interpretable_model

        self.context: Optional[TraceContext] = None
        self.output: Any = None
        self.raw_output: Any = None
        self.saved_values: Dict[str, Any] = {}
        self.activations: Dict[str, Any] = {}
        self._patched_modules: Dict[str, Any] = {}  # name -> (parent, attr_name, original_module)
        self._module_map: Dict[str, Any] = {}  # name -> wrapper module (for lookups)

    def __enter__(self) -> 'Trace':
        """
        Enter the trace context.

        Sets up the global trace context and executes the forward pass.
        """
        # Create and push context
        self.context = TraceContext()

        # Register interventions
        for name, fn in self.interventions.items():
            self.context.interventions[name] = fn

        TraceContext.push(self.context)

        # Patch model layers with proxies
        if self.interpretable_model is not None:
            self._patch_model_layers()

        # Execute the forward pass immediately
        # This allows users to access activations in the with block
        try:
            args, kwargs = self._build_forward_call()

            # Execute the forward pass
            # During execution, ModuleProxies will populate the context
            self.raw_output = self.model_forward(*args, **kwargs)
            self.output = self._normalize_output(self.raw_output)

            # Store output in context for access via model.output
            if self.context:
                self.context.activations["__model_output__"] = self.output
                # Copy activations so they're available inside the with block
                self.activations = self.context.activations.copy()
                # Note: saved_values are updated in __exit__ to capture
                # values saved inside the with block via .save() calls
        except Exception as e:
            # Clean up on error
            self._restore_model_layers()
            TraceContext.pop()
            raise

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the trace context.

        Cleans up the global context and restores original layers.
        Copies saved_values and activations from context so they're
        available after the with block.
        """
        # Copy saved_values and activations from context BEFORE cleanup
        # This ensures values saved inside the with block are available
        if self.context:
            self.saved_values = self.context.saved_values.copy()
            self.activations = self.context.activations.copy()

        # Restore original layers
        self._restore_model_layers()

        # Pop the context
        TraceContext.pop()

        # Don't suppress exceptions
        return False

    def _patch_model_layers(self):
        """
        Recursively discover and patch ALL nn.Modules in the model.

        This builds a complete name → module map and patches every module,
        not just top-level layers. This enables nested access like:
            model.layers[3].self_attn.output.save()
        """
        if self.interpretable_model is None:
            return

        import mlx.nn as nn

        model = self.interpretable_model.model

        try:
            # Recursively discover all modules
            discovered = self._discover_modules(model, "model")

            # Patch each discovered module
            patched_count = 0
            skipped_count = 0

            # First pass: organize modules by parent to handle list patching correctly
            modules_by_parent = {}
            for name, info in discovered.items():
                parent_or_list, attr_or_idx, module = info
                parent_id = id(parent_or_list)
                if parent_id not in modules_by_parent:
                    modules_by_parent[parent_id] = []
                modules_by_parent[parent_id].append((name, info))

            # Second pass: patch modules, re-accessing real lists
            for parent_id, module_list in modules_by_parent.items():
                for name, info in module_list:
                    try:
                        parent_or_list, attr_or_idx, module = info
                        wrapper = self._create_layer_wrapper(module, name)

                        # Replace the module in its parent
                        if isinstance(attr_or_idx, int):
                            # It's a list element - need to access the REAL list, not a copy from .children()
                            # Parse the name to find the parent and attribute
                            # e.g., "model.model.layers.0" -> parent="model.model", attr="layers"
                            parts = name.rsplit('.', 1)
                            if len(parts) == 2:
                                parent_path, _  = parts
                                # Find the attribute name (e.g., "layers" from "model.model.layers.0")
                                path_parts = parent_path.split('.')
                                if len(path_parts) >= 2:
                                    attr_name = path_parts[-1]  # Get "layers"

                                    # Navigate to parent object (NOT including the list attribute itself)
                                    # For "model.model.layers", navigate to "model.model", then get "layers"
                                    parent_obj = model
                                    for part in path_parts[1:-1]:  # Skip "model" prefix AND the list attr
                                        parent_obj = getattr(parent_obj, part, None)
                                        if parent_obj is None:
                                            break

                                    if parent_obj is not None:
                                        # Get the REAL list via attribute access
                                        real_list = getattr(parent_obj, attr_name, None)
                                        if real_list is not None and isinstance(real_list, list):
                                            real_list[attr_or_idx] = wrapper
                                            patched_count += 1
                                            # Store for restoration
                                            self._patched_modules[name] = info
                                            self._module_map[name] = wrapper
                                            continue

                            # Fallback: use the list we have (might be a copy)
                            parent_or_list[attr_or_idx] = wrapper
                            patched_count += 1
                        else:
                            # It's a direct attribute - check if it's a property first
                            # Properties can't be replaced, so skip them
                            parent_type = type(parent_or_list)
                            attr_descriptor = getattr(parent_type, attr_or_idx, None)
                            if isinstance(attr_descriptor, property):
                                # Can't replace properties - skip this one
                                # The child modules within will still be wrapped
                                skipped_count += 1
                                continue

                            setattr(parent_or_list, attr_or_idx, wrapper)
                            patched_count += 1

                        # Store for restoration
                        self._patched_modules[name] = info
                        self._module_map[name] = wrapper
                    except (AttributeError, TypeError) as e:
                        # Skip modules that can't be patched (e.g., properties without deleters)
                        # These are typically container properties (like 'layers' or 'model')
                        # The actual modules inside them are still wrapped individually
                        skipped_count += 1
                        continue

            # Optional: Log patching summary for debugging
            # print(f"[Trace] Patched {patched_count} modules, skipped {skipped_count}")

        except Exception as e:
            import warnings
            warnings.warn(f"Failed to patch modules: {e}. Activations may not be captured.")

    def _discover_modules(
        self,
        module,
        prefix: str,
        discovered: Optional[Dict] = None,
        visited: Optional[set] = None
    ) -> Dict:
        """
        Recursively discover all nn.Module instances in a module tree.

        Args:
            module: The module to explore
            prefix: The name prefix for this module
            discovered: Dict of discovered modules (accumulated)
            visited: Set of visited module IDs (for cycle detection)

        Returns:
            Dict mapping module names to (parent, attr_name/index, module) tuples
        """
        import mlx.nn as nn

        if discovered is None:
            discovered = {}

        if visited is None:
            visited = set()

        # Skip if we've already visited this module (cycle detection)
        module_id = id(module)
        if module_id in visited:
            return discovered

        visited.add(module_id)

        # Use MLX's built-in children() method to discover submodules
        if hasattr(module, 'children'):
            try:
                children = module.children()
                for attr_name, child in children.items():
                    if isinstance(child, nn.Module):
                        full_name = f"{prefix}.{attr_name}"

                        # Skip if already discovered
                        if full_name not in discovered and id(child) not in visited:
                            discovered[full_name] = (module, attr_name, child)

                            # Recursively discover submodules
                            self._discover_modules(child, full_name, discovered, visited)

                    # Check if it's a list of modules (like layers)
                    elif isinstance(child, (list, tuple)):
                        for i, item in enumerate(child):
                            if isinstance(item, nn.Module):
                                full_name = f"{prefix}.{attr_name}.{i}"

                                # Skip if already discovered
                                if full_name not in discovered and id(item) not in visited:
                                    # Store list, index, and item for proper restoration
                                    discovered[full_name] = (child, i, item)

                                    # Recursively discover submodules
                                    self._discover_modules(item, full_name, discovered, visited)
            except:
                pass

        return discovered

    def _create_layer_wrapper(self, layer, name):
        """
        Create a simple wrapper object that intercepts __call__ and delegates everything else.

        This uses composition rather than inheritance to avoid issues with MLX's complex module internals.
        """

        class SimpleWrapper:
            """Minimal wrapper that intercepts calls and delegates everything else"""

            def __init__(self, wrapped_layer, layer_name):
                object.__setattr__(self, '_wrapped_layer', wrapped_layer)
                object.__setattr__(self, '_layer_name', layer_name)

            def __call__(self, *args, **kwargs):
                """Intercept calls to capture activations"""
                ctx = TraceContext.current()

                # Call the original layer
                result = self._wrapped_layer(*args, **kwargs)

                # If we're in a trace, capture the activation
                if ctx is not None:
                    # Apply intervention if registered
                    if ctx.should_intervene(self._layer_name):
                        result = ctx.apply_intervention(self._layer_name, result)

                    # Store activation
                    ctx.activations[self._layer_name] = result

                return result

            def __getattr__(self, name):
                """Delegate all other attribute access to wrapped layer"""
                return getattr(self._wrapped_layer, name)

            def __setattr__(self, name, value):
                """Delegate attribute setting to wrapped layer"""
                if name in ('_wrapped_layer', '_layer_name'):
                    object.__setattr__(self, name, value)
                else:
                    setattr(self._wrapped_layer, name, value)

            def __getitem__(self, key):
                """Delegate dictionary access"""
                return self._wrapped_layer[key]

            def __setitem__(self, key, value):
                """Delegate dictionary setting"""
                self._wrapped_layer[key] = value

        return SimpleWrapper(layer, name)

    def _restore_model_layers(self):
        """Restore all patched modules to their original state"""
        if self.interpretable_model is None:
            return

        model = self.interpretable_model.model

        for name, info in self._patched_modules.items():
            parent_or_list, attr_or_idx, original_module = info
            try:
                if isinstance(attr_or_idx, int):
                    # It's a list element - re-access the REAL list
                    parts = name.rsplit('.', 1)
                    if len(parts) == 2:
                        parent_path, _ = parts
                        path_parts = parent_path.split('.')
                        if len(path_parts) >= 2:
                            attr_name = path_parts[-1]

                            # Navigate to parent object (NOT including the list attribute itself)
                            # For "model.model.layers", navigate to "model.model", then get "layers"
                            parent_obj = model
                            for part in path_parts[1:-1]:  # Skip "model" prefix AND the list attr
                                parent_obj = getattr(parent_obj, part, None)
                                if parent_obj is None:
                                    break

                            if parent_obj is not None:
                                # Get the REAL list via attribute access
                                real_list = getattr(parent_obj, attr_name, None)
                                if real_list is not None and isinstance(real_list, list):
                                    real_list[attr_or_idx] = original_module
                                    continue

                    # Fallback
                    parent_or_list[attr_or_idx] = original_module
                else:
                    # It's a direct attribute
                    setattr(parent_or_list, attr_or_idx, original_module)
            except Exception:
                pass

        # Clear tracking dictionaries
        self._patched_modules = {}
        self._module_map = {}

    def _build_forward_call(self) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """
        Build the positional/keyword call used for the traced forward pass.

        Supported modes:
        - legacy positional input via ``inputs`` (text, tokens, arrays)
        - prepared mapping via ``inputs`` when it is a dict
        - direct model kwargs via ``model_inputs``
        """
        if self.inputs is not None and self.model_inputs:
            raise ValueError(
                "Pass either 'inputs' or keyword model inputs to trace(), not both."
            )

        if isinstance(self.inputs, dict):
            return (), self._adapt_model_inputs(dict(self.inputs))

        if self.model_inputs:
            return (), self._adapt_model_inputs(dict(self.model_inputs))

        if self.inputs is None:
            raise ValueError("trace() requires inputs or keyword model inputs.")

        return (self._process_inputs(self.inputs),), {}

    def _adapt_model_inputs(self, model_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply lightweight compatibility fixes to kwargs-style model inputs.

        Some processors emit ``attention_mask`` while the wrapped model's
        forward signature expects ``mask`` (common for mlx-vlm models).
        """
        if (
            "attention_mask" in model_inputs
            and "mask" not in model_inputs
            and self._model_accepts_argument("mask")
            and not self._model_accepts_argument("attention_mask")
        ):
            model_inputs["mask"] = model_inputs.pop("attention_mask")

        return model_inputs

    def _model_accepts_argument(self, argument_name: str) -> bool:
        """Check whether the wrapped model exposes a named ``__call__`` argument."""
        if self.interpretable_model is None:
            return False

        try:
            signature = inspect.signature(self.interpretable_model.model.__call__)
        except (TypeError, ValueError):
            return False

        return argument_name in signature.parameters

    def _normalize_output(self, output: Any) -> Any:
        """
        Normalize model outputs for interpretability workflows.

        If a model returns an object with ``.logits`` (common for mlx-vlm
        models), expose the logits tensor as ``Trace.output``. Tuple outputs
        are unwrapped to their primary tensor. The unmodified value is always
        preserved in ``Trace.raw_output``.
        """
        logits = getattr(output, "logits", None)
        if logits is not None:
            return logits

        if isinstance(output, (tuple, list)) and output:
            try:
                return get_primary_tensor(output)
            except TypeError:
                return output

        return output

    def _process_inputs(self, inputs: Any) -> mx.array:
        """
        Process positional inputs into the format expected by the model.

        Handles:
        - String inputs (tokenize if tokenizer available)
        - mx.array inputs (pass through)
        - List inputs (batch)
        """
        if isinstance(inputs, str):
            if self.tokenizer is None:
                raise ValueError(
                    "String input provided but no tokenizer available. "
                    "Pass tokenizer to InterpretableModel or use token arrays."
                )
            # Tokenize
            tokens = self.tokenizer.encode(inputs)
            return mx.array([tokens])

        elif isinstance(inputs, (list, tuple)) and all(isinstance(x, str) for x in inputs):
            # Batch of strings
            if self.tokenizer is None:
                raise ValueError("String inputs require a tokenizer")

            token_lists = [self.tokenizer.encode(text) for text in inputs]
            # Pad to same length
            max_len = max(len(t) for t in token_lists)
            padded = [t + [0] * (max_len - len(t)) for t in token_lists]
            return mx.array(padded)

        elif isinstance(inputs, mx.array):
            # Ensure batch dimension exists
            if inputs.ndim == 1:
                return inputs.reshape(1, -1)
            return inputs

        elif isinstance(inputs, (list, tuple)):
            # Assume it's already tokenized - check if it's a list of ints (single sequence)
            # or a list of lists (batch of sequences)
            if inputs and all(isinstance(x, int) for x in inputs):
                # Single sequence of token IDs - wrap in batch dimension
                return mx.array([inputs])
            else:
                # Batch of sequences
                return mx.array(inputs)

        else:
            raise ValueError(f"Unsupported input type: {type(inputs)}")

    def get(self, name: str) -> Optional[Any]:
        """
        Get a saved value by name.

        Accepts shorthand names: "layers.3.attn.output" also matches values
        saved under wrapped keys like "model.language_model.model.layers.3.attn.output".

        Args:
            name: The name of the saved value (e.g., "layers.3.attn.output")

        Returns:
            The saved value, or None if not found
        """
        matched_name = _find_matching_name(self.saved_values, name)
        if matched_name is not None:
            return self.saved_values[matched_name]
        return None

    def get_activation(self, name: str) -> Optional[Any]:
        """
        Get an activation by module name.

        Accepts shorthand names: "layers.3.attn" also matches activations
        stored under wrapped keys like "model.language_model.model.layers.3.attn".

        Args:
            name: The module name (e.g., "layers.3.attn")

        Returns:
            The activation, or None if not found
        """
        matched_name = _find_matching_name(self.activations, name)
        if matched_name is not None:
            return self.activations[matched_name]
        return None

    def __repr__(self):
        n_saved = len(self.saved_values)
        n_activations = len(self.activations)
        return f"Trace(saved_values={n_saved}, activations={n_activations})"
