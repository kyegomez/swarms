"""Build a Pydantic model from a class's ``__init__`` signature.

The signature of a class is already a schema — parameter names, type
annotations, which arguments are required, and (in this codebase) a Google-style
``Args:`` block describing each one. This module turns that into a
``BaseModel`` so it can be used for validation, structured LLM output, or an
OpenAI function-calling schema, without hand-maintaining a parallel definition
that drifts from the constructor.

Example:

    from swarms import Agent
    from swarms.utils.class_to_pydantic import class_init_to_pydantic_model

    AgentSchema = class_init_to_pydantic_model(
        Agent, include=("agent_name", "system_prompt", "model_name")
    )

    AgentSchema(agent_name="Analyst", system_prompt="...", model_name="gpt-5.4")
"""

import inspect
from typing import (
    Any,
    Dict,
    Optional,
    Sequence,
    Tuple,
    Type,
    get_type_hints,
)

from pydantic import BaseModel, ConfigDict, Field, create_model

from swarms.utils.docstring_parser import parse
from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger(log_folder="class_to_pydantic")


class ArbitraryTypesModel(BaseModel):
    """Default base allowing fields typed with non-Pydantic classes.

    Constructors in this codebase routinely accept arbitrary objects — an
    ``Agent``, a ``Callable``, a thread pool. Pydantic cannot build a core
    schema for those, so without this the generated model raises
    ``PydanticSchemaGenerationError`` for most real classes. Such fields are
    validated by ``isinstance`` rather than by a full schema.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)


# Mutable defaults must go through default_factory — sharing one instance
# across every model instance is the classic Python footgun, and Pydantic
# rejects them outright.
_MUTABLE_TYPES = (list, dict, set, bytearray)


def _param_descriptions(cls: type) -> Dict[str, str]:
    """Collect per-parameter descriptions from the class and its ``__init__``.

    Both docstrings are read because the convention in this codebase is to
    document constructor arguments in ``__init__`` while the class docstring
    carries an ``Attributes:`` block. ``__init__`` wins on conflict, being the
    more specific of the two.

    Args:
        cls (type): The class to read docstrings from.

    Returns:
        Dict[str, str]: Mapping of parameter name to description. Empty when
        neither docstring documents anything.
    """
    descriptions: Dict[str, str] = {}
    for source in (cls, cls.__init__):
        doc = inspect.getdoc(source)
        if not doc:
            continue
        try:
            for param in parse(doc).params:
                descriptions[param.arg_name] = param.description
        except (
            Exception
        ) as e:  # a malformed docstring must not be fatal
            logger.debug(
                f"Could not parse docstring for {cls!r}: {e}"
            )
    return descriptions


def _resolve_hints(cls: type) -> Dict[str, Any]:
    """Resolve ``__init__``'s type hints, tolerating unresolvable forward refs.

    ``get_type_hints`` evaluates string annotations, which raises if a name is
    not importable at runtime (a ``TYPE_CHECKING``-only import, for instance).
    Falling back to the raw ``__annotations__`` keeps the rest of the signature
    usable instead of failing the whole model.

    Args:
        cls (type): The class whose ``__init__`` hints to resolve.

    Returns:
        Dict[str, Any]: Parameter name to resolved type.
    """
    try:
        return get_type_hints(cls.__init__)
    except Exception as e:
        logger.debug(
            f"Falling back to raw annotations for {cls.__name__}: {e}"
        )
        return dict(getattr(cls.__init__, "__annotations__", {}))


def _field_for(
    param: inspect.Parameter,
    annotation: Any,
    description: Optional[str],
) -> Tuple[Any, Any]:
    """Build the ``(type, FieldInfo)`` pair for one constructor parameter.

    Args:
        param (inspect.Parameter): The parameter being converted.
        annotation (Any): Resolved type for the parameter.
        description (Optional[str]): Description pulled from the docstring.

    Returns:
        Tuple[Any, Any]: A ``create_model``-compatible field definition.
    """
    if param.default is inspect.Parameter.empty:
        return annotation, Field(..., description=description)

    default = param.default
    if isinstance(default, _MUTABLE_TYPES):
        # Bind by value so each model instance gets its own copy.
        return annotation, Field(
            default_factory=lambda d=default: type(d)(d),
            description=description,
        )

    return annotation, Field(default, description=description)


def class_init_to_pydantic_model(
    cls: type,
    model_name: Optional[str] = None,
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
    require_annotations: bool = False,
    base: Type[BaseModel] = ArbitraryTypesModel,
) -> Type[BaseModel]:
    """Build a Pydantic model mirroring a class's ``__init__`` parameters.

    Each constructor parameter becomes a field: parameters with defaults are
    optional and keep that default, parameters without one are required.
    Descriptions are lifted from the class or ``__init__`` docstring so the
    generated model carries the same documentation as the constructor.

    ``self``, ``*args`` and ``**kwargs`` are always skipped — they are not
    nameable fields.

    Args:
        cls (type): The class whose ``__init__`` signature to mirror.
        model_name (Optional[str]): Name for the generated model. Defaults to
            ``f"{cls.__name__}Schema"``, derived from the class name.
        include (Optional[Sequence[str]]): If given, only these parameters
            become fields. Names that do not exist on the signature are ignored.
        exclude (Optional[Sequence[str]]): Parameters to omit. Applied after
            ``include``.
        require_annotations (bool): When True, skip parameters that carry no
            type annotation. When False (default) they are typed ``Any``.
        base (Type[BaseModel]): Base class for the generated model, for
            attaching shared config or validators. Defaults to
            :class:`ArbitraryTypesModel`, which permits fields typed with
            non-Pydantic classes such as ``Agent``. Pass a plain ``BaseModel``
            subclass to require every field be fully schema-able — that is
            stricter, but raises on most real constructors.

    Returns:
        Type[BaseModel]: The generated model class.

    Raises:
        TypeError: If ``cls`` is not a class.
        ValueError: If no usable parameters remain after filtering.

    Example:
        >>> class Widget:
        ...     def __init__(self, name: str, size: int = 3):
        ...         '''Build a widget.
        ...
        ...         Args:
        ...             name (str): What to call it.
        ...             size (int): How big it is.
        ...         '''
        >>> Model = class_init_to_pydantic_model(Widget)
        >>> Model.__name__
        'WidgetSchema'
        >>> Model(name="bolt").size
        3
    """
    if not inspect.isclass(cls):
        raise TypeError(
            f"Expected a class, got {type(cls).__name__}: {cls!r}"
        )

    try:
        signature = inspect.signature(cls.__init__)
    except (ValueError, TypeError) as e:
        raise TypeError(
            f"Could not read the signature of {cls.__name__}.__init__: {e}"
        ) from e

    hints = _resolve_hints(cls)
    descriptions = _param_descriptions(cls)

    include_set = set(include) if include is not None else None
    exclude_set = set(exclude or ())

    fields: Dict[str, Tuple[Any, Any]] = {}
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        # *args / **kwargs have no fixed name or arity, so they cannot map to
        # a field. Skipping them silently matches how the signature reads.
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if include_set is not None and name not in include_set:
            continue
        if name in exclude_set:
            continue

        annotation = hints.get(name, param.annotation)
        if annotation is inspect.Parameter.empty:
            if require_annotations:
                logger.debug(
                    f"Skipping unannotated parameter {name!r} on {cls.__name__}"
                )
                continue
            annotation = Any

        fields[name] = _field_for(
            param, annotation, descriptions.get(name)
        )

    if not fields:
        raise ValueError(
            f"No usable __init__ parameters found on {cls.__name__}. "
            "Check the include/exclude filters and require_annotations."
        )

    return create_model(
        model_name or f"{cls.__name__}Schema",
        __base__=base,
        **fields,
    )
