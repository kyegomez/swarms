"""
Per-tool permission policy for the autonomous loop.

The ``max_loops="auto"`` loop ships side-effecting tools - ``create_file``,
``update_file``, ``delete_file``, ``run_bash`` - that execute unconditionally.
The only guard is a keyword blocklist on ``run_bash``
(:func:`~swarms.structs.autonomous_loop_utils._check_bash_command`), which both
over-blocks benign commands and is bypassed by trivial rewrites. There is no
way for a caller to say "this agent may read and grep, but must ask before it
writes, and must never delete".

This module is that missing layer. It is a *policy*, not a sandbox: it decides
whether a call runs, and records the decision. Path confinement is a separate
concern and lives with the file tools.

Nothing here changes behavior unless a policy is configured. A denial is
returned to the model as an ordinary tool result rather than raised, so the
loop carries on and the model can choose another route - the same contract
tool errors already have.

Example:
    >>> policy = ToolPermissionPolicy(
    ...     tool_permissions={
    ...         "read_file": "allow",
    ...         "delete_file": "deny",
    ...         "run_bash": lambda name, args: (
    ...             "allow" if args.get("command", "").startswith("pytest")
    ...             else "deny"
    ...         ),
    ...     },
    ...     default_permission="ask",
    ...     permission_callback=lambda name, args: input(f"{name}? ") == "y",
    ... )
    >>> policy.decide("delete_file", {"file_path": "notes.md"}).allowed
    False
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from loguru import logger

ALLOW = "allow"
DENY = "deny"
ASK = "ask"

_DEFAULT_VALUES = (ALLOW, DENY, ASK)
_POLICY_VALUES = (ALLOW, DENY, ASK)

# Value returned to the model when a call is refused. Prefixed so the model can
# recognise a policy refusal rather than reading it as a tool failure it should
# retry with different arguments.
_DENIAL_TEMPLATE = (
    "Permission denied for {tool_name}: {reason} "
    "Do not retry this call; choose a different approach or ask the user."
)

PermissionValue = Union[str, Callable[[str, Dict[str, Any]], str]]


@dataclass(frozen=True)
class PermissionDecision:
    """
    One allow/deny ruling, kept so a run can be audited afterwards.

    Attributes:
        tool_name: The tool the model asked for.
        arguments: The arguments it asked for, as the loop received them.
        decision: ``"allow"`` or ``"deny"`` - the resolved outcome, never
            ``"ask"``, which is a route rather than a result.
        reason: Why, in the words the model is shown on a denial.
        source: Which rule decided - ``"policy"`` (an entry in
            ``tool_permissions``), ``"callable"``, ``"callback"``,
            ``"default"``, or ``"no-callback"`` when ``ask`` degraded to deny.
    """

    tool_name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    decision: str = ALLOW
    reason: str = ""
    source: str = "default"

    @property
    def allowed(self) -> bool:
        """True when the call may run."""
        return self.decision == ALLOW


class ToolPermissionPolicy:
    """
    Resolves whether a tool call may run, and records every ruling.

    Args:
        tool_permissions: Per-tool rules, keyed by tool name. A value is
            ``"allow"``, ``"deny"``, ``"ask"``, or a callable
            ``(tool_name, arguments) -> "allow" | "deny" | "ask"``. The
            callable form is what lets a caller express "allow ``run_bash``
            when the command starts with ``pytest``" without the framework
            guessing at a blocklist. ``None`` means no per-tool rules.
        default_permission: Applied to any tool with no rule of its own.
            Defaults to ``"allow"``, which is exactly today's behavior.
        permission_callback: Called for ``"ask"`` as
            ``(tool_name, arguments) -> bool``. With no callback, ``"ask"``
            degrades to deny and the model is told why, because a loop that
            blocks on a prompt nobody is watching is worse than one that
            refuses and moves on.

    Raises:
        ValueError: If ``default_permission`` or any policy value is not one
            of ``"allow"``, ``"deny"``, ``"ask"`` or a callable.

    Notes:
        A callable that raises is treated as a denial, not as a crash: the
        loop must not die because a caller's policy has a bug in it. The
        exception text becomes the reason the model is shown.
    """

    def __init__(
        self,
        tool_permissions: Optional[Dict[str, PermissionValue]] = None,
        default_permission: str = ALLOW,
        permission_callback: Optional[
            Callable[[str, Dict[str, Any]], bool]
        ] = None,
    ) -> None:
        if default_permission not in _DEFAULT_VALUES:
            raise ValueError(
                f"default_permission must be one of "
                f"{_DEFAULT_VALUES}, got {default_permission!r}"
            )

        self.tool_permissions: Dict[str, PermissionValue] = dict(
            tool_permissions or {}
        )
        for name, value in self.tool_permissions.items():
            if callable(value):
                continue
            if value not in _POLICY_VALUES:
                raise ValueError(
                    f"tool_permissions[{name!r}] must be one of "
                    f"{_POLICY_VALUES} or a callable, got {value!r}"
                )

        self.default_permission = default_permission
        self.permission_callback = permission_callback
        self.log: List[PermissionDecision] = []

    @property
    def is_permissive(self) -> bool:
        """
        True when this policy can never refuse anything.

        The loop uses it to skip wrapping its handlers entirely, so an agent
        that configured nothing runs the exact code path it ran before.
        """
        return (
            not self.tool_permissions
            and self.default_permission == ALLOW
        )

    def decide(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> PermissionDecision:
        """
        Rule on one call and record the ruling.

        Args:
            tool_name: The tool the model asked for.
            arguments: The arguments it asked for.

        Returns:
            The :class:`PermissionDecision`, already appended to ``log``.
        """
        arguments = dict(arguments or {})
        decision = self._resolve(tool_name, arguments)
        self.log.append(decision)

        if not decision.allowed:
            logger.warning(
                f"Tool permission denied: {tool_name} "
                f"({decision.source}) - {decision.reason}"
            )
        return decision

    def _resolve(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> PermissionDecision:
        """Resolve a rule to allow/deny without touching ``log``."""
        rule = self.tool_permissions.get(tool_name)
        source = "policy"

        if rule is None:
            rule = self.default_permission
            source = "default"
        elif callable(rule):
            source = "callable"
            try:
                rule = rule(tool_name, arguments)
            except Exception as e:
                return PermissionDecision(
                    tool_name=tool_name,
                    arguments=arguments,
                    decision=DENY,
                    reason=f"the permission rule raised {type(e).__name__}: {e}.",
                    source=source,
                )
            if rule not in _POLICY_VALUES:
                return PermissionDecision(
                    tool_name=tool_name,
                    arguments=arguments,
                    decision=DENY,
                    reason=(
                        f"the permission rule returned {rule!r}, which is "
                        f"not one of {_POLICY_VALUES}."
                    ),
                    source=source,
                )

        if rule == ALLOW:
            return PermissionDecision(
                tool_name=tool_name,
                arguments=arguments,
                decision=ALLOW,
                reason="",
                source=source,
            )

        if rule == DENY:
            return PermissionDecision(
                tool_name=tool_name,
                arguments=arguments,
                decision=DENY,
                reason="the configured policy denies this tool.",
                source=source,
            )

        # rule == ASK
        if self.permission_callback is None:
            return PermissionDecision(
                tool_name=tool_name,
                arguments=arguments,
                decision=DENY,
                reason=(
                    "this tool requires approval and no permission_callback "
                    "was configured, so it cannot be approved."
                ),
                source="no-callback",
            )

        try:
            approved = bool(
                self.permission_callback(tool_name, arguments)
            )
        except Exception as e:
            return PermissionDecision(
                tool_name=tool_name,
                arguments=arguments,
                decision=DENY,
                reason=f"the permission callback raised {type(e).__name__}: {e}.",
                source="callback",
            )

        return PermissionDecision(
            tool_name=tool_name,
            arguments=arguments,
            decision=ALLOW if approved else DENY,
            reason="" if approved else "the user declined this call.",
            source="callback",
        )

    def denial_message(self, decision: PermissionDecision) -> str:
        """
        The tool result a refused call returns to the model.

        Args:
            decision: A denied :class:`PermissionDecision`.

        Returns:
            The refusal text, phrased so the model does not retry the same
            call with cosmetically different arguments.
        """
        return _DENIAL_TEMPLATE.format(
            tool_name=decision.tool_name, reason=decision.reason
        )

    def guard(
        self, tool_name: str, handler: Callable[..., Any]
    ) -> Callable[..., Any]:
        """
        Wrap one tool handler so it is checked before it runs.

        Args:
            tool_name: The name the model calls this handler by.
            handler: The handler, taking keyword arguments only - which is how
                the loop dispatches every built-in tool.

        Returns:
            A handler with the same signature that returns the denial string
            instead of executing when the policy refuses.
        """

        def _guarded(**kwargs: Any) -> Any:
            decision = self.decide(tool_name, kwargs)
            if not decision.allowed:
                return self.denial_message(decision)
            return handler(**kwargs)

        return _guarded
