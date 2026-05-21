"""Command interpreter abstractions for text-controlled analysis."""

from __future__ import annotations

from typing import Protocol

from multimodalcv.commands.parser import CommandIntent, UnsupportedCommandError, interpret_command


class CommandInterpreter(Protocol):
    """Common interface for deterministic and model-backed command interpreters."""

    def interpret(self, command: str, *, zone_name: str = "main") -> CommandIntent:
        """Return a structured intent for a user command."""


class DeterministicCommandInterpreter:
    """Interpreter backed by the restricted local parser."""

    def interpret(self, command: str, *, zone_name: str = "main") -> CommandIntent:
        return interpret_command(command, zone_name=zone_name)


class FallbackCommandInterpreter:
    """Try interpreters in order until one understands the command."""

    def __init__(self, interpreters: tuple[CommandInterpreter, ...]) -> None:
        if not interpreters:
            raise ValueError("at least one command interpreter is required")
        self._interpreters = interpreters

    def interpret(self, command: str, *, zone_name: str = "main") -> CommandIntent:
        last_error: UnsupportedCommandError | None = None

        for interpreter in self._interpreters:
            try:
                return interpreter.interpret(command, zone_name=zone_name)
            except UnsupportedCommandError as error:
                last_error = error

        if last_error is not None:
            raise last_error

        raise UnsupportedCommandError(f"Команда пока не поддерживается: {command}")


def default_command_interpreter() -> CommandInterpreter:
    """Return the current default command interpreter."""
    return DeterministicCommandInterpreter()
