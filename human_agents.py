"""Advisers for the human experiment: the Sonar Operator, the Track Analyst and the Periscope Operator answer the human only when asked.

They are normal Mahilo agents (LockedAgent), but:
  - they cannot talk to other agents (chat_with_agent is removed),
  - their instructions come from prompts/<agent>.txt + prompts/reference.txt (edit these files, not the code),
  - the human's question and any released data go straight into their own session history.
"""
import hashlib
import os
from pathlib import Path

from agents import LockedAgent, SONAR
from agents_v2 import TA

PROMPT_DIR = Path(os.getenv("PROMPT_DIR", "prompts"))
PERI = "periscope_operator"
LABELS = {SONAR: "Sonar Operator", TA: "Track Analyst", PERI: "Periscope Operator"}


def reference_text() -> str:
    return (PROMPT_DIR / "reference.txt").read_text(encoding="utf-8").strip()


def agent_prompt(name: str) -> str:
    return (PROMPT_DIR / f"{name}.txt").read_text(encoding="utf-8").strip() + "\n" + reference_text()


def prompt_version() -> str:
    """Short fingerprint of all prompt files, written to the log of every session."""
    digest = hashlib.sha256()
    for name in (SONAR, TA, PERI, "reference"):
        digest.update((PROMPT_DIR / f"{name}.txt").read_bytes())
    return digest.hexdigest()[:10]


class _Capture:
    """Stands in for a websocket: if the model calls contact_human anyway, its text is kept, not lost."""

    def __init__(self):
        self.parts = []

    async def send_text(self, text):
        self.parts.append(str(text))


class AdviserAgent(LockedAgent):
    @property
    def tools(self):
        return [t for t in super().tools if t["function"]["name"] != "chat_with_agent"]

    def prompt_message(self) -> str:
        return self.description                    # plain instructions: no other agents, no contact_human

    def add_data(self, block: str) -> None:
        """Put a released data block into this agent's history without a model call."""
        self._session.add_message(block, "user")

    def holds(self, block: str) -> bool:
        """True if this agent's history really contains the block (used to confirm every data step arrived)."""
        return any(isinstance(m, dict) and m.get("content") == block for m in getattr(self._session, "messages", []))

    async def ask(self, question: str) -> str:
        capture = _Capture()
        result = await self.process_chat_message(question, websockets=[capture])
        text = str((result or {}).get("response") or "").strip()
        return text or "\n".join(capture.parts).strip()


def create_advisers():
    return [
        AdviserAgent(type=SONAR, name=SONAR, short_description="Gives the sonar cues when asked.",
                     description=agent_prompt(SONAR), can_contact=[]),
        AdviserAgent(type=TA, name=TA, short_description="Gives own position and the tracks' positions, ranges and speeds when asked.",
                     description=agent_prompt(TA), can_contact=[]),
        AdviserAgent(type=PERI, name=PERI, short_description="Gives the periscope reports when asked.",
                     description=agent_prompt(PERI), can_contact=[]),
    ]