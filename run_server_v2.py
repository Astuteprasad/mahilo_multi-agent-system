"""Start the three version 2 agents and the feeder.   python run_server_v2.py

Set OPENAI_API_KEY (and MAHILO_LLM_MODEL, for example gpt-6-luna) in this terminal first.
Never put the key in a file you share."""
import asyncio
import os

from mahilo import AgentManager
from mahilo.server import ServerManager

import eventlog
from agents import limit_messages
from agents_v2 import create_agents
from feeder_v2 import run_feed


class FeedServer(ServerManager):
    """The normal Mahilo server that also starts the feeder when it starts up."""

    def __init__(self, manager):
        super().__init__(manager)
        self.app.on_event("startup")(self._start_feed)

    def sockets_for(self, name):
        return list(self.websocket_connections.get(name, {}).values())

    async def _start_feed(self):
        asyncio.create_task(run_feed(self.agent_manager, self.sockets_for))


def pin_model():
    """Two changes to how Mahilo calls the model:
    1. Mahilo quietly falls back to openai/gpt-4o-mini whenever a call to the chosen model fails. For
       this test that would hide a wrong model name, so the fallback is made the same model (an
       error then shows up instead).
    3. Reasoning models sometimes return an empty answer (no text, no tool call). That is asked again
       (up to MAHILO_EMPTY_RETRIES times, default 5) so that no step is lost.
    2. Mahilo asks for temperature 0.7, which reasoning models such as gpt-6-luna refuse (they accept
       only 1). Every call is sent with MAHILO_TEMPERATURE instead (default 1.0)."""
    from mahilo.llm_config import llm_config
    llm_config.default_model = llm_config.model
    temperature = float(os.getenv("MAHILO_TEMPERATURE", "1.0"))
    original = llm_config.chat_completion

    retries = int(os.getenv("MAHILO_EMPTY_RETRIES", "5"))

    async def chat(*args, **kwargs):
        kwargs["temperature"] = temperature
        for attempt in range(retries + 1):
            response = await original(*args, **kwargs)
            message = response.choices[0].message
            if (getattr(message, "content", None) or "").strip() or getattr(message, "tool_calls", None):
                return response
            # An empty answer (no text and no tool call) would make an operator drop its step: ask again.
            eventlog.log("empty_retry", agent=eventlog.current_agent.get(), text=f"empty answer, try {attempt + 1}")
        return response

    llm_config.chat_completion = chat


def main():
    pin_model()
    eventlog.reset()
    manager = AgentManager(db_path="submarine_messages_v2.db")
    eventlog.install(manager)
    limit_messages(manager)
    for agent in create_agents():
        manager.register_agent(agent)
        agent.activate()
    print("Model:", os.getenv("MAHILO_LLM_MODEL", "(Mahilo default)"))
    print("Agents:", ", ".join(a.name for a in manager.get_all_agents()))
    for agent in manager.get_all_agents():
        print(f"    mahilo connect --agent-name {agent.name}")
    FeedServer(manager).run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    main()
