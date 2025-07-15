import os
import asyncio
import nest_asyncio
import json
from dotenv import load_dotenv
from typing import List, Dict, Any

from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import StructuredTool
from pydantic_model_schema import create_pydantic_model_from_schema

# Apply nested event loop patch
nest_asyncio.apply()
load_dotenv()


class MCPChatBotLangChain:
    def __init__(self, model_name="gemma2-9b-it"):
        self.model_name = model_name
        self.chat_groq = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model=model_name,
            temperature=0.1,
            max_tokens=2048,
        )
        self.available_tools: List[Any] = []
        self.agent_executor: AgentExecutor = None
        self.conversation_history: List[Any] = []

        self.exit_stack = AsyncExitStack()
        self.sessions: List[ClientSession] = []
        self.tool_to_session: Dict[str, ClientSession] = {}

    def _convert_mcp_tool_to_langchain_tool(self, tool_def: Dict[str, Any]):
        name = tool_def["name"]
        description = tool_def["description"]
        input_schema = tool_def.get("input_schema", {})

        ToolInputModel = create_pydantic_model_from_schema(f"{name}Input", input_schema)

        async def _tool_func(**kwargs) -> str:
            try:
                session = self.tool_to_session[name]
                result = await session.call_tool(name, arguments=kwargs)

                # Force flattening into string
                content = result.content
                if isinstance(content, list):
                    # Extract any 'text' fields (Anthropic-style content chunks)
                    content = "\n".join(
                        part.text if hasattr(part, "text") else str(part)
                        for part in content
                    )
                elif hasattr(content, "text"):
                    content = content.text
                else:
                    content = str(content)

                return content

            except Exception as e:
                return f"Error calling tool {name}: {str(e)}"


        return StructuredTool.from_function(
            name=name,
            description=description,
            func=_tool_func,
            coroutine=_tool_func,
            args_schema=ToolInputModel
        )

    def _setup_agent(self):

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use tools when appropriate."),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])

        agent = create_tool_calling_agent(
            llm=self.chat_groq,
            tools=self.available_tools,
            prompt=prompt
        )

        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.available_tools,
            verbose=True,
            handle_parsing_errors=True
        )

    async def process_query(self, query: str):
        try:
            response = await self.agent_executor.ainvoke({
                "input": query,
                "chat_history": self.conversation_history
            })

            self.conversation_history.append(HumanMessage(content=query))
            self.conversation_history.append(AIMessage(content=response["output"]))

            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]

            print(response["output"])
            return response["output"]
        except Exception as e:
            print(f"❌ Error during query: {e}")
            return f"Error: {e}"

    async def chat_loop(self):
        print("\n🧠 MCP Chatbot (LangChain + ChatGroq) Started!")
        print("Type your questions, or 'quit' to exit.")

        while True:
            query = input("\nQuery: ").strip()
            if query.lower() == "quit":
                break
            await self.process_query(query)

    async def connect_to_server(self, server_name: str, server_config: dict):
        try:
            server_params = StdioServerParameters(**server_config)
            read, write = await self.exit_stack.enter_async_context(stdio_client(server_params))
            session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self.sessions.append(session)

            response = await session.list_tools()
            for tool in response.tools:
                self.tool_to_session[tool.name] = session
                self.available_tools.append(
                    self._convert_mcp_tool_to_langchain_tool({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema
                    })
                )
            print(f"✅ Connected to {server_name} with tools:", [tool.name for tool in response.tools])
        except Exception as e:
            print(f"❌ Failed to connect to {server_name}: {e}")

    async def connect_to_servers(self):
        try:
            with open("server_config.json", "r") as f:
                config = json.load(f)
                servers = config.get("mcpServers", {})
                for name, conf in servers.items():
                    await self.connect_to_server(name, conf)
        except Exception as e:
            print(f"❌ Failed to load server configuration: {e}")
            raise

    async def cleanup(self):
        await self.exit_stack.aclose()


async def main():
    chatbot = MCPChatBotLangChain()
    try:
        await chatbot.connect_to_servers()
        chatbot._setup_agent()
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
