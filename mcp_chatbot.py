import os
import asyncio
import nest_asyncio
from dotenv import load_dotenv
from typing import List, Dict, Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain.tools import StructuredTool
from pydantic_model_schema import create_pydantic_model_from_schema

# Apply nested event loop patch for Jupyter/async environments
nest_asyncio.apply()
load_dotenv()

class MCPChatBotLangChain:
    def __init__(self, model_name="gemma2-9b-it"):
        self.session: ClientSession = None
        self.model_name = model_name
        self.chat_groq = ChatGroq(
            model=model_name,
            temperature=0.1,
            max_tokens=2048
        )
        self.available_tools: List[Any] = []
        self.agent_executor: AgentExecutor = None
        self.conversation_history: List[Any] = []

    def _convert_mcp_tool_to_langchain_tool(self, tool_def: Dict[str, Any]):
        name = tool_def["name"]
        description = tool_def["description"]
        input_schema = tool_def.get("input_schema", {})

        # Create dynamic model from input schema
        ToolInputModel = create_pydantic_model_from_schema(f"{name}Input", input_schema)

        async def _tool_func(**kwargs) -> str:
            try:
                arguments = kwargs  # Already structured
                result = await self.session.call_tool(name, arguments=arguments)
                return str(result.content)
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

            # Limit history length
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]

            print(response["output"])
            return response["output"]

        except Exception as e:
            print(f"Error during query: {e}")
            return f"Error: {e}"

    async def chat_loop(self):
        print("\n🧠 MCP Chatbot (LangChain + ChatGroq) Started!")
        print("Type your questions, or 'quit' to exit.")

        while True:
            query = input("\nQuery: ").strip()
            if query.lower() == "quit":
                break
            await self.process_query(query)

    async def connect_to_server_and_run(self):
        server_params = StdioServerParameters(
            command="uv",
            args=["run", "research_server.py"]
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                self.session = session
                await session.initialize()

                # List available tools from MCP server
                response = await session.list_tools()
                mcp_tools = [{
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.inputSchema
                } for t in response.tools]

                print("Connected to MCP server with tools:", [t["name"] for t in mcp_tools])

                # Convert each to LangChain tool
                for tool_def in mcp_tools:
                    langchain_tool = self._convert_mcp_tool_to_langchain_tool(tool_def)
                    self.available_tools.append(langchain_tool)

                # Initialize the agent executor
                self._setup_agent()

                # Start interaction loop
                await self.chat_loop()


async def main():
    chatbot = MCPChatBotLangChain(
        model_name="gemma2-9b-it"  # Or any other Groq-supported model
    )
    await chatbot.connect_to_server_and_run()

if __name__ == "__main__":
    asyncio.run(main())
