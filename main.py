import os
import json
import asyncio
from datetime import datetime
from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Depends
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Dict, List, Optional, Union
from uuid import UUID, uuid4
from dotenv import load_dotenv
import logging

from autogen_ext.tools.mcp import StdioServerParams, SseServerParams, StreamableHttpServerParams, McpWorkbench
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage, UserMessage as AutoGenUserMessage, ToolCallRequestEvent, ToolCallExecutionEvent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_core import CancellationToken

from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# --- Basic Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Database Setup ---
DATABASE_URL = "sqlite:///./chat_history.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(String, primary_key=True, index=True)
    title = Column(String, nullable=False)
    user_id = Column(String)
    timestamp = Column(DateTime, default=datetime.now)

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    sender = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    thought_process = Column(Text)
    file_path = Column(String)

def init_db():
    logging.info("Initializing database...")
    try:
        Base.metadata.create_all(bind=engine)
        logging.info("Database initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing database: {e}", exc_info=True)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Pydantic Models ---
class McpServerConfig(BaseModel):
    type: str
    command: Optional[str] = None
    args: Optional[List[str]] = []
    env: Optional[Dict[str, str]] = {}
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    timeout: Optional[float] = 30.0
    sse_read_timeout: Optional[float] = 300.0
    terminate_on_close: Optional[bool] = True
    read_timeout_seconds: Optional[float] = 60.0

class Settings(BaseModel):
    user_id: Optional[str] = "default-user"
    password: Optional[str] = None
    theme: Optional[str] = "dark"
    system_prompt: Optional[str] = "You are a helpful AI assistant. When you have completed your response and there are no further questions, your FINAL message MUST be the exact string 'TERMINATE' and nothing else. Do NOT include 'TERMINATE' in any other part of your responses."
    mcpServers: Optional[Dict[str, McpServerConfig]] = {}

class ChatRequest(BaseModel):
    message: str
    chat_id: Optional[UUID] = None

# --- Settings Management ---
SETTINGS_FILE = "./settings.json"

def load_settings() -> Settings:
    if not os.path.exists(SETTINGS_FILE):
        return Settings()
    with open(SETTINGS_FILE, "r") as f:
        return Settings(**json.load(f))

def save_settings(settings: Settings):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings.model_dump(exclude_unset=True), f, indent=4)

# --- Global Variables ---
agents: Dict[str, AssistantAgent] = {}
mcp_workbenches: Dict[str, McpWorkbench] = {}
assistant_initialized_event = asyncio.Event()

# --- FastAPI App ---
app = FastAPI()

# --- FastAPI Events ---
@app.on_event("startup")
async def startup_event():
    init_db()
    global agent, mcp_workbenches
    settings = load_settings()

    # Initialize MCP Workbenches
    for name, config in settings.mcpServers.items():
        try:
            params: Union[StdioServerParams, SseServerParams, StreamableHttpServerParams]
            if config.type == "stdio":
                params = StdioServerParams(command=config.command, args=config.args, env=config.env, read_timeout_seconds=config.read_timeout_seconds)
            elif config.type == "sse":
                params = SseServerParams(url=config.url, headers=config.headers, timeout=config.timeout, sse_read_timeout=config.sse_read_timeout)
            elif config.type == "streamable_http":
                params = StreamableHttpServerParams(url=config.url, headers=config.headers, timeout=config.timeout, sse_read_timeout=config.sse_read_timeout, terminate_on_close=config.terminate_on_close)
            else:
                logging.warning(f"Unsupported MCP server type: {config.type}")
                continue
            
            workbench = McpWorkbench(server_params=params)
            await workbench.start()
            mcp_workbenches[name] = workbench
            logging.info(f"Successfully started MCP workbench: {name}")
        except Exception as e:
            logging.error(f"Failed to start MCP workbench {name}: {e}", exc_info=True)

    # Initialize Gemini Client
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        logging.error("GEMINI_API_KEY environment variable not set.")
        # Decide how to handle this - maybe the app shouldn't start.
        # For now, we'll let it proceed but the agent will fail.
    
    model_client = OpenAIChatCompletionClient(
        model="gemini-2.5-flash-preview-05-20", 
        api_key=gemini_api_key,
        api_type="google",
        model_info={
            "vision": True,
            "function_calling": True,
            "json_output": False,
            "family": 'gemini-2.5-flash',
            "structured_output": True,
        },
    )

    # Initialize Assistant Agents for each MCP workbench
    global agents
    for name, workbench in mcp_workbenches.items():
        try:
            agent_name = name.replace("-", "_") + "_agent" # Sanitize name for valid Python identifier
            agent = AssistantAgent(
                name=agent_name,
                system_message=settings.system_prompt + " When you are finished with the conversation, your very last message should be ONLY the word 'TERMINATE'. No other text should be included with 'TERMINATE'.",
                model_client=model_client,
                workbench=workbench, # Pass the workbench directly
                model_client_stream=True,
                reflect_on_tool_use=True,
            )
            agents[name] = agent
            logging.info(f"Assistant agent '{agent_name}' initialized with workbench '{name}'.")
        except Exception as e:
            logging.error(f"Failed to initialize agent for workbench {name}: {e}", exc_info=True)

    if not agents:
        logging.warning("No agents were initialized. Check MCP server configurations.")
    assistant_initialized_event.set()
    assistant_initialized_event.set()

@app.on_event("shutdown")
async def shutdown_event():
    global mcp_workbenches
    for name, workbench in mcp_workbenches.items():
        try:
            await workbench.stop()
            logging.info(f"Successfully stopped MCP workbench: {name}")
        except Exception as e:
            logging.error(f"Error stopping MCP workbench {name}: {e}", exc_info=True)

# --- FastAPI Middleware & Routes ---
app.mount("/static", StaticFiles(directory="./static"), name="static")
templates = Jinja2Templates(directory="./templates")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/settings", response_model=Settings)
async def get_settings():
    return load_settings()

@app.post("/api/settings")
async def update_settings(settings: Settings):
    save_settings(settings)
    # Consider restarting the agent or workbenches if critical settings changed
    return {"message": "Settings updated successfully. Restart the application for changes to take effect."}

@app.post("/api/chat")
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    await assistant_initialized_event.wait()
    if not agents:
        raise HTTPException(status_code=503, detail="AI assistant is not initialized.")

    chat_id = request.chat_id or uuid4()
    initial_message = request.message
    
    # Create or get chat session
    session = db.query(ChatSession).filter(ChatSession.id == str(chat_id)).first()
    if not session:
        title = initial_message[:50] if initial_message else "New Chat"
        session = ChatSession(id=str(chat_id), title=title, user_id=load_settings().user_id, timestamp=datetime.now())
        db.add(session)
        db.commit()
        db.refresh(session)

    async def event_generator():
        full_reply_content = ""
        thought_process = []

        yield f'data: {json.dumps({"chat_id": str(chat_id), "title": session.title, "type": "start"})}\n\n'
        
        try:
            # Fetch chat history for LLM context
            history_messages = db.query(Message).filter(Message.session_id == str(chat_id)).order_by(Message.id.asc()).all()
            
            formatted_history = ""
            if history_messages:
                for msg in history_messages:
                    if msg.sender == 'user':
                        formatted_history += f"User: {msg.message}\n"
                    elif msg.sender == 'ai':
                        # Remove TERMINATE from historical AI messages to prevent premature termination
                        cleaned_ai_message = msg.message.replace("TERMINATE", "").strip()
                        formatted_history += f"AI: {cleaned_ai_message}\n"
                formatted_history = "Previous conversation:\n" + formatted_history + "\n"

            # Fetch chat history for LLM context
            history_messages = db.query(Message).filter(Message.session_id == str(chat_id)).order_by(Message.id.asc()).all()
            
            messages_for_agent = []
            if history_messages:
                for msg in history_messages:
                    if msg.sender == 'user':
                        messages_for_agent.append(AutoGenUserMessage(content=msg.message, source="user"))
                    elif msg.sender == 'ai':
                        # Remove TERMINATE from historical AI messages for agent's internal context
                        cleaned_ai_message = msg.message.replace("TERMINATE", "").strip()
                        messages_for_agent.append(TextMessage(content=cleaned_ai_message, source="ai")) # Assuming AI messages are TextMessage

            # Add the current user message
            messages_for_agent.append(AutoGenUserMessage(content=initial_message, source="user"))

            # The task for the agent to run (now just the initial message, history is in messages_for_agent)
            task = f"{formatted_history}User query: {initial_message}"
            
            # Use a simple group chat for streaming
            team = RoundRobinGroupChat(list(agents.values()), termination_condition=TextMentionTermination(text="TERMINATE"))
            
            async for event in team.run_stream(task=task, cancellation_token=CancellationToken()):
                event_data = {}
                if isinstance(event, TextMessage):
                    # Filter out empty or whitespace-only text messages
                    if event.content and event.content.strip():
                        event_data["type"] = "TEXT_MESSAGE"
                        content = event.content
                        full_reply_content += content
                        event_data["content"] = content
                    else:
                        continue # Skip empty or whitespace-only text messages
                elif isinstance(event, ToolCallRequestEvent):
                    event_data["type"] = "TOOL_CALL"
                    tool_calls = event.content
                    for tool_call in tool_calls:
                        thought = f"Calling tool `{tool_call.name}` with arguments: {tool_call.arguments}"
                        thought_process.append(thought)
                        event_data["content"] = thought
                        event_data["tool_name"] = tool_call.name
                        event_data["tool_args"] = tool_call.arguments
                        yield f'data: {json.dumps(event_data)}\n\n'
                    continue # Continue to next event after handling all tool calls
                elif isinstance(event, ToolCallExecutionEvent):
                    event_data["type"] = "TOOL_RESULT"
                    tool_results = event.content
                    for tool_result in tool_results:
                        thought = f"Tool `{tool_result.name}` returned: {tool_result.content}"
                        thought_process.append(thought)
                        event_data["content"] = thought
                        event_data["tool_result"] = tool_result.content
                        yield f'data: {json.dumps(event_data)}\n\n'
                    continue # Continue to next event after handling all tool results
                else:
                    continue # Skip other event types for now
                
                yield f'data: {json.dumps(event_data)}\n\n'

            # End of stream
            yield f'data: {json.dumps({"type": "end", "final_reply": full_reply_content, "thought_process": thought_process})}\n\n'

        except Exception as e:
            logging.error(f"Error during chat streaming: {e}", exc_info=True)
            yield f'data: {json.dumps({"type": "error", "content": str(e)})}\n\n'
            full_reply_content = f"An error occurred: {e}"

        # Save messages to DB
        try:
            user_msg = Message(session_id=str(chat_id), sender="user", message=initial_message)
            ai_msg = Message(session_id=str(chat_id), sender="ai", message=full_reply_content, thought_process=json.dumps(thought_process))
            db.add_all([user_msg, ai_msg])
            db.commit()
            logging.info(f"Messages for chat_id {chat_id} saved to DB.")
        except Exception as db_error:
            db.rollback()
            logging.error(f"Error saving messages to DB for chat_id {chat_id}: {db_error}", exc_info=True)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), chat_id: Optional[UUID] = None, db: Session = Depends(get_db)):
    file_location = f"temp/{file.filename}"
    os.makedirs(os.path.dirname(file_location), exist_ok=True)
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    if chat_id:
        new_message = Message(session_id=str(chat_id), sender="user", message=f"Uploaded file: {file.filename}", file_path=file_location)
        db.add(new_message)
        db.commit()

    return {"reply": f"File '{file.filename}' uploaded successfully."}

@app.get("/api/chats")
async def get_chats(db: Session = Depends(get_db)):
    return db.query(ChatSession).order_by(ChatSession.timestamp.desc()).all()

@app.get("/api/chat/{chat_id}")
async def get_chat_history(chat_id: UUID, db: Session = Depends(get_db)):
    messages = db.query(Message).filter(Message.session_id == str(chat_id)).order_by(Message.id.asc()).all()
    if not messages:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    response = []
    for msg in messages:
        response.append({
            "sender": msg.sender,
            "message": msg.message,
            "thought_process": json.loads(msg.thought_process) if msg.thought_process else [],
            "file_path": msg.file_path
        })
    return response
