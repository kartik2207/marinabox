from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import  tool
from langgraph.types import Command, interrupt
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from langgraph.graph import END, StateGraph, START
from langgraph.types import Command
import os
from typing import List, TypedDict, Annotated
from marinabox import MarinaboxSDK
import json
import sys
from pathlib import Path
from langchain_anthropic import ChatAnthropic
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode,tools_condition
import base64
from io import BytesIO
import logging

mb = MarinaboxSDK(videos_path="outputs/videos")

# Tool Definitions
@tool
def GoogleDocsTool(action: str) -> str:
    """Use Google Docs in browser"""
    try:
        
        responses = mb.computer_use_command(
            "gdoc",
            action
        )
        return str(responses)
    except Exception as e:
        print(f"Error initializing Google Docs session: {e}")
        print("Please ensure Docker is running and has enough resources")
        raise
@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]

@tool
def GmailTool(action: str) -> str:
    """Use Gmail in browser"""
    try:
        # Increase timeout for container creation
        
        responses = mb.computer_use_command(
            "gmail",
            action
        )
        return str(responses)
    except Exception as e:
        print(f"Error initializing Gmail session: {e}")
        print("Please ensure Docker is running and has enough resources")
        raise

# State Management
class GraphState(TypedDict):
    input_task: str
    conversation_history: Annotated[List[BaseMessage], add_messages]
    orchestrator_thought: str
    screen_description: str
    computer_guy_steps: str
    screenshots: List[str]
    current_tool: str
    session_mapping: dict  # Stores tool -> session_id mapping

# Initialize tools and LLM
tools = [GoogleDocsTool, GmailTool, human_assistance]
llm = ChatAnthropic(model="claude-3-sonnet-20240229")
llm_with_tools = llm.bind_tools(tools)

class ShouldContinueOutput(BaseModel):
    should_continue: str

def claude_the_vision_guy(state: GraphState):
    # Parse which tool to use from orchestrator's thought
    try:
        tool = parse_tool_from_thought(state['orchestrator_thought'])
        state['current_tool'] = tool
        session_id = state['session_mapping'].get(tool)
        
        if not session_id:
            raise ValueError(f"No session ID found for tool: {tool}")
            
        vision_prompt = f"""Based on this instruction from the orchestrator: {state['orchestrator_thought']}
        
        Take a screenshot of the current browser page.
        Then scroll down a bit to capture more content and take another screenshot.
        
        Steps:
        1. screenshot
        2. scroll down
        3. screenshot again
        
        Just execute these actions, no need to analyze or describe anything."""

        responses = mb.computer_use_command(
            session_id,  # First argument is session_id
            vision_prompt  # Second argument is command
        )
        
        # Just store the screenshots, no analysis needed
        screenshots = []
        for resp in responses:
            if resp[0] == "tool_output" and resp[1].get("base64_image"):
                screenshots.append(resp[1]["base64_image"])
        
        state['screenshots'] = screenshots
        return state
    except ValueError as e:
        # Handle cases where tool isn't specified
        state['conversation_history'].append(
            AIMessage(content="Error: Please specify which tool to use (Gmail or Google Docs)")
        )
        return state

def sam_the_thinker(state: GraphState):
    # If this is the first interaction, get the input task
    if state['orchestrator_thought'] == '':
        print("INPUT TASK: ", state['input_task'])

        # Updated initial prompt to emphasize tool selection
        first_message = f"""You are O1, an intelligent agent coordinating a task using multiple tools. 
        You can only see what's on the screen through screenshots that will be provided.
        You need to give step-by-step instructions, ONE STEP AT A TIME, to accomplish the task.
        
        When giving instructions, ALWAYS specify which tool to use by starting your instruction with one of:
        - "Using Gmail: <instruction>"
        - "Using Google Docs: <instruction>"
        
        Current task: {state['input_task']}
        
        Let's start by understanding what's currently on the screen through the provided screenshots."""
        
        state['conversation_history'].append(HumanMessage(content=first_message))

    # Process screenshots and update prompt to reinforce tool selection
    if state['screenshots']:
        screenshot_message = [
            {
                "type": "text",
                "text": f"""Here are the current screenshots of the browser. 
                Based on these, determine the next step for our task: {state['input_task']}
                
                Remember to start your instruction with which tool to use:
                - "Using Gmail: <instruction>"
                - "Using Google Docs: <instruction>"
                """
            }
        ]
        
        # Convert base64 screenshots to raw binary
        for base64_screenshot in state['screenshots']:
            # Decode base64 to binary
            binary_screenshot = base64.b64decode(base64_screenshot)
            screenshot_message.append({
                "type": "image",
                "image": binary_screenshot
            })
        
        state['conversation_history'].append(HumanMessage(content=screenshot_message))

    # Get O1's next instruction
    prompt = ChatPromptTemplate.from_messages(state['conversation_history'])

    llm = ChatOpenAI(
        model="o1",
        temperature=1,
        max_tokens=None
    )

    chain = prompt | llm
    response = chain.invoke({})
    orchestrator_thought = response.content
    state['conversation_history'].append(AIMessage(content=orchestrator_thought))
    state['orchestrator_thought'] = orchestrator_thought
    print("ORCHESTRATOR THOUGHT: ", orchestrator_thought)
    
    return state

def should_continue(state: GraphState):
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0    
        )
    
    lm_structured = llm.with_structured_output(ShouldContinueOutput)
    
    messages = [SystemMessage(content="""You are a task completion checker.
    Analyze the last instruction from O1 and determine if the task is complete.
    Return 'should_continue' if more steps are needed.
    Return 'should_not_continue' if the task is complete.
    Look for explicit completion indicators in O1's message.""")]

    messages.append(HumanMessage(content=f"The last message from O1 is: {state['conversation_history'][-1].content}"))
    
    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | lm_structured
    response = chain.invoke({})
    should_continue = response.should_continue

    print("SHOULD CONTINUE: ", should_continue)

    if should_continue == "should_continue":
        return Command(goto="claude_the_computer_guy")
    else:
        print("TASK COMPLETED")
        return Command(goto=END)

def claude_the_computer_guy(state: GraphState):
    # Parse which tool to use from orchestrator's thought
    try:
        tool = parse_tool_from_thought(state['orchestrator_thought'])
        state['current_tool'] = tool
        session_id = state['session_mapping'].get(tool)
        
        if not session_id:
            raise ValueError(f"No session ID found for tool: {tool}")

        print("\n\nCOMPUTER ACTION: ", state['orchestrator_thought'])

        # Remove the tool prefix before sending the instruction
        instruction = state['orchestrator_thought'].split(':', 1)[1].strip()
        
        # First, check if the instruction is complete and clear
        completion_check_messages = [
            SystemMessage(content="""You are an instruction validator. Analyze the given instruction to determine if it's clear and actionable.
            The instruction should be specific enough for a computer to execute.
            Return 'proceed' if the instruction is clear and actionable.
            Return 'need_clarification' if the instruction is vague or incomplete.
            Consider things like:
            - Are all necessary details provided?
            - Is the action specific and unambiguous?
            - Are any required parameters or inputs specified?"""),
            HumanMessage(content=f"Instruction: {instruction}")
        ]
        
        completion_check = llm_with_tools.invoke(completion_check_messages)
        
        if "need_clarification" in completion_check.content.lower():
            state['conversation_history'].append(
                AIMessage(content=f"I need more specific instructions to proceed. Please clarify what exactly should be done in {tool}.")
            )
            return state
            
        # If instruction is clear, proceed with execution
        responses = mb.computer_use_command(
            session_id,  # First argument is session_id
            f"Execute this instruction exactly: {instruction}"  # Second argument is command
        )
        
        # Collect all responses into a readable format
        computer_response = []
        for resp in responses:
            if resp[0] == "text":
                computer_response.append(f"Computer Action: {resp[1]}")
            elif resp[0] == "tool_output":
                computer_response.append(f"Action Result: {resp[1]}")
            elif resp[0] == "tool_use":
                computer_response.append(f"Using Tool: {resp[1]} with {resp[2]}")
            elif resp[0] == "tool_error":
                computer_response.append(f"Error: {resp[1]}")
        
        formatted_response = "\n".join(computer_response)
        state['computer_guy_steps'] = formatted_response
        
        return state
        
    except ValueError as e:
        state['conversation_history'].append(
            AIMessage(content=f"Error: {str(e)}")
        )
        return state

def parse_tool_from_thought(thought: str) -> str:
    """Extract tool name from orchestrator's thought"""
    thought = thought.lower()
    if "using gmail:" in thought:
        return "gmail"
    elif "using google docs:" in thought:
        return "gdoc"
    else:
        raise ValueError("No tool specified in orchestrator's thought")

print("Initializing the samthropic agent")

# Graph Construction
graph = StateGraph(GraphState)

# Add nodes
graph.add_node("orchestrator", sam_the_thinker)
graph.add_node("vision_agent", claude_the_vision_guy)
graph.add_node("computer_agent", claude_the_computer_guy)
graph.add_node("should_continue", should_continue)

# Add tool nodes
tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)



# Add edges
graph.add_edge(START, "orchestrator")  # Add START to orchestrator edge

graph.add_conditional_edges(
    "vision_agent",
    tools_condition,
)

graph.add_conditional_edges(
    "computer_agent",
    tools_condition,
)

# # Connect agents to tools
# graph.add_edge("vision_agent", "tools")
# graph.add_edge("computer_agent", "tools")

# Connect back to orchestrator
graph.add_edge( "orchestrator", "vision_agent")
graph.add_edge( "orchestrator", "computer_agent")
graph.add_edge("orchestrator", "should_continue")

# Compile graph
workflow = graph.compile()

def setup_output_directories():
    # Create all required directories
    for directory in ["outputs", "outputs/logs", "outputs/videos"]:
        Path(directory).mkdir(parents=True, exist_ok=True)

def process_single_task(task: str, tools_to_use: List[str]):
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Create sessions for each tool and store their IDs
    session_mapping = {}
    
    # Initialize sessions for each tool
    for tool in tools_to_use:
        if tool.lower() == 'gmail':
            session = mb.create_session(
                env_type="browser", 
                tag="gmail",            
            )
            session_mapping['gmail'] = session.session_id
        elif tool.lower() == 'googledocs':
            session = mb.create_session(
                env_type="browser", 
                tag="gdoc",
            )
            session_mapping['gdoc'] = session.session_id
        # Add more tool sessions as needed
    
    try:
        logger.info(f"Starting task: {task}")
        logger.info(f"Using tools: {tools_to_use}")

        # Initialize state with session mapping
        initial_state = {
            "input_task": task,
            "conversation_history": [
                HumanMessage(content=f"Task: {task}. Use the available tools ({', '.join(tools_to_use)}) to complete this task.")
            ],
            "orchestrator_thought": "",
            "screen_description": "",
            "computer_guy_steps": "",
            "screenshots": [],
            "current_tool": None,
            "session_mapping": session_mapping  # Add session mapping to state
        }

        # Run the workflow
        result = workflow.invoke(initial_state)
        
        logger.info("Task completed successfully")
        return result

    except Exception as e:
        logger.error(f"Error during task execution: {e}")
        raise

def main():
    # Example task
    task = "Go to Google Docs, summarize everything in the document, and draft an email with this summary in Gmail"
    tools_to_use = ["googledocs", "gmail"]


    try:
        # Create output directories
        Path("outputs/logs").mkdir(parents=True, exist_ok=True)
        Path("outputs/videos").mkdir(parents=True, exist_ok=True)

        # Process the task
        result = process_single_task(task, tools_to_use)
        
        print("\nTask Execution Summary:")
        print("----------------------")
        print(f"Task: {task}")
        print(f"Tools used: {tools_to_use}")
        print("----------------------")
        
        # Print final conversation
        for message in result["conversation_history"]:
            if isinstance(message, HumanMessage):
                print(f"\nHuman: {message.content}")
            elif isinstance(message, AIMessage):
                print(f"\nAssistant: {message.content}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

# Provide a recipe for vegetarian lasagna with more than 100 reviews and a rating of at least 4.5 stars suitable for 6 people on the website https://www.allrecipes.com/.
