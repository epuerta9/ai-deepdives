# SmolaGents Architecture Documentation

This document provides a comprehensive overview of the SmolaGents framework architecture, explaining the core components, their relationships, and how they work together to create powerful AI agents.

## Table of Contents

1. [Overview](#overview)
2. [Core Components](#core-components)
3. [Agent Types](#agent-types)
4. [Models](#models)
5. [Tools](#tools)
6. [Memory and State Management](#memory-and-state-management)
7. [Code Execution](#code-execution)
8. [Monitoring and Logging](#monitoring-and-logging)
9. [Architecture Diagrams](#architecture-diagrams)
10. [Extension Points](#extension-points)

## Overview

SmolaGents is a framework for building AI agents that can perform complex tasks by breaking them down into smaller steps, using tools, and maintaining memory of previous actions. The framework is designed to be modular, extensible, and easy to use.

At a high level, SmolaGents consists of:
- **Agents**: The core entities that process tasks and make decisions
- **Models**: LLM interfaces that provide the reasoning capabilities
- **Tools**: Functions that agents can use to interact with the world
- **Memory**: Storage for agent state and history
- **Executors**: Components that run code or other operations
- **Monitoring**: Systems for tracking agent behavior and performance

```mermaid
graph TD
    User[User] --> |Task| Agent
    Agent --> |Prompts| Model
    Model --> |Responses| Agent
    Agent --> |Uses| Tools
    Tools --> |Results| Agent
    Agent --> |Stores| Memory
    Memory --> |Retrieves| Agent
    Agent --> |Executes| Executors
    Executors --> |Results| Agent
    Agent --> |Logs| Monitoring
    Agent --> |Final Answer| User
```

## Core Components

### Agents (`agents.py`)

Agents are the central components that orchestrate the entire process. The main agent classes are:

- **MultiStepAgent**: The base agent class that implements the core agent loop
- **ToolCallingAgent**: Specialized for tool usage with structured outputs
- **CodeAgent**: Specialized for generating and executing code

Agents follow a step-based execution model:
1. Receive a task
2. Plan how to approach the task (optional)
3. Execute steps by calling the model and using tools
4. Provide a final answer

```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant Model
    participant Tools
    participant Memory
    
    User->>Agent: Submit task
    Agent->>Memory: Initialize memory with task
    opt Planning enabled
        Agent->>Model: Generate initial plan
        Model->>Agent: Return plan
        Agent->>Memory: Store plan
    end
    
    loop Until completion or max steps
        Agent->>Memory: Retrieve context
        Agent->>Model: Send context and prompt
        Model->>Agent: Generate next action
        Agent->>Tools: Execute tool call
        Tools->>Agent: Return tool results
        Agent->>Memory: Store step results
        opt Planning interval reached
            Agent->>Model: Update plan
            Model->>Agent: Return updated plan
            Agent->>Memory: Store updated plan
        end
    end
    
    Agent->>User: Return final answer
```

### Agent Types (`agent_types.py`)

The framework defines special types for agent inputs and outputs:

- **AgentType**: Base class for agent-specific types
- **AgentText**: For text outputs
- **AgentImage**: For image outputs
- **AgentAudio**: For audio outputs

These types provide consistent handling of different data modalities across the framework.

## Models (`models.py`)

Models provide the reasoning capabilities for agents. The framework supports multiple model backends:

- **Model**: Base class for all models
- **HfApiModel**: Uses the Hugging Face API
- **MLXModel**: Uses MLX for local inference
- **TransformersModel**: Uses the Transformers library
- **LiteLLMModel**: Uses LiteLLM for various API providers
- **OpenAIServerModel**: Uses OpenAI's API
- **AzureOpenAIServerModel**: Uses Azure OpenAI's API

Models handle:
- Converting agent messages to the format expected by the model provider
- Managing tool definitions for function calling
- Processing model responses into a standardized format

```mermaid
classDiagram
    class Model {
        +__call__(messages, stop_sequences, grammar, tools_to_call_from)
        +to_dict()
        +from_dict(model_dictionary)
    }
    
    class HfApiModel {
        +model_id
        +provider
        +token
        +timeout
    }
    
    class TransformersModel {
        +model_id
        +device_map
        +torch_dtype
    }
    
    class LiteLLMModel {
        +model_id
        +api_base
        +api_key
    }
    
    class OpenAIServerModel {
        +model_id
        +api_base
        +api_key
        +organization
    }
    
    Model <|-- HfApiModel
    Model <|-- MLXModel
    Model <|-- TransformersModel
    Model <|-- LiteLLMModel
    Model <|-- OpenAIServerModel
    OpenAIServerModel <|-- AzureOpenAIServerModel
```

## Tools (`tools.py`, `default_tools.py`)

Tools are functions that agents can use to interact with the world. The framework provides:

- **Tool**: Base class for all tools
- **PipelineTool**: Specialized for Transformer pipelines
- **Default tools**: Built-in tools like web search, Python interpreter, etc.

Tools have:
- A name and description
- Input schema
- Output type
- Implementation logic

Tools can be:
- Created from functions using the `@tool` decorator
- Loaded from the Hugging Face Hub
- Created from Gradio or LangChain tools

```mermaid
classDiagram
    class Tool {
        +name: str
        +description: str
        +inputs: Dict
        +output_type: str
        +forward(*args, **kwargs)
        +__call__(*args, **kwargs)
        +setup()
        +to_dict()
        +save(output_dir)
        +push_to_hub(repo_id)
    }
    
    class PipelineTool {
        +model_class
        +pre_processor_class
        +post_processor_class
        +default_checkpoint
        +setup()
        +encode(raw_inputs)
        +forward(inputs)
        +decode(outputs)
    }
    
    class PythonInterpreterTool {
        +authorized_imports
        +base_python_tools
        +python_evaluator
        +forward(code)
    }
    
    class FinalAnswerTool {
        +forward(answer)
    }
    
    class WebSearchTool {
        +forward(query)
    }
    
    Tool <|-- PipelineTool
    Tool <|-- PythonInterpreterTool
    Tool <|-- FinalAnswerTool
    Tool <|-- WebSearchTool
```

## Memory and State Management (`memory.py`)

Memory components store the agent's state and history:

- **AgentMemory**: Manages the agent's memory
- **MemoryStep**: Base class for memory steps
- **ActionStep**: Represents a tool execution step
- **PlanningStep**: Represents a planning step
- **TaskStep**: Represents a task definition step
- **SystemPromptStep**: Represents a system prompt step

Memory is used to:
- Store the history of agent actions
- Provide context for the model
- Enable replay and visualization of agent execution

```mermaid
classDiagram
    class AgentMemory {
        +steps: List[MemoryStep]
        +system_prompt: str
        +reset()
        +get_succinct_steps()
        +get_full_steps()
        +replay(logger, detailed)
    }
    
    class MemoryStep {
        +dict()
        +to_messages()
    }
    
    class ActionStep {
        +model_input_messages
        +tool_calls
        +model_output
        +observations
        +action_output
        +to_messages()
    }
    
    class PlanningStep {
        +model_input_messages
        +facts
        +plan
        +to_messages()
    }
    
    class TaskStep {
        +task
        +task_images
        +to_messages()
    }
    
    class SystemPromptStep {
        +system_prompt
        +to_messages()
    }
    
    AgentMemory o-- MemoryStep
    MemoryStep <|-- ActionStep
    MemoryStep <|-- PlanningStep
    MemoryStep <|-- TaskStep
    MemoryStep <|-- SystemPromptStep
```

## Code Execution (`local_python_executor.py`, `remote_executors.py`)

The framework provides several ways to execute code:

- **PythonExecutor**: Base class for Python code execution
- **LocalPythonExecutor**: Executes Python code locally with safety constraints
- **DockerExecutor**: Executes code in a Docker container
- **E2BExecutor**: Executes code using the E2B service

The code execution system:
- Parses and evaluates Python code safely
- Provides a sandboxed environment
- Limits imports and operations for security
- Captures outputs and errors

```mermaid
classDiagram
    class PythonExecutor {
        +__call__(code_action)
        +send_variables(variables)
        +send_tools(tools)
    }
    
    class LocalPythonExecutor {
        +additional_authorized_imports
        +max_print_outputs_length
        +__call__(code_action)
    }
    
    class DockerExecutor {
        +docker_image
        +__call__(code_action)
    }
    
    class E2BExecutor {
        +api_key
        +__call__(code_action)
    }
    
    PythonExecutor <|-- LocalPythonExecutor
    PythonExecutor <|-- DockerExecutor
    PythonExecutor <|-- E2BExecutor
```

## Monitoring and Logging (`monitoring.py`)

The framework provides monitoring and logging capabilities:

- **Monitor**: Tracks agent metrics like token usage and step duration
- **AgentLogger**: Provides logging with different verbosity levels
- **LogLevel**: Defines logging verbosity levels

Monitoring helps:
- Track agent performance
- Debug agent behavior
- Visualize agent execution

```mermaid
classDiagram
    class Monitor {
        +tracked_model
        +logger
        +step_durations
        +total_input_token_count
        +total_output_token_count
        +get_total_token_counts()
        +reset()
        +update_metrics(step_log)
    }
    
    class LogLevel {
        +OFF
        +ERROR
        +INFO
        +DEBUG
    }
    
    class AgentLogger {
        +level: LogLevel
        +console
        +log(*args, level, **kwargs)
        +log_error(error_message)
        +log_markdown(content, title)
        +log_code(title, content)
        +log_rule(title)
        +log_task(content, subtitle, title)
        +log_messages(messages)
        +visualize_agent_tree(agent)
    }
    
    Monitor o-- AgentLogger
    AgentLogger o-- LogLevel
```

## Architecture Diagrams

### High-Level Component Interaction

```mermaid
graph TD
    subgraph User Interface
        CLI[CLI]
        GradioUI[Gradio UI]
    end
    
    subgraph Agent Framework
        Agents[Agents]
        Models[Models]
        Tools[Tools]
        Memory[Memory]
        Executors[Executors]
        Monitoring[Monitoring]
    end
    
    subgraph External Services
        ModelAPIs[Model APIs]
        WebServices[Web Services]
        DockerRuntime[Docker Runtime]
    end
    
    CLI --> Agents
    GradioUI --> Agents
    
    Agents --> Models
    Agents --> Tools
    Agents --> Memory
    Agents --> Executors
    Agents --> Monitoring
    
    Models --> ModelAPIs
    Tools --> WebServices
    Executors --> DockerRuntime
```

### Agent Execution Flow

```mermaid
stateDiagram-v2
    [*] --> Initialize
    Initialize --> Planning: if planning_enabled
    Initialize --> Execution: if not planning_enabled
    
    Planning --> Execution
    
    state Execution {
        [*] --> PrepareContext
        PrepareContext --> CallModel
        CallModel --> ParseModelOutput
        ParseModelOutput --> ExecuteTool
        ExecuteTool --> StoreResults
        StoreResults --> CheckCompletion
        CheckCompletion --> UpdatePlan: if planning_interval reached
        UpdatePlan --> PrepareContext
        CheckCompletion --> PrepareContext: if not complete
        CheckCompletion --> [*]: if complete or max_steps
    }
    
    Execution --> FinalAnswer
    FinalAnswer --> [*]
```

## Extension Points

SmolaGents is designed to be extensible. Key extension points include:

1. **Custom Agents**: Create new agent types by subclassing `MultiStepAgent`
2. **Custom Tools**: Create new tools by subclassing `Tool` or using the `@tool` decorator
3. **Custom Models**: Support new model providers by subclassing `Model`
4. **Custom Executors**: Create new execution environments by subclassing `PythonExecutor`
5. **Custom Prompt Templates**: Customize agent behavior with custom prompt templates

To extend the framework:

```python
# Create a custom tool
from smolagents.tools import Tool

class MyCustomTool(Tool):
    name = "my_custom_tool"
    description = "Does something amazing"
    inputs = {"input": {"type": "string", "description": "Input description"}}
    output_type = "string"
    
    def forward(self, input: str) -> str:
        # Implementation
        return result

# Create a custom agent
from smolagents.agents import MultiStepAgent

class MyCustomAgent(MultiStepAgent):
    def initialize_system_prompt(self) -> str:
        # Custom system prompt
        return "You are a specialized agent that..."
    
    def step(self, memory_step):
        # Custom step implementation
        # ...
```

This architecture document provides a high-level overview of the SmolaGents framework. For more detailed information, refer to the source code and documentation. 