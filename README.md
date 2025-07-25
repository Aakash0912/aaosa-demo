# AAOSA Demo: AI-Powered Customer Support

An interactive demonstration of **Adaptive Agent-Oriented Software Architecture (AAOSA)** using AI-powered multi-agent customer support system with real LLM routing.

## Core AAOSA Concepts

- **White-Box Module**: AI-powered coordination and routing logic (the "social brain")
- **Black-Box Module**: AI specialist logic for domain-specific tasks
- **AAOSA Cycle**: Determine → Fulfill → Follow-Up

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Billing Agent  │    │Technical Agent  │    │ General Agent   │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│   White-Box     │◄──►│   White-Box     │◄──►│   White-Box     │
│ (AI Coordination│    │ (AI Coordination│    │ (AI Coordination│
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│   Black-Box     │    │   Black-Box     │    │   Black-Box     │
│ (AI Specialist) │    │ (AI Specialist) │    │ (AI Specialist) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Quick Start

### Interactive AI-Powered Demo
```bash
# Set up your API key
cp .env.template .env
# Add your ANTHROPIC_API_KEY to .env

# Install dependencies
pip install anthropic python-dotenv

# Run interactive demo
python3 ai_aaosa_demo.py
```

## Project Structure

```
aaosa-demo/
├── ai_aaosa_demo.py           # Interactive AI-powered AAOSA demo
├── .env.template              # Environment configuration template
├── requirements.txt           # Dependencies
└── README.md                  # This documentation
```

## What This Demo Shows

### 1. Determine Phase
- AI analyzes task requirements and context
- Assesses available agents and their capabilities
- Makes intelligent routing decisions with reasoning
- Considers historical performance and current load

### 2. Fulfill Phase  
- Executes task locally or delegates to best-suited peer
- Uses AI-powered domain-specific specialists
- Maintains separation between coordination and execution

### 3. Follow-Up Phase
- Records AI decisions and performance metrics
- Updates routing policies based on outcomes
- Builds performance memory for future routing decisions

## Key Features

- **AI-Powered Routing**: LLM makes intelligent delegation decisions
- **Interactive Interface**: Enter your own customer queries
- **Specialized Agents**: Billing, technical, and general support specialists
- **Real-time Delegation**: Watch tasks route to appropriate agents
- **Adaptive Learning**: System improves routing over time
- **Transparent Reasoning**: See AI's decision-making process

## Example Session

```
Customer Query: I was charged twice for my subscription

Processing task USER001 (Type: billing)
------------------------------

Processing: USER001 by general_agent
AI ROUTING: The task is related to billing, and the billing_agent is a specialist...
DELEGATED to billing_agent
LEARNED: Success=True, Quality=0.80

Response: Dear USER001, Thank you for reaching out to our billing support team...
Quality Score: 0.80
Processing Time: 3.39s
Handled by: billing
```

## Benefits Demonstrated

- **Intelligent Coordination**: AI decides optimal task routing
- **Domain Expertise**: Specialists provide expert responses
- **Scalable Architecture**: Easy to add new agent types
- **Performance Tracking**: Built-in quality and learning metrics
- **Framework Agnostic**: Works with any AI/ML backend

## Extending the Demo

Add your own specialist by implementing the `AISpecialist` class:

```python
class YourSpecialist(AISpecialist):
    def __init__(self):
        super().__init__("your_domain")
    
    def _get_specialist_prompt(self) -> str:
        return "You are an expert in your domain..."
```

## Requirements

- Python 3.8+
- Anthropic API key
- Dependencies: `anthropic`, `python-dotenv`

## Related Article

This demo accompanies the Medium article: "Adaptive Agent-Oriented Software Architecture (AAOSA): A Modular Approach to Multi-Agent Coordination"
