#!/usr/bin/env python3
"""
AI-Powered AAOSA Demo: Real LLM-based routing and coordination
This demonstrates true AAOSA with AI making routing decisions
"""

import os
import time
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from enum import Enum

try:
    import anthropic
    from dotenv import load_dotenv
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    print("Missing dependencies. Install with: pip install anthropic python-dotenv")

load_dotenv()

class TaskType(Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    GENERAL = "general"

class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class CustomerTask:
    id: str
    content: str
    task_type: TaskType
    priority: Priority
    customer_id: str
    metadata: Dict[str, Any]
    created_at: datetime

@dataclass
class RoutingDecision:
    handle_locally: bool
    target_agent: Optional[str] = None
    confidence: float = 0.0
    reasoning: str = ""
    ai_analysis: str = ""

@dataclass
class TaskResult:
    success: bool
    response: str
    quality_score: float
    execution_time: float
    handled_by: str

class AICoordinator:
    """
    AI-powered coordination module that makes intelligent routing decisions
    This is the WHITE-BOX module with AI-enhanced decision making
    """
    
    def __init__(self):
        self.client = None
        self.routing_history = []
        self.performance_memory = {}
        self.delegation_threshold = 0.7
        
        if DEPENDENCIES_AVAILABLE:
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if api_key:
                self.client = anthropic.Anthropic(api_key=api_key)
    
    def determine_routing(self, task: CustomerTask, available_agents: Dict[str, Any], current_agent_id: str = "general_agent") -> RoutingDecision:
        """
        AI-POWERED DETERMINE PHASE: Use LLM to make intelligent routing decisions
        """
        if not self.client:
            return self._fallback_routing(task, available_agents)
        
        # Prepare context for AI routing decision
        agent_info = self._format_agent_capabilities(available_agents)
        historical_context = self._get_historical_context(task.task_type)
        
        system_prompt = f"""You are an intelligent task routing coordinator in an AAOSA (Adaptive Agent-Oriented Software Architecture) system.

Your job is to analyze customer support tasks and decide which agent should handle them based on:
1. Agent specializations and current capabilities
2. Historical performance data
3. Task complexity and priority
4. Current system load

The task is currently being processed by: {current_agent_id}

If the recommended agent is different from the current agent ({current_agent_id}), set "handle_locally" to false.
If the recommended agent is the same as the current agent, set "handle_locally" to true.

Respond with a JSON object containing:
{{
    "recommended_agent": "agent_name",
    "handle_locally": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "detailed explanation",
    "alternative_agents": ["agent1", "agent2"],
    "risk_factors": ["factor1", "factor2"]
}}"""

        user_prompt = f"""
TASK TO ROUTE:
- ID: {task.id}
- Type: {task.task_type.value}
- Priority: {task.priority.name}
- Content: {task.content}
- Customer: {task.customer_id}
- Currently handled by: {current_agent_id}

AVAILABLE AGENTS:
{agent_info}

HISTORICAL PERFORMANCE:
{historical_context}

Please analyze this task and recommend the best routing decision.
"""

        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            ai_response = response.content[0].text
            routing_data = self._parse_ai_routing_response(ai_response, current_agent_id)
            
            return RoutingDecision(
                handle_locally=routing_data.get("handle_locally", True),
                target_agent=routing_data.get("recommended_agent"),
                confidence=routing_data.get("confidence", 0.7),
                reasoning=routing_data.get("reasoning", "AI routing decision"),
                ai_analysis=ai_response
            )
            
        except Exception as e:
            print(f"AI routing failed: {e}")
            return self._fallback_routing(task, available_agents)
    
    def _parse_ai_routing_response(self, ai_response: str, current_agent_id: str) -> Dict[str, Any]:
        """Parse AI response and extract routing decision"""
        try:
            # Try to extract JSON from the response
            start_idx = ai_response.find('{')
            end_idx = ai_response.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = ai_response[start_idx:end_idx]
                parsed_data = json.loads(json_str)
                
                # Fix logic: if AI recommends a different agent, don't handle locally
                recommended_agent = parsed_data.get("recommended_agent", "")
                if recommended_agent and recommended_agent != current_agent_id:
                    parsed_data["handle_locally"] = False
                else:
                    parsed_data["handle_locally"] = True
                
                return parsed_data
        except:
            pass
        
        # Fallback parsing
        return {
            "handle_locally": True,
            "confidence": 0.7,
            "reasoning": "Parsed from AI response",
            "recommended_agent": current_agent_id
        }
    
    def _format_agent_capabilities(self, agents: Dict[str, Any]) -> str:
        """Format agent information for AI context"""
        info = []
        for agent_id, agent in agents.items():
            specialization = getattr(agent, 'specialization', 'general')
            success_rate = self._get_agent_success_rate(agent_id)
            current_load = getattr(agent, 'current_load', 0)
            
            info.append(f"- {agent_id}: {specialization} specialist, {success_rate:.1%} success rate, load: {current_load}")
        
        return "\n".join(info)
    
    def _get_historical_context(self, task_type: TaskType) -> str:
        """Get historical performance context for this task type"""
        if task_type.value not in self.performance_memory:
            return "No historical data available"
        
        history = self.performance_memory[task_type.value]
        if not history:
            return "No historical data available"
        
        avg_success = sum(h.get('success', 0) for h in history) / len(history)
        avg_quality = sum(h.get('quality_score', 0) for h in history) / len(history)
        
        return f"Historical performance: {avg_success:.1%} success rate, {avg_quality:.2f} avg quality"
    
    def _get_agent_success_rate(self, agent_id: str) -> float:
        """Get success rate for specific agent"""
        # Simplified - would normally track per-agent performance
        return 0.85  # Default success rate
    
    def _fallback_routing(self, task: CustomerTask, available_agents: Dict[str, Any]) -> RoutingDecision:
        """Fallback routing when AI is unavailable"""
        # Simple rule-based routing
        preferred_agent = f"{task.task_type.value}_agent"
        if preferred_agent in available_agents:
            return RoutingDecision(
                handle_locally=True,
                target_agent=preferred_agent,
                confidence=0.6,
                reasoning=f"Rule-based routing to {preferred_agent}"
            )
        
        return RoutingDecision(
            handle_locally=True,
            target_agent="general_agent",
            confidence=0.5,
            reasoning="Fallback to general agent"
        )

class AISpecialist:
    """
    AI-powered specialist that handles domain-specific tasks
    This is the BLACK-BOX module with AI execution
    """
    
    def __init__(self, specialization: str):
        self.specialization = specialization
        self.client = None
        self.current_load = 0
        
        if DEPENDENCIES_AVAILABLE:
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if api_key:
                self.client = anthropic.Anthropic(api_key=api_key)
    
    def fulfill(self, task: CustomerTask) -> TaskResult:
        """
        AI-POWERED FULFILL PHASE: Use LLM for intelligent task execution
        """
        start_time = time.time()
        self.current_load += 1
        
        try:
            if self.client:
                response = self._ai_fulfill(task)
                quality_score = self._ai_evaluate_quality(task, response)
            else:
                response = self._fallback_fulfill(task)
                quality_score = 0.7
            
            execution_time = time.time() - start_time
            
            return TaskResult(
                success=True,
                response=response,
                quality_score=quality_score,
                execution_time=execution_time,
                handled_by=self.specialization
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TaskResult(
                success=False,
                response=f"Error: {str(e)}",
                quality_score=0.0,
                execution_time=execution_time,
                handled_by=self.specialization
            )
        finally:
            self.current_load = max(0, self.current_load - 1)
    
    def _ai_fulfill(self, task: CustomerTask) -> str:
        """Use AI to fulfill the task"""
        system_prompt = self._get_specialist_prompt()
        
        user_prompt = f"""
Customer Support Request:
- Customer ID: {task.customer_id}
- Priority: {task.priority.name}
- Issue: {task.content}
- Context: {json.dumps(task.metadata, indent=2)}

Please provide a professional, helpful response that addresses this customer's specific needs.
"""

        response = self.client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=800,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        
        return response.content[0].text
    
    def _ai_evaluate_quality(self, task: CustomerTask, response: str) -> float:
        """Use AI to evaluate response quality"""
        try:
            eval_prompt = f"""
Rate the quality of this customer support response on a scale of 0.0 to 1.0:

CUSTOMER ISSUE: {task.content}
RESPONSE: {response}

Consider: relevance, helpfulness, professionalism, completeness.
Respond with just a number between 0.0 and 1.0.
"""

            eval_response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": eval_prompt}]
            )
            
            score_text = eval_response.content[0].text.strip()
            return float(score_text)
            
        except:
            return 0.8  # Default quality score
    
    def _get_specialist_prompt(self) -> str:
        """Get specialized system prompt"""
        prompts = {
            "billing": "You are an expert billing support specialist. Handle payment issues, refunds, account billing, and subscription matters with precision and empathy.",
            "technical": "You are a senior technical support engineer. Diagnose and solve technical issues, bugs, and system problems with clear step-by-step guidance.",
            "general": "You are an experienced customer support representative. Handle diverse inquiries with professionalism, empathy, and solution-focused approach."
        }
        return prompts.get(self.specialization, prompts["general"])
    
    def _fallback_fulfill(self, task: CustomerTask) -> str:
        """Fallback when AI is unavailable"""
        return f"Thank you for contacting {self.specialization} support. We have received your inquiry about: {task.content}. Our team will review and respond within 24 hours."

class AIAAOSAAgent:
    """
    Complete AAOSA Agent with AI-powered coordination and execution
    """
    
    def __init__(self, agent_id: str, specialization: str):
        self.agent_id = agent_id
        self.specialization = specialization
        self.coordinator = AICoordinator()  # WHITE-BOX: AI coordination
        self.specialist = AISpecialist(specialization)  # BLACK-BOX: AI specialist
        self.peer_network = {}
    
    def register_peer(self, peer_agent):
        """Register a peer agent for coordination"""
        self.peer_network[peer_agent.agent_id] = peer_agent
    
    def process_task(self, task: CustomerTask) -> TaskResult:
        """
        Complete AAOSA cycle with AI-powered DETERMINE, FULFILL, and FOLLOW-UP
        """
        print(f"\nProcessing: {task.id} by {self.agent_id}")
        
        # DETERMINE: AI-powered routing decision
        routing_decision = self.coordinator.determine_routing(task, self.peer_network, self.agent_id)
        print(f"AI ROUTING: {routing_decision.reasoning}")
        if routing_decision.ai_analysis:
            print(f"AI ANALYSIS: {routing_decision.ai_analysis[:200]}...")
        
        # FULFILL: Execute task (locally or delegate)
        if routing_decision.handle_locally or not routing_decision.target_agent:
            result = self.specialist.fulfill(task)
            print(f"FULFILLED locally by {self.agent_id}")
        else:
            # Delegate to peer (simplified for demo)
            target_peer = self.peer_network.get(routing_decision.target_agent)
            if target_peer:
                result = target_peer.specialist.fulfill(task)
                print(f"DELEGATED to {routing_decision.target_agent}")
            else:
                result = self.specialist.fulfill(task)
                print(f"FALLBACK to local processing")
        
        # FOLLOW-UP: Learn from outcome
        self._follow_up(task, routing_decision, result)
        
        return result
    
    def _follow_up(self, task: CustomerTask, routing_decision: RoutingDecision, result: TaskResult):
        """AI-enhanced learning and adaptation"""
        # Record interaction for learning
        interaction = {
            'task_id': task.id,
            'task_type': task.task_type.value,
            'routing_decision': asdict(routing_decision),
            'result': asdict(result),
            'timestamp': datetime.now().isoformat()
        }
        
        self.coordinator.routing_history.append(interaction)
        
        # Update performance memory
        task_type = task.task_type.value
        if task_type not in self.coordinator.performance_memory:
            self.coordinator.performance_memory[task_type] = []
        
        self.coordinator.performance_memory[task_type].append({
            'success': result.success,
            'quality_score': result.quality_score,
            'execution_time': result.execution_time
        })
        
        print(f"LEARNED: Success={result.success}, Quality={result.quality_score:.2f}")

def main():
    """Demonstrate AI-powered AAOSA in action"""
    print("AI-Powered AAOSA Customer Support Demo")
    print("=" * 50)
    
    if not DEPENDENCIES_AVAILABLE:
        print("Missing dependencies. Please install: pip install anthropic python-dotenv")
        return
    
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("Missing ANTHROPIC_API_KEY. Please set it in your .env file")
        return
    
    # Create AI-powered agents
    agents = {
        'billing_agent': AIAAOSAAgent("billing_agent", "billing"),
        'technical_agent': AIAAOSAAgent("technical_agent", "technical"),
        'general_agent': AIAAOSAAgent("general_agent", "general")
    }
    
    # Register peer relationships
    for agent_id, agent in agents.items():
        for peer_id, peer in agents.items():
            if agent_id != peer_id:
                agent.register_peer(peer)
    
    print(f"Initialized {len(agents)} AI-powered AAOSA agents")
    print("\nAvailable agents:")
    print("- billing_agent: Handles payment, refund, and billing issues")
    print("- technical_agent: Handles technical problems and bugs")
    print("- general_agent: Handles general inquiries and questions")
    
    # Interactive mode
    print("\n" + "=" * 50)
    print("Enter your customer support queries (type 'quit' to exit)")
    print("=" * 50)
    
    task_counter = 1
    
    while True:
        try:
            # Get user input
            user_query = input("\nCustomer Query: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("Thank you for using AAOSA Customer Support!")
                break
            
            if not user_query:
                print("Please enter a valid query.")
                continue
            
            # Determine task type based on keywords (simple classification)
            task_type = classify_query(user_query)
            
            # Create customer task
            task = CustomerTask(
                id=f"USER{task_counter:03d}",
                content=user_query,
                task_type=task_type,
                priority=Priority.MEDIUM,
                customer_id="USER001",
                metadata={"source": "interactive", "session": datetime.now().isoformat()},
                created_at=datetime.now()
            )
            
            print(f"\nProcessing task {task.id} (Type: {task_type.value})")
            print("-" * 30)
            
            # Route to most appropriate agent (AI will decide)
            primary_agent = agents['general_agent']  # Start with general, AI will route
            result = primary_agent.process_task(task)
            
            print(f"\nResponse: {result.response}")
            print(f"Quality Score: {result.quality_score:.2f}")
            print(f"Processing Time: {result.execution_time:.2f}s")
            print(f"Handled by: {result.handled_by}")
            
            task_counter += 1
            
        except KeyboardInterrupt:
            print("\n\nSession interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error processing query: {str(e)}")

def classify_query(query: str) -> TaskType:
    """Simple keyword-based query classification"""
    query_lower = query.lower()
    
    # Billing keywords
    billing_keywords = ['bill', 'payment', 'charge', 'refund', 'invoice', 'subscription', 'money', 'cost', 'price', 'fee']
    if any(keyword in query_lower for keyword in billing_keywords):
        return TaskType.BILLING
    
    # Technical keywords  
    tech_keywords = ['error', 'bug', 'crash', 'broken', 'not working', 'issue', 'problem', 'app', 'website', 'login', 'upload']
    if any(keyword in query_lower for keyword in tech_keywords):
        return TaskType.TECHNICAL
    
    # Default to general
    return TaskType.GENERAL

if __name__ == "__main__":
    main()
