"""
This module implements the Agent Coordinator pattern for the Sleep Health Chatbot.
It coordinates different specialized agents to handle various types of user queries.
"""

from typing import Dict, Any, Optional, Union, Generator
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from operator import itemgetter

class SpecializedAgent:
    """Base class for specialized agents"""
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.chat = ChatOpenAI(model=model, streaming=True)
    
    def process(self, *args, **kwargs):
        raise NotImplementedError("Specialized agents must implement process method")

class RecommendationAgent(SpecializedAgent):
    """Agent specialized in providing activity recommendations"""
    def __init__(self, model: str = "gpt-4o"):
        super().__init__(model)
        self.prompt = ChatPromptTemplate.from_template("""
        As a compassionate and helpful sleep health expert, your mission is to offer personalized, detailed, and actionable advice to enhance sleep quality. Here's how you can make your recommendations impactful:

        1. Tailor your advice to fit the current context—consider the time of day, weather, temperature, and location, as well as the user's sleep bed times.
        2. Align your suggestions with the user's habits, life styles, and current capabilities, utilizing our UCB-based multi-armed bandit model's recommendations as a starting point.
        3. Explain why each piece of advice is beneficial, referencing sleep health theories, the user's habit data, and their personal profile.
        4. Frame your suggestions as a negotiation, encouraging users to adopt positive sleep habits.
        
        Considering users' sleep bed times:
        - sleep bedtime starts: {bedtime}
                                                                 
        Consider users' personal preferences/limitations/life styles from the conversation context:
        - User input: {question}
        - Context: {context}
                                                                 
        And the current contextual factors:
        - Current time: {time}
        - Temperature: {temperature}
        - Weather: {weather}
        - Location: {location}

        And considering the recommendation from our behavior recommendation model to [engage in/do/try] {recommendations} based on the user's lifestyle,

        Please refine, modify, or explain this recommendation to ensure it fits both the current context and the personal preferences/limitations. 
        Offer creative, yet feasible, alternatives if necessary. 
        You should address the user in the second person. Your response should be concise and 2-3 short sentences max (within 50 words).
        """)
        self.chain = self.prompt | self.chat | StrOutputParser()

    def process(self, context: Dict[str, Any]) -> str:
        return self.chain.stream(context)

class HealthDataAgent(SpecializedAgent):
    """Agent specialized in analyzing health data"""
    def __init__(self, model: str = "gpt-4o"):
        super().__init__(model)
    
    def process(self, df, query: str) -> str:
        agent = create_pandas_dataframe_agent(
            ChatOpenAI(model="gpt-4o", temperature=0),
            df,
            prefix="""Analyze the health data to answer the user's question. Follow these steps:

            1. Identify key quantitative insights where possible. Focus on patterns, trends, and notable statistics. (This is for your reference only)

            2. Relate your findings to one relevant theoretical framework dimension from this list:
                1. Consequences and reinforcement
                2. Feedback and monitoring
                3. Goals
                4. Knowledge
                5. Environmental context and resources
                6. Skills and capabilities
                7. Emotional support
            (This connection is for your reference only)

            3. Act as a health coach, craft a natural, conversational response based on your analysis. This response should:
            - Be concise (2-3 short sentences)
            - INCLUDE key quantitative insights requested by users in response, such as patterns, trends, and notable statistics
            - Relate the key insights to relevant theoretical framework dimension without using technical language

            Only output the final conversational response. Do not include headers, summaries, or analysis sections in your output.""",
            verbose=True,
            include_df_in_prompt=True,
            number_of_head_rows=30,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            agent_executor_kwargs={"handle_parsing_errors":True},
            allow_dangerous_code=True,
            return_intermediate_steps=False
        )
        return agent.invoke(query)["output"]

class GeneralQueryAgent(SpecializedAgent):
    """Agent specialized in handling general queries"""
    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__(model)
        self.framework_selection_prompt = ChatPromptTemplate.from_template(
            """Based on the user's input and previous conversation context, determine the most relevant theoretical dimensions to guide the response. Consider the following dimensions:

            1. Consequences, reinforcement
            2. Feedback and monitoring
            3. Goals
            4. Knowledge
            5. Environmental context and resources
            6. Skills and capabilities
            7. Emotional support

            User input: {question}
            Conversation context: {context}

            Select ONLY the most relevant dimensions (<=3) and explain why they are relevant. 
            Format your response as a JSON object with 'dimensions' as a list and 'explanations' as a dictionary.
            Keep 'explanations' in 1-2 sentences (or no more than 40 words) and relevant to the user's current situation.
            """
        )
        self.framework_selection_chain = self.framework_selection_prompt | self.chat | StrOutputParser()

        self.oura_analysis_prompt = ChatPromptTemplate.from_template(
            """Analyze the user's Oura data and provide insights relevant to the selected theoretical dimensions. Focus on sleep patterns, activity levels, readiness scores, and stress levels.

            Oura data: {oura_data}
            Selected dimensions: {selected_dimensions}

            Provide a concise summary of insights related to each selected dimension. Please provide no more than 2 sentences per dimension.
            """
        )
        self.oura_analysis_chain = self.oura_analysis_prompt | self.chat | StrOutputParser()

    def process(self, query: str, context: str, oura_data: Any) -> str:
        selected_frameworks = self.framework_selection_chain.invoke({
            "question": query,
            "context": context
        })

        oura_insights = self.oura_analysis_chain.invoke({
            "oura_data": oura_data,
            "selected_dimensions": selected_frameworks
        })

        return {
            "selected_frameworks": selected_frameworks,
            "oura_insights": oura_insights
        }

class AgentCoordinator:
    """Coordinates different specialized agents based on query type"""
    def __init__(self):
        self.recommendation_agent = RecommendationAgent()
        self.health_data_agent = HealthDataAgent()
        self.general_query_agent = GeneralQueryAgent()
        
        # Initialize classification prompts
        self.recommendation_classifier = ChatPromptTemplate.from_template(
            """Based on the user's input and previous conversation context, determine if the user is requesting initial recommendations for sleep-enhancing activities. 
            Exclude follow-up questions about specific activities. 
            Respond with only "yes" or "no".

            <input>
            {question}
            </input>

            <context>
            {context}
            </context>

            Classification:"""
        ) | ChatOpenAI(streaming=True, temperature=0, model="gpt-4o-mini") | StrOutputParser()

        self.health_data_classifier = ChatPromptTemplate.from_template(
            """Given the user input question, decide whether the user asks the questions related to their sleep, activity, stress, readiness, and physiological data (e.g., heart rate,  breath) in dataframe. 
            - Regarding sleep, the data includes time in bed, bedtime start, bedtime end, total sleep duration, average breath, average hrv, and lowest heart rate.
            - Regarding activity, the data includes activity score (1-100), activity (e.g., walking, running).
            - Regarding readiness, the data includes readiness score (1-100).
            - Regarding stress, the data includes stress level (restored, normal, stressful).
            Please respond "yes" or "no.
            Do not respond with more than one word.

            <input>
            {question}
            </input>

            Classification:"""
        ) | ChatOpenAI(streaming=True, temperature=0, model="gpt-4o-mini") | StrOutputParser()

    def classify_query(self, query: str, context: Optional[str] = None) -> str:
        """Classify the type of query to determine which agent should handle it"""
        if "yes" in self.recommendation_classifier.invoke({
            "question": query,
            "context": context if context else "none"
        }).lower():
            return "recommendation"
        elif "yes" in self.health_data_classifier.invoke({
            "question": query
        }).lower():
            return "health_data"
        else:
            return "general"

    def process_query(self, query: str, context: Dict[str, Any]) -> Union[str, Generator]:
        """Process the query using the appropriate specialized agent"""
        query_type = self.classify_query(query, context.get("conversation_context"))
        
        try:
            if query_type == "recommendation":
                # Map context variables to match the prompt template's expected variables
                recommendation_context = {
                    "question": query,  # Map query to question
                    "context": context.get("conversation_context", ""),  # Map conversation_context to context
                    "bedtime": context["bedtime"],
                    "time": context["time"],
                    "temperature": context["temperature"],
                    "weather": context["weather"],
                    "location": context["location"],
                    "recommendations": context["recommendations"]
                }
                return self.recommendation_agent.process(recommendation_context)
            elif query_type == "health_data":
                # Health data agent returns a string, not a stream
                response = self.health_data_agent.process(context["health_data"], query)
                return response
            else:
                general_response = self.general_query_agent.process(
                    query=query,
                    context=context.get("conversation_context", ""),
                    oura_data=context.get("health_data")
                )
                return self.process_general_response(general_response, context)
        except Exception as e:
            print(f"Error in process_query: {str(e)}")
            return f"I apologize, but I encountered an error while processing your request. Please try rephrasing your question or try again later."

    def process_general_response(self, general_response: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Process the general response using the chat chain"""
        chatchain = (
            RunnablePassthrough.assign(
                history=RunnableLambda(context["memory"].load_memory_variables) | itemgetter("history")
            )
            | context["chat_prompt"]
            | context["chat"]
            | StrOutputParser()
        )
        
        inputs = {
            "input": context["query"],
            "oura_insights": general_response["oura_insights"],
            "selected_dimensions": general_response["selected_frameworks"]
        }
        return chatchain.stream(inputs) 