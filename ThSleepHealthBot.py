# Standard library imports
import os
import uuid
import time
from datetime import datetime, timedelta

# Third-party imports
import streamlit as st
import pandas as pd
import dill as pickle
import tiktoken

# LangChain imports
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from operator import itemgetter

# Local imports
import utils
from agent_coordinator import AgentCoordinator


# Pre-defined constants
MODEL_NAME = "gpt-4o-mini"  # Use standard OpenAI model name

st.set_page_config(page_title="Sleep Health Chatbot", page_icon="🤖")
# Get current time
current_day = datetime.now().strftime("%Y-%m-%d")
start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
# system type
system_type = "healthguru"

with st.sidebar:
    st.page_link("ThSleepHealthBot.py", label="SleepHealthChabot", icon="🏠")

    # Try to load API keys from secrets first
    try:
        openai_api_key = st.secrets["openai"]["openaikey"]
        oura_token = st.secrets["oura"]["oura_token"]
        weatherapi_key = st.secrets["weatherapi"]["weatherapi_key"]
    except Exception:
        # If loading from secrets fails, prompt for user input
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        oura_token = st.text_input("Oura Ring Token", type="password")
        weatherapi_key = st.text_input("WeatherAPI (api.weatherapi.com) Key", type="password")
    
    if not openai_api_key:
        st.info("Please enter your OpenAI API key to continue")
        st.stop()
    
    if not oura_token:
        st.info("Please enter your Oura ring token to continue")
        st.stop()

    if not weatherapi_key:
        st.info("Please enter your WeatherAPI key to continue")
        st.stop()

    os.environ["OPENAI_API_KEY"] = openai_api_key
    st.session_state["oura_token"] = oura_token
    st.session_state["weatherapi_key"] = weatherapi_key

def load_participant_data(oura_token):
    """
    Load participant data using Oura ring token.
    Returns True if data loaded successfully, False otherwise.
    """
    # load user information
    st.session_state["user_name"] = st.secrets["user"]["user_name"]
    st.session_state["pid"] = st.secrets["user"]["pid"]

    # Load the recommendation model
    if "recommender_model" not in st.session_state:
        print("loading recommendation model")
        user_model = pickle.load(open(st.secrets["user"]["user_model_path"], "rb"))
        st.session_state["recommender_model"] = user_model

    if oura_token:
        try:
            # Load health data
            if "health_data" not in st.session_state:
                health_data = utils.get_health_data(oura_token, start_date=start_date, end_date=current_day)
                st.session_state["health_data"] = health_data

            # Load personal data
            if "oura_personal_data" not in st.session_state:
                oura_personal_data = utils.get_oura_data(oura_token, "personal_info", start_date=start_date, end_date=current_day)
                # print(f"oura_personal_data: {oura_personal_data}")
                
                st.session_state['age'] = oura_personal_data["age"]
                st.session_state['gender'] = oura_personal_data["biological_sex"]
                st.session_state['weight'] = oura_personal_data["weight"]
                st.session_state['height'] = oura_personal_data["height"]

            return True
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    else:
        st.error("Please provide a valid Oura ring token")
        return False

# Load participant data 
if not load_participant_data(oura_token):
    st.stop()


st.title("🤖 Sleep Health Chatbot")
"""
**Description**
This is a chatbot built to enhance your sleep health. It can answer your questions related to your health and suggest personalized and actionable activities to improve your sleep quality. 
"""

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

# Function to get location with timeout
def get_location_with_timeout(timeout=10):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            location_info = utils.get_location()
            if location_info['latitude'] is not None:
                return location_info
        except Exception:
            pass
        time.sleep(1)
    return None

# Get location and weather information
if "location_info" not in st.session_state:
    with st.spinner('Getting location information...'):
        location_info = get_location_with_timeout()
        # location_info = None
        if location_info:
            lat = location_info['latitude']
            lon = location_info['longitude']
            weather_info = utils.get_weather_nowcast(lat, lon, st.session_state["weatherapi_key"])
            if weather_info:
                st.session_state["location_info"] = location_info
                st.session_state["weather_info"] = weather_info
                st.session_state["timezone"] = weather_info["timezone"]
            else:
                st.error("Could not fetch weather information.")
        else:
            warning = st.empty()
            warning.warning("Could not determine location automatically. Please enter your location manually.")

            # Use a form for manual input
            placeholder = st.empty()
            with placeholder.form("location_form"):
                location_input = st.text_input("Enter your city or postal code:")
                submit_button = st.form_submit_button("Submit")
            if submit_button and location_input:
                try:
                    weather_info, location_info = utils.get_weather_and_location_by_user_input(location_input, st.session_state["weatherapi_key"])
                    if weather_info and location_info:
                        lat = location_info['latitude']
                        lon = location_info['longitude']
                        st.session_state["location_info"] = location_info
                        st.session_state["weather_info"] = weather_info
                        st.session_state["timezone"] = weather_info["timezone"]
                        placeholder.empty()
                        warning.empty()
                    else:
                        st.error("Invalid location. Please try again.")
                except Exception as e:
                    st.error(f"Error processing manual location input: {e}")

# Check if weather_info is initialized before accessing it
if "weather_info" in st.session_state and "location_info" in st.session_state:
    timeCol, tempCol, weatherCol, locCol = st.columns(4)
    timeCol.metric("Time (EST)", st.session_state.weather_info['timeCategory'])
    tempCol.metric("Temperature", f"{st.session_state.weather_info['temperature']} {st.session_state.weather_info['temperatureUnit']}", help=f"{st.session_state.weather_info['temperature']} {st.session_state.weather_info['temperatureUnit']}")
    weatherCol.metric("Weather Forecast", st.session_state.weather_info['weatherCategory'], help=st.session_state.weather_info['shortForecast'])
    locCol.metric("Location", st.session_state.location_info['location'], help=st.session_state.location_info['location'])
else:
    st.warning("Location and weather information not available yet.")

st.session_state["chat_input"] = ""
def update_chat_input(question):
    st.session_state.chat_input = question

expander_label = "**Try some example questions**"
with st.expander(expander_label, expanded=False):
    with st.container():
        # st.subheader('Recommended Questions')
        col1, col2, col3, col4 = st.columns(4)
        if col1.button("What do you recommend me to do?",):
            update_chat_input("What do you recommend me to do?")
        if col2.button('What activities do I normally do?'): # 
            update_chat_input("What activities do I normally do?")
        if col3.button('Are there any noticeable patterns in stress levels?'):
            update_chat_input("Are there any noticeable patterns in stress levels?")
        if col4.button('How is my sleep efficiency for the past a few days?'):
            update_chat_input("How is my sleep efficiency for the past a few days?")
    
        
# Set up the LangChain, passing in Message History
openai_api_key = st.secrets["openai"]["openaikey"]
# st.sidebar.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Enter an OpenAI API Key to continue")
    st.stop()
os.environ["OPENAI_API_KEY"] = openai_api_key


def run_chatbot():
    chat = ChatOpenAI(
        model=MODEL_NAME,
        streaming=True,
        temperature=0
    )
    
    # Use window memory instead of summary buffer
    memory = ConversationBufferWindowMemory(
        return_messages=True,
        memory_key="history",
        k=5  # Keep last 5 conversations
    )
    
    # Set up memory
    if "memory" not in st.session_state:
        st.session_state["memory"] = memory

    sys_message = f"""You are a helpful, compassionate, friendly sleep expert to help users improve sleep quality.
    The user has the following personal profile:
    - Age: {st.session_state.age}
    - Gender: {st.session_state.gender}
    - Weight: {st.session_state.weight}
    - Height: {st.session_state.height}

    In your responses, consider the following dimensions from the Theoretical Domains Framework and Taxonomy of Behaviour Change Techniques, focusing on the most relevant aspects for each interaction:

    1. Consequences and reinforcement: Discuss specific outcomes of sleep behaviors based on the user's data.
    2. Feedback and monitoring: Suggest personalized tracking methods.
    3. Goals: Set clear, achievable sleep objectives based on the user's current data.
    4. Knowledge: Provide tailored information about sleep health.
    5. Environmental context and resources: Address the user's specific sleep environment.
    6. Skills and capabilities: Teach personalized techniques for better sleep.
    7. Emotional support: Providing empathy and encouragement to help users overcome sleep challenges.

    Use your judgment to determine which of these dimensions are most relevant to each user question and the conversation context. You don't need to address all dimensions in every response. Focus on the most pertinent aspects to provide tailored, effective advice.

    Address the user in the second person and tailor your responses to their specific situation, needs, and data from their Oura ring or other sources mentioned in the conversation.
    Keep responses to 2-3 short sentences max.
    """
    initial_message = "Hello! I'm your sleep-enhancing assistant. I can provide tips to help you improve your sleep quality. How can I assist you today?"

    chatbotPrompt = ChatPromptTemplate.from_messages(
        [
            ("system", sys_message),
            ("ai", initial_message),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
            ("human", "Oura data insights: {oura_insights}"),
            ("human", "Selected theoretical dimensions: {selected_dimensions}"),
        ]
    )

    # Initialize the agent coordinator
    if "agent_coordinator" not in st.session_state:
        st.session_state.agent_coordinator = AgentCoordinator()

    # Set up the streamlit message history
    if "msgs" not in st.session_state:
        st.session_state["msgs"] = StreamlitChatMessageHistory(key="langchain_messages")
        st.session_state["msg_added"] = []
        if len(st.session_state.msgs.messages) == 0:
            st.session_state.msgs.add_ai_message(initial_message)
            st.session_state.msg_added.append("no")

    # Set up recommendation record
    if "act_rec" not in st.session_state:
        st.session_state.act_rec = {}

    view_messages = st.expander("View your health data collected by Oura ring")
        
    if "msgs" in st.session_state:
        # Render current messages from StreamlitChatMessageHistory
        for msgIdx, msg in enumerate(st.session_state.msgs.messages):
            st.chat_message(msg.type).write(msg.content)

    # If user inputs a new prompt, generate and draw a new response
    content_inputs = st.chat_input()
    if content_inputs or st.session_state.chat_input:
        contents = st.session_state.chat_input if st.session_state.chat_input else content_inputs
        st.chat_message("human").write(contents)
        st.session_state.msgs.add_user_message(contents)
        st.session_state.msg_added.append("no")
        
        prev_messages = st.session_state.memory.chat_memory.messages
        
        with st.chat_message("ai"):
            # Prepare context for agent coordinator
            conversation_history = ""
            if prev_messages:
                conversation_history = "\n".join([
                    f"{msg.type}: {msg.content}" 
                    for msg in prev_messages[-5:]  # Only use last 5 messages for context
                ])
            
            context = {
                "query": contents,
                "conversation_context": conversation_history if conversation_history else "none",
                "health_data": st.session_state.health_data,
                "memory": st.session_state.memory,
                "chat_prompt": chatbotPrompt,
                "chat": chat,
                "bedtime": st.session_state.health_data["bedtime_start"][st.session_state.health_data["sleep_type (long_sleep: 3+ hours, sleep: naps <3 hours)"] == "long_sleep"].tolist()[-10:],
                "time": st.session_state.weather_info["timestamp"],
                "temperature": f"{st.session_state.weather_info['temperature']} {st.session_state.weather_info['temperatureUnit']}",
                "weather": st.session_state.weather_info['weatherCategory'],
                "location": st.session_state.location_info['location'],
                "recommendations": st.session_state.recommender_model.predict([
                    [st.session_state.weather_info['timeCategory'],
                     st.session_state.weather_info['temperatureCategory'].lower(),
                     st.session_state.weather_info['weatherCategory'].lower()]
                ])
            }

            # Process query through agent coordinator
            response_generator = st.session_state.agent_coordinator.process_query(contents, context)
            
            # Handle both stream and string responses
            if hasattr(response_generator, '__iter__') and not isinstance(response_generator, str):
                response = st.write_stream(response_generator)
            else:
                response = response_generator
                st.write(response)

            # Save context and update message history
            st.session_state.memory.save_context({"input": contents}, {"output": response})
            st.session_state.msgs.add_ai_message(response)
            st.session_state.msg_added.append("no")
            
            # Save action recommendations if it was a recommendation query
            query_type = st.session_state.agent_coordinator.classify_query(contents, context["conversation_context"])
            if query_type == "recommendation":
                msg_len = len(st.session_state.msgs.messages)
                st.session_state.act_rec[str(msg_len - 1)] = {
                    "age": st.session_state.age,
                    "gender": st.session_state.gender,
                    "weight": st.session_state.weight,
                    "height": st.session_state.height,
                    "time": st.session_state.weather_info['timeCategory'],
                    "temperature": st.session_state.weather_info['temperatureCategory'],
                    "weather": st.session_state.weather_info['weatherCategory'],
                    "location": st.session_state.location_info['location'],
                    "recommendations": context["recommendations"],
                }
    
    # Draw the messages at the end, so newly generated ones show up immediately
    with view_messages:
        """
        Health data:
        """
        if "health_data" in st.session_state:
            st.dataframe(st.session_state.health_data)

if "weather_info" in st.session_state and "location_info" in st.session_state:
    run_chatbot()