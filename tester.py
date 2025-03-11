import openai
import streamlit as st
from supabase import create_client
import logging
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Supabase client using Streamlit secrets
def init_clients():
    try:
        # Fetch Supabase credentials from st.secrets
        supabase_url = st.secrets["supabase"]["url"]
        supabase_key = st.secrets["supabase"]["key"]
        openai.api_key = st.secrets["openai"]["api_key"]  # Fetch OpenAI API Key from secrets
        
        # Initialize Supabase client
        supabase = create_client(supabase_url, supabase_key)
        return supabase
    except Exception as e:
        logger.error(f"Initialization error: {str(e)}")
        return None

supabase = init_clients()

# Function to generate query embeddings using OpenAI
def generate_query_embedding(query_text):
    try:
        embedding = openai.Embedding.create(input=query_text, model="text-embedding-ada-002")
        return embedding['data'][0]['embedding']
    except Exception as e:
        logger.error(f"Error generating query embedding: {str(e)}")
        return None

# Function to get related reports from Supabase
def get_contextual_data(query_embedding):
    try:
        response = supabase.rpc('find_related_reports', {'question_vector': query_embedding}).execute()
        if response.data:
            return response.data
        else:
            logger.warning("No related reports found for the given query embedding")
            return []
    except Exception as e:
        logger.error(f"Context data error: {str(e)}")
        return []

# Function to check for similar past feedback
def get_similar_feedback(query_embedding):
    try:
        response = supabase.rpc('find_similar_feedback', {'query_vector': query_embedding}).execute()
        if response.data:
            return response.data
        else:
            logger.warning("No similar feedback found for the given query embedding")
            return []
    except Exception as e:
        logger.error(f"Error in fetching similar feedback: {str(e)}")
        return []

# Function to generate AI response based on context and similar feedback
def generate_ai_response(user_query, similar_feedback, context_data):
    try:
        combined_context = " ".join([feedback['ai_response'] for feedback in similar_feedback]) + "\n" + "\n".join([item['report_text'] for item in context_data])

        # Initialize ChatOpenAI with the correct API key
        llm = ChatOpenAI(
            temperature=0.3, 
            model="gpt-4",
            openai_api_key=st.secrets["openai"]["api_key"]  # Pass the OpenAI API key from Streamlit secrets
        )

        prompt_template = """
        As a CFO assistant, analyze the financial data and context provided to answer the user's question.

        Context from similar past feedback and reports:
        {context}

        Question: {question}

        Please provide a clear, concise, and human-friendly response, avoiding jargon.
        """

        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = LLMChain(llm=llm, prompt=prompt)

        response = chain.run({
            'context': combined_context,
            'question': user_query
        })

        return response
    except Exception as e:
        logger.error(f"AI response generation error: {str(e)}")
        return "Unable to generate a response."

# Save feedback function
def store_feedback(user_query, ai_response, human_feedback, corrected_response):
    try:
        # Log the feedback to check arguments
        logger.info(f"Storing feedback with parameters: {user_query}, {ai_response}, {human_feedback}, {corrected_response}")
        
        # Generate embedding for the corrected response
        feedback_embedding = openai.Embedding.create(input=str(corrected_response), model="text-embedding-ada-002").data[0].embedding
        
        # Insert feedback into Supabase
        response = supabase.rpc('insert_feedback', {
            'user_question': user_query,
            'ai_response': ai_response,
            'human_feedback': human_feedback,
            'corrected_response': corrected_response,
            'feedback_embedding': feedback_embedding
        }).execute()

        if response.status_code == 204:
            logger.info("Feedback stored successfully")
            return True
        else:
            logger.error(f"Failed to store feedback: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Feedback storage error: {str(e)}")
        return False

# Function to handle session state for storing past questions and answers
def manage_session_memory(user_query, ai_response):
    if "history" not in st.session_state:
        st.session_state.history = []
    
    st.session_state.history.append({"question": user_query, "answer": ai_response})

# Streamlit UI
st.set_page_config(page_title="CFO AI Assistant", page_icon="💼", layout="wide")
st.title("CFO Expertise AI - Financial Insights Assistant")

# Display user input box for financial questions
question = st.text_input("Ask your financial question:", "")
if question:
    st.session_state.previous_question = question  # Store question for future reference

    with st.spinner("Analyzing..."):
        # Generate embedding for the question
        query_embedding = generate_query_embedding(question)
        
        # Retrieve similar feedback and related reports
        similar_feedback = get_similar_feedback(query_embedding)
        context_data = get_contextual_data(query_embedding)
        
        # Generate AI response
        ai_response = generate_ai_response(question, similar_feedback, context_data)
        st.write(f"AI Response: {ai_response}")

        # Store AI response in session history
        manage_session_memory(question, ai_response)

    # Ask for feedback on the answer
    feedback = st.radio("Is the AI's answer correct?", ("Yes", "No"))

    if feedback == "Yes":
        st.success("Thank you for confirming! The answer has been accepted.")
    else:
        # If the feedback is "No", ask for the corrected response
        corrected_response = st.text_area("Please provide the corrected answer:")

        if corrected_response:
            # Validate the corrected response (using financial metrics)
            st.write(f"Validating the corrected response: {corrected_response}")
            store_feedback(question, ai_response, feedback, corrected_response)
            st.success("Your correction has been stored!")

# Show the session history (optional for debugging or tracking)
if "history" in st.session_state:
    st.write("Session History:")
    for entry in st.session_state.history:
        st.write(f"Q: {entry['question']} | A: {entry['answer']}")
