import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMChain, LLMMathChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import initialize_agent
from langchain.tools import Tool
from langchain.callbacks import StreamlitCallbackHandler

# Setup Streamlit app
st.set_page_config(page_title='Maths Problem Solver')
st.title('Maths Problem Solver using Google Gemma2 model')

# Sidebar: API key
groq_api_key = st.sidebar.text_input('Insert Groq API key', type='password')
st.secrets['groq_api_key']

if not groq_api_key:
    st.info('Please add the Groq API key')
    st.stop()

# Initialize LLM
llm = ChatGroq(model='gemma2-9b-it', groq_api_key=groq_api_key)

# Initialize tools
wiki_wrapper = WikipediaAPIWrapper()
mathematical_chain = LLMMathChain.from_llm(llm=llm)

prompt = '''
You are an agent to solve the mathematical question.
Logically arrive at the solution and give the detailed explanation pointwise.

Question: {question}
Answer:
'''
prompt_template = PromptTemplate(input_variables=['question'], template=prompt)
chain = LLMChain(llm=llm, prompt=prompt_template)

# Tools using Tool class
wiki_tool = Tool(
    name="wikipedia",
    func=wiki_wrapper.run,
    description="Search Wikipedia for information."
)

calculator_tool = Tool(
    name="calculator",
    func=mathematical_chain.run,
    description="Solve mathematical expressions."
)

reasoning_tool = Tool(
    name="reasoning",
    func=chain.run,
    description="Answer logic-based and reasoning questions."
)

# Initialize agent
assistant_agent = initialize_agent(
    tools=[wiki_tool, calculator_tool, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

# Initialize session
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {'role': 'assistant', 'content': 'Hello! I am your mathematical assistant.'}
    ]

# Display chat history
for msg in st.session_state['messages']:
    st.chat_message(msg['role']).write(msg['content'])

# Input & button
question = st.text_area('Ask a Question', 'What is 10 percent of 100?')

if st.button('Find Answer'):
    if question:
        with st.spinner('Generating answer...'):
            st.session_state['messages'].append({'role': 'user', 'content': question})
            st.chat_message('user').write(question)

            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

            try:
                response = assistant_agent.run(question, callbacks=[st_cb])
            except Exception as e:
                response = f"⚠️ Error: {e}"

            st.session_state['messages'].append({'role': 'assistant', 'content': response})
            st.chat_message('assistant').write(response)
    else:
        st.error('Please enter a question')

