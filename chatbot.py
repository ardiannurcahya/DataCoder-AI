import re
import streamlit as st
from llm_models import text_llm
from prompt_models import prompt_chatbot


def init_state():
    """Initialize chatbot state in session."""
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = {
            'chat_history': [],
            'prev_df': None,
            'prev_raw_dataframes': None,
        }


def handle_user_query():
    """Render chat history and handle new user queries."""
    st.header("💬 Chatbot Assistant")

    if st.button("🗑️ Clear all chat", key="clear_all_chat"):
        st.session_state.chatbot['chat_history'] = []

    for i in range(0, len(st.session_state.chatbot['chat_history']), 2):
        user_msg = st.session_state.chatbot['chat_history'][i]
        assistant_msg = (
            st.session_state.chatbot['chat_history'][i + 1]
            if i + 1 < len(st.session_state.chatbot['chat_history'])
            else None
        )
        with st.expander(f"Chat {i//2 + 1}", expanded=False):
            with st.chat_message("user"):
                st.markdown(user_msg["content"])
            if assistant_msg:
                with st.chat_message("assistant"):
                    st.markdown(assistant_msg["content"])

    user_query = st.chat_input("Ask coding questions...", key="chatbot_input")

    if user_query:
        st.session_state.chatbot['prev_df'] = st.session_state.get('df')
        st.session_state.chatbot['prev_raw_dataframes'] = st.session_state.get('raw_dataframes')

        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chatbot['chat_history'].append({"role": "user", "content": user_query})

        with st.spinner("Thinking..."):
            if st.session_state.get("df") is not None:
                try:
                    df = st.session_state.df
                    response = text_llm(
                        prompt_chatbot(
                            user_query,
                            df.columns,
                            df.shape,
                            df.dtypes,
                        )
                    )
                    clean_response = re.sub(
                        r'<\s*think\s*>.*?<\s*/\s*think\s*>',
                        '',
                        response,
                        flags=re.DOTALL | re.IGNORECASE,
                    )
                    with st.chat_message("assistant"):
                        st.markdown(clean_response)
                    st.session_state.chatbot['chat_history'].append(
                        {"role": "assistant", "content": clean_response}
                    )
                except AttributeError as e:
                    st.error(f"Please confirm dataset before starting the chat: {e}")
            else:
                st.write('You need to upload the data first')
