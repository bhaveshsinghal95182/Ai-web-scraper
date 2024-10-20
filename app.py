from online_rag import online_rag, decider_agent, chatbot_response
import streamlit as st

st.set_page_config(page_title="A chatbot with search capabilities", layout="centered")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

if prompt := st.chat_input():
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    response_tool = decider_agent(prompt)
    if response_tool.get("tool_calls"):
        available_functions = {
            "online_rag": online_rag,
            "chatbot_response": chatbot_response
        }

        for tool in response_tool["tool_calls"]:
            function_to_call = available_functions[tool["function"]["name"]]

            if function_to_call == online_rag:
                answer = online_rag(prompt)
            elif function_to_call == chatbot_response:
                answer = chatbot_response(prompt)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)