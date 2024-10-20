from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
import ollama
import os
import json

def online_rag(query: str)-> str:
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ["LANGCHAIN_PROJECT"] = "L3 Research Agent"

    wrapper = DuckDuckGoSearchAPIWrapper(max_results=25)
    web_search_tool = DuckDuckGoSearchRun(api_wrapper=wrapper)

    web_results = web_search_tool.invoke(query)
    template=f"""
    
    <|begin_of_text|>
    
    <|start_header_id|>system<|end_header_id|> 
    
    You are an AI assistant for Research Question Tasks, that synthesizes web search results. 
    Strictly use the following pieces of web search context to answer the question. If you don't know the answer, just say that you don't know. 
    keep the answer concise, but provide all of the details you can in the form of a research report. 
    Only make direct references to material if provided in the context.
    
    <|eot_id|>
    
    <|start_header_id|>user<|end_header_id|>
    
    Question: {query} 
    Web Search Context: {web_results} 
    Answer: 
    
    <|eot_id|>
    
    <|start_header_id|>assistant<|end_header_id|>"""

    response = ollama.chat(model="llama3.1",
                            messages=[{"role": "user",
                                        "content": template}])
    
    return response["message"]["content"]

def chatbot_response(query):
    response = ollama.chat(model="llama3.1", 
                           messages=[{
                               "role": "user",
                               "content": query
                           }])
    
    return response["message"]["content"]

def decider_agent(query: str)-> json:
    response = ollama.chat(
        model='llama3.2',
        messages=[{'role': 'user', 'content': query}],

        tools=[
        {
            "type": "function",
            "function": {
                "name": "online_rag",
                "description": "best for queries that requires realtime data from internet as this function scrapes online data and returns an output based on that data",
                "parameters": {
                "type": "object",
                "properties": {},
                "required": [query]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "chatbot_response",
                "description": "best for queries that requires factual data or pretrained data to an ai model before july 2024",
                "parameters": {
                "type": "object",
                "properties": {},
                "required": [query]
                }
            }
        },
        ]
    )

    return response["message"]

if __name__ == "__main__":
    
    resp = online_rag("how did ratan tata die")
    print(resp)