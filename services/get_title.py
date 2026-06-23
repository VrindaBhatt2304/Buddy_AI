from llama_index.core import PromptTemplate

from llm_factory.get_llm import get_ollama_llm

def get_chat_title(model, user_query):
    llm = get_ollama_llm(model)
    title_prompt_template = ("You are a helpful assistant that generates short, clear, and catchy titles. \n\n"
                             "Task: \n- Read the given user query.\n- Create a concise title (max 7 words). \n"
                             "- The title should summarize the intent of the query. \n"
                             "- Avoid unnecessary words, punctuation, or filter. \n"
                             "- Keep it professional and easy to understand. \n\n"
                             "User Query:\n{user_query}\n\n"
                             "Output:\nTitle:")
    
    title_prompt = PromptTemplate(title_prompt_template).format(user_query=user_query)
    title = llm.complete(prompt=title_prompt).text
    return title

# Example usage
#model = "llama3.2:latest"
#user_query = " Can you explain the concept of reinforcement learning and it's application in modern AI"
#title = get_chat_title(model, user_query)
#print(title)
