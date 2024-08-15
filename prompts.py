from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("""
I want to preprocess a query so that it contains only necessary information only. 
Remove any Personal Identifiable Information (PII) such as Name, Birthdays, or Passport Numbers.
If in another language, Translate to English.

# Example 1
Original Query: Hi, My name is Jane Doe. My passport number is PE1210432. I lost my passport when I was walking at the park. What should I do?
Preprocessed Query: "I lost my passport, what should I do"?

# Example 2
Original Query: Hi. I want to ask what are the requirements for applying for a passport? Thank you very much.
Preprocessed Query: "What are the requirements for applying for a passport?"

# Example 3
Original Query: Hi, My name is Jane Doe. Nawala ang aking passport. Ano ang kailangan kong gawin?
Preprocessed Query: "I lost my passport, what should I do"?

Do the same for this last query. Preprocess the following query, return plain text only:
Original Query: ```{question}```
(NO CODE) Preprocessed Query:""")


dfa_rag_prompt = PromptTemplate.from_template("""
You are an assistant for the Department of Foreign Affairs.
I need you to address the question: {question} 
Possibly relevant context was retrieved here: 
([FAQ] is just a separator) [Start of Context FAQs]{context}[End of Context FAQs] 
                                              
### Main Instruction: Address the following question in triple backticks: 
```{question}``` 
                                              
### Other Instructions: Answer must strictly be based on the context only. 
Just say 'I cannot answer your question.' if no answer from context. 
Please return strictly as plain text and not markdown format. 
The 'context' is internal, do not mention it's in the answer, give an answer as if you are the source of information. 
Please give a detailed answer. Provide instructions or links that exist in the context if needed. 

Answer:

""")

# rag_prompt1 = PromptTemplate.from_template(template="{question}"])


prompt2 = PromptTemplate.from_template("""
I want to preprocess a query so that it contains only necessary information only. 
Remove any Personal Identifiable Information (PII) such as Name, Birthdays, or Passport Numbers.
If in another language, Translate to English.
Separate each query.

# Example 1
Original Query: Hi, My name is Jane Doe. My passport number is PE1210432. I lost my passport when I was walking at the park. What should I do? What is a Diplomatic e-Passport?
Preprocessed Query: ["I lost my passport, what should I do?", "What is a Diplomatic e-Passport?"]

# Example 2
Original Query: Hi. I want to ask what are the requirements for applying for a passport? Thank you very much.
Preprocessed Query: ["What are the requirements for applying for a passport?"]

# Example 3
Original Query: Hi, My name is Jane Doe. Nawala ang aking passport. Ano ang kailangan kong gawin? 
Preprocessed Query: ["I lost my passport, what should I do"?]

Do the same for this last query. Preprocess the following query, return plain as python list:
Original Query: ```{query}```
(NO CODE) Preprocessed Query:""")