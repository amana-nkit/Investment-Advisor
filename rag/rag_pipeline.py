from langchain_openai import ChatOpenAI
from Prompts.Prompts import proposal_prompt
#from prompts.prompts import proposal_prompt

def generate_proposal(context):

    # docs = vector_db.similarity_search(
    #     "investment strategy mutual funds risk",
    #     k=5
    # )

    # context = "\n\n".join([d.page_content for d in docs])

    llm = ChatOpenAI(model="gpt-4.1")

    chain = proposal_prompt | llm

    response = chain.invoke({"context": context})

    return response.content