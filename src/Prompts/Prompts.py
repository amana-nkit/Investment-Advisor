from langchain_core.prompts import PromptTemplate

proposal_prompt = PromptTemplate.from_template("""
# You are a senior wealth advisor.

# Use customer data + market insights to create a professional proposal.

# Context:
# {context}

# Output:
# 1. Profile Summary
# 2. Financial Analysis
# 3. Risk Level
# 4. Strategy
# 5. Allocation
# 6. Products
# 7. Returns
# 8. Risk Mitigation
# 9. Conclusion

# Keep it realistic and client-ready.
                                               
You are a senior wealth manager.

Use BOTH:
1. Customer financial data (priority)
2. Market insights

If customer data is missing, infer reasonably but avoid placeholders.

Context:
{context}

Generate a complete professional investment proposal with actual usable values where possible.

""")