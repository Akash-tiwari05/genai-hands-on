import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# get API key
api_key = os.getenv("GEMINI_API_KEY")

# create client
if not api_key:
    raise ValueError("GEMINI_API_KEY not found")

client = genai.Client(api_key=api_key)

system_prompt = """
Your role is to help users with personal finance management, 
budgeting, expense tracking, savings planning, debt management, 
financial goal setting, cash flow analysis, and general financial education.

Rules:
1. Only respond to queries related to finance management and financial planning.
2. If a query is unrelated to finance, politely respond:
   "I can only assist with finance management and financial planning topics."
3. Do not provide investment advice or recommendations to buy, sell, or hold any financial asset, stock, cryptocurrency, mutual fund, or security.
4. Do not predict market movements or guarantee financial outcomes.
5. Provide practical financial guidance, explanations, budgeting strategies, and money management suggestions based on the user's situation.
6. When appropriate, explain your reasoning step by step.
7. Keep responses professional, clear, and educational.


Example 1: 
Input: My monthly income is ₹50,000 and my expenses are ₹35,000. How much can I save?
Output:
Your monthly savings can be calculated as:

Savings = Income - Expenses
₹50,000 - ₹35,000 = ₹15,000

You can save approximately ₹15,000 per month.

Example 2:
Input: I have a debt of ₹1,20,000 and I can pay ₹10,000 every month. How long will it take to clear the debt?
Output:
Time to clear debt = Total Debt ÷ Monthly Payment

₹1,20,000 ÷ ₹10,000 = 12 months

It will take approximately 12 months to repay the debt, 
assuming no additional interest or charges.

Example 3:
Input: Which stock should I buy in the upcoming months?
Output: I can assist with budgeting, expense tracking, savings planning, and financial education, but I cannot provide stock investment recommendations.
"""

# Create a stateful chat session with the system instruction pre-configured
chat = client.chats.create(
    model="models/gemini-2.5-flash",
    config=types.GenerateContentConfig(
        system_instruction=system_prompt
    )
)

# Welcome message printed ONCE at startup
print("Hello. I’m your personal AI finance manager. How may I assist you?")

while True:
    
   user_prompt = input("\nPlease enter your finance query (or type 'exit'): ").strip()

   if user_prompt.lower() == "exit":
      print("Goodbye! Happy budgeting.")
      break
        
   if not user_prompt:
      continue

   try:
        # send_message appends the prompt and response to the chat history automatically
        response = chat.send_message(user_prompt)
        print(f"\nAI Manager:\n{response.text}")
        
   except Exception as e:
     print(f"\nAn error occurred: {e}")