import os
from pprint import pprint

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()


genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

model = genai.GenerativeModel("models/gemini-1.5-flash")

prompt = "The quick brown fox jumps over the lazy dog."
prompt = "לידיעתך"
prompt = "generalization"
prompt = "nkdlzft845hltgunvls95tg"

response = model.count_tokens(prompt)

# Call `count_tokens` to get the input token count (`total_tokens`).
# print("total_tokens: ", )
# ( total_tokens: 10 )

# response = model.generate_content(prompt)

# On the response for `generate_content`, use `usage_metadata`
# to get separate input and output token counts
# (`prompt_token_count` and `candidates_token_count`, respectively),
# as well as the combined token count (`total_token_count`).
# print(response.usage_metadata)
pprint(response)
# ( prompt_token_count: 11, candidates_token_count: 73, total_token_count: 84 )