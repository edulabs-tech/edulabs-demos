import pprint

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()


llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

json_schema = {
    "title": "person",
    "description": "person profile",
    "type": "object",
    "properties": {
        "first_name": {
            "type": "string",
            "description": "The first name of the person",
            "default": ""
        },
        "last_name": {
            "type": "string",
            "description": "The last name of the person",
        },
        "birth_date": {
            "type": "string",
            "description": "the birth date of the person"
        },
        "birth_year": {
            "type": "number",
            "description": "the birth year of the person",
            "default": None,
        },
        "companies": {
            "type": "array",
            "description": "a list of companies the person associated with",
            "items": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "the company the person works at"
                    },
                    "role": {
                        "type": "string",
                        "description": "the role of the person in the company"
                    },
                    "industry": {
                        "type": "string",
                        "description": "the industry of the company",
                        "enum": ["automotive", "healthcare", "retail", "software", "governmental", "hardware"]
                    }
                }
            }
        }
    },
    "required": ["last_name", "first_name"],
}
structured_llm = llm.with_structured_output(json_schema)

text = """
Elon Reeve Musk (born June 28, 1971) is a businessman known 
for his key roles in the space company SpaceX and the 
automotive company Tesla, Inc. 
His other involvements include ownership of X Corp., 
the company that operates the social media platform X (formerly Twitter), 
and his role in the founding of the Boring Company, xAI, Neuralink, and OpenAI. 
Musk is the wealthiest individual in the world; 
as of December 2024, Forbes estimates his net worth to be US$432 billion.
"""
result = llm.with_structured_output(json_schema).invoke(text)
pprint.pprint(result)
pprint.pprint(type(result[0]['args']))

