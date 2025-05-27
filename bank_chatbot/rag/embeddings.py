import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

example_text = """Dogs are loyal, intelligent companions known for their playful energy and unconditional love."""

embedding_model = OpenAIEmbeddings()
# embedding = embedding_model.embed_query(example_text)
# print(f"Example embedding for text {example_text}:\n{embedding}")
# print(len(embedding))



# p1 = embedding_model.embed_query("I adopted a small dog from the animal shelter.")
# p2 = embedding_model.embed_query("I got a little puppy from the rescue center.")

# p1 = embedding_model.embed_query("The stock market crashed during the economic crisis.")
# p2 = embedding_model.embed_query("I took my dog for a walk in the park this morning.")

# p1 = embedding_model.embed_query("The stock market crashed during the economic crisis.")
# p2 = embedding_model.embed_query("""מה קורה איש?""")


# cosine_similarity = np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))
# print(cosine_similarity)
