# small chatbot to try 
from dotenv import load_dotenv
import openai

# Set your OpenAI API key
# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key from the environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

# Function to generate chat completion
def generate_chat_completion(prompt, model="gpt-3.5-turbo", temperature=0.7, max_tokens=150):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
            stop=None
        )
        # Extracting and returning the text from the response
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Example usage
if __name__ == "__main__":
    prompt = "What is the capital of France?"
    completion = generate_chat_completion(prompt)
    print(completion)
