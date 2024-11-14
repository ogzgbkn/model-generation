from openai import OpenAI

class ChatGPT:
    def __init__(self, api_key = None):
        if not api_key:
            raise Exception("ChatGPT API key is not provided!")
        self.client = OpenAI(api_key = api_key)

    def chat_with_gpt(self, prompt):
        try:
            # Make a request to the OpenAI API
            response = self.client.chat.completions.create(
                model = "gpt-4o-mini",  # You can specify "gpt-3.5-turbo" or another model if needed
                messages = [{"role": "user", "content": prompt}],
                temperature = 0.7
            )
            # Extract the response content
            return response['choices'][0]['message']['content']
        
        except Exception as e:
            print(f"Error: {e}")
            return None
