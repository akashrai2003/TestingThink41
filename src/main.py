import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
from groq import Groq
import requests

app = FastAPI()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise Exception("Please set the GROQ_API_KEY environment variable")

client = Groq(api_key=api_key)

class Query(BaseModel):
    text: str

MAX_INPUT_LENGTH = 1000  

@app.post("/chat")
async def chat(query: Query):
    '''
    Chat endpoint that takes a text input and returns a response from the Groq API
    input: query (Query) - text input
    output: response (str) - response from Groq API

    '''
    # Check for empty input
    if not query.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    # Check for input length
    if len(query.text) > MAX_INPUT_LENGTH:
        raise HTTPException(status_code=400, detail=f"Input text exceeds maximum length of {MAX_INPUT_LENGTH} characters")

    try:
        response = get_groq_response(query.text)
        if response is None:
            raise HTTPException(status_code=500, detail="Error getting response from Groq API")
        return {"response": response}
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input data: {e.errors()}")
    except HTTPException as e:
        raise e  
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_groq_response(text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                '''
                You are a helpful assistant that answers only medical queries. If you answer anything except medical queries you'll be fired
                If the user's query is not related to medical topics, respond with:  
                'This query is not related to medical topics. Please ask a medical question.'
                And terminate the answer after this one line. Don't generate any other things or you'll be fired immediately.'''
            )
        },
        {
            "role": "user",
            "content": text,
        }
    ]
    response_content = ""
    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="mixtral-8x7b-32768",
            temperature=0.5,
            max_tokens=1024,
            top_p=1,
            stop=None,
            stream=False,
        )
        response_content = chat_completion.choices[0].message.content
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail="Error communicating with Groq API")
    except KeyError as e:
        raise HTTPException(status_code=500, detail="Unexpected response format from Groq API")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return response_content

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
