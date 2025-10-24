# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import mimetypes
import os
from google import genai
from google.genai import types


def save_binary_file(file_name, data):
    f = open(file_name, "wb")
    f.write(data)
    f.close()
    print(f"\nFile saved to: {file_name}")


def generate():
    # Ensure your GEMINI_API_KEY is set in your environment variables
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.5-flash-image"
    
    # ----------------------------------------------------
    # 1. INSERT THE USER'S TEXT PROMPT HERE
    # ----------------------------------------------------
    user_prompt = "I want to start a vegan diet. Give me 3 tips for muscle growth and generate an image of a simple, high-protein vegan meal."

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=user_prompt),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        # Request both IMAGE and TEXT modalities in the response
        response_modalities=[
            "IMAGE",
            "TEXT",
        ],
        system_instruction=[
            types.Part.from_text(text="""you are a fitness trainer named FitBot, your task is to help people by clarifying their doubts regarding fitness and diet. Provide clear, actionable advice and encouraging feedback."""),
        ],
    )

    file_index = 0
    print("FitBot is processing your request...\n")
    
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if (
            chunk.candidates is None
            or chunk.candidates[0].content is None
            or chunk.candidates[0].content.parts is None
        ):
            continue
        
        # Check if the part contains image data (inline_data)
        if chunk.candidates[0].content.parts[0].inline_data and chunk.candidates[0].content.parts[0].inline_data.data:
            # ----------------------------------------------------
            # 2. FILE NAME PREFIX INSERTION
            # ----------------------------------------------------
            file_name_prefix = f"vegan_meal_image"
            
            inline_data = chunk.candidates[0].content.parts[0].inline_data
            data_buffer = inline_data.data
            
            # Guess the file extension (e.g., .jpeg, .png) from the MIME type
            file_extension = mimetypes.guess_extension(inline_data.mime_type)
            
            final_file_name = f"{file_name_prefix}_{file_index}{file_extension}"
            save_binary_file(final_file_name, data_buffer)
            file_index += 1
            
        else:
            # Print text chunks as they arrive
            print(chunk.text, end="", flush=True)

if __name__ == "__main__":
    generate()