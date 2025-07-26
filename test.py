from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

import PIL.Image

client = genai.Client(api_key="AIzaSyA0UE4rh5PCyw_HEmHDeZ3aEVAx85TfmGA")

image = PIL.Image.open('data/demo/lighting/2.png')
text_input = ('Hi, This is a picture. Can you output a light enhanced image of it to relight it?',)

response = client.models.generate_content(
    model="gemini-2.0-flash-preview-image-generation",
    contents=[text_input, image],
    config=types.GenerateContentConfig(
      response_modalities=['TEXT', 'IMAGE']
    )
)

for part in response.candidates[0].content.parts:
  if part.text is not None:
    print(part.text)
  elif part.inline_data is not None:
    image = Image.open(BytesIO((part.inline_data.data)))
    image.show()
    image.save('./output.png')
