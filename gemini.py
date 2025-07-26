from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

client = genai.Client(api_key="AIzaSyA0UE4rh5PCyw_HEmHDeZ3aEVAx85TfmGA")


def read_image_bytes(path, mime):
    with open(path, 'rb') as f:
        return types.Part.from_bytes(data=f.read(), mime_type=mime)

image1_part = read_image_bytes("./data/demo/deraining/1.jpg", "image/jpeg")
image2_part = read_image_bytes("./data/demo/deraining/1-derain.jpg", "image/jpeg")
image3_part = read_image_bytes("./data/demo/removal/2.png", "image/jpeg")

prompt_text = "Images 1 and 1-derain teach you what's the task of deraining. There's another task related to deraining. \
    Inspired by image deraining, where localized high-frequency degradations are removed to restore scene consistency, we consider a parallel challenge — \
    restoring regions affected by low-frequency, spatially coherent illumination degradations. \
    These regions, akin to 'inverse rain streaks', require careful modeling of light attenuation and shadow boundaries to achieve consistent visual restoration. \
    Please do visual in-context learning and output the described task's result of the image 2.png"

contents = [
    types.Part(text=prompt_text),
    image1_part,
    image2_part,
    image3_part
]

response = client.models.generate_content(
    model="gemini-2.0-flash-preview-image-generation",
    contents=contents,
    config=types.GenerateContentConfig(
        response_modalities=['TEXT', 'IMAGE']
    )
)

for part in response.candidates[0].content.parts:
    if part.text:
        print(part.text)
    elif part.inline_data:
        image = Image.open(BytesIO(part.inline_data.data))
        image.save("output.png")
