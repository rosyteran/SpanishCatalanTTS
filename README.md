# SpanishCatalanTTS

## API Docs

### StyleEncoder API

This API is used to obtain the reference style tensor for a given audio file. The reference style tensor is used to control the style of the synthesized speech. This API call is optional, and you can directly use the StyleTTS2Catalan/StyleTTS2Spanish API to synthesize speech without using the reference style tensor. However, using the reference style tensor can help you control the style of the synthesized speech.

#### Endpoint
- `POST /predictions/STTS2Encoder` (Catalan) OR
- `POST /predictions/STTS2SpanishEncoder` (Spanish)

#### Request
e.g.
- **URL**: `{inference api url}/predictions/STTS2Encoder` # Replace `{inference api url}` with the Inference API URL (e.g., `http://localhost:8080`, `https://example.com`)
- **Method**: `POST`
- **Headers**: None required
- **Body**: Multipart form data containing the audio file
    - `audio`: The audio file to be processed (e.g., `maria_v2.mp3`)

#### Example Request
```python
import requests
import os
import time

# Define the URL
url = "http://127.0.0.1:8080/predictions/STTS2Encoder"

# Define the file path (assuming the script is running in the same directory as the audio file)
file_path = os.path.join(os.getcwd(), "maria_v2.mp3")

# Open the file in binary mode and send as part of the request
with open(file_path, 'rb') as audio_file:
        files = {'audio': audio_file}
        start_time = time.time()
        response = requests.post(url, files=files)
        print(f"Time Taken", time.time() - start_time)

# Print the response (optional, for debugging)
print(response.status_code)
print(response.json())
ref_s = response.json()['ref_s']
hash_s = response.json()['hash']
```

#### Response
- **Status Code**: `200 OK`
- **Body**: JSON object containing the following fields:
    - `ref_s`: Reference Audio's Style Tensor (The Embeddings of the Speaker's Voice)
    - `hash`: Hash string (Can be reused to save Recomputing Time and Network Overhead)

#### Example Response
```json
{
    "ref_s": [[0.07641012966632843, -0.17390969395637512, 0.0005185827612876892, ... ]],
    "hash": "9ef0e0e7887193a4587443fcc2fec940"
}
```

### StyleTTS2Catalan API

This API is used to synthesize speech in Catalan using the StyleTTS2 model. The model can be controlled using the reference style tensor obtained from the StyleEncoder API. If you don't provide a reference style tensor, the model will use a default reference style tensor. You can also control the interpolation between the reference style tensor and the predicted style tensor using the `alpha` and `beta` parameters. The `alpha` parameter determines how much of the reference style tensor to use, while the `beta` parameter determines how much of the predicted style tensor to use. The `diffusion_steps` parameter determines the number of diffusion steps to use in the sampling process. Increasing this value can make the speech more clear but will also increase the time taken to synthesize the speech. The `embedding_scale` parameter determines the speech expressiveness. Increasing this value makes the speech more expressive but will also increase the time taken to synthesize the speech.

#### Endpoint
- `POST /predictions/STTS2Catalan` (Catalan TTS)

#### Request
e.g.
- **URL**: `{inference api url}/predictions/STTS2Catalan` # Replace `{inference api url}` with the Inference API URL (e.g., `http://localhost:8080`, `https://example.com`)
- **Method**: `POST`
- **Headers**: `Content-Type: application/json`
- **Body**: JSON object containing the following fields:
    - `text`: The text to be synthesized.
    - `ref_s`: (Optional) The reference style tensor obtained from the StyleEncoder API.
    - `hash`: (Optional) The hash string obtained from the StyleEncoder API.
    - `alpha`: (Optional) Interpolation factor for the reference style tensor. Default is `0.3`.
    - `beta`: (Optional) Interpolation factor for the predicted style tensor. Default is `0.7`.
    - `diffusion_steps`: (Optional) Number of diffusion steps. Default is `5`. 
    - `embedding_scale`: (Optional)Speech Expressiveness. Default is `1`. 

> Note: If hash is provided, the reference style tensor will be computed using the hash string. If hash is not provided, the reference style tensor `ref_s` will be used directly. If both `ref_s` and `hash` are provided, `hash` will be used. If neither `ref_s` nor `hash` is provided, the model will use the default reference style tensor.

#### Example Request
```python
import requests
import json
import time
from IPython.display import Audio

# Define the URL
url = "http://127.0.0.1:8080/predictions/STTS2Catalan"

# Define the JSON data
data = {
        "text": "Quan vaig tornar al poble, aquell estiu, i vaig veure els carrers plens de gent, de música i d’alegria, em vaig sentir com si la vida m’hagués regalat una postal dels meus millors somnis. Aquell somriure que se m’escapava no era només meu; era el batec col·lectiu d’una felicitat compartida, d’instants que mai volen desaparèixer.",
        "hash": "9ef0e0e7887193a4587443fcc2fec940",
        "alpha": 0.3,
        "beta": 0.7,
        "diffusion_steps": 5,
        "embedding_scale": 1
}

# Send the POST request with JSON data
start_time = time.time()
response = requests.post(url, json=data)
print(f"Time Taken", time.time() - start_time)
# Preview the audio file
Audio(response.content, autoplay=True)
```

#### Response
- **Status Code**: `200 OK`
- **Body**: Audio content in binary format.

#### Example Response
The response will be an audio file that can be played directly in a Jupyter notebook using the `IPython.display.Audio` class.

#### Note: 
`alpha` and `beta` is the factor to determine much we use the style sampled based on the text instead of the reference. The higher the value of `alpha` and `beta`, the more suitable the style it is to the text but less similar to the reference. Using higher beta makes the synthesized speech more emotional, at the cost of lower similarity to the reference. `alpha` determines the timbre of the speaker while `beta` determines the prosody.
`alpha` and `beta` determine the diversity of the synthesized speech. There are two extreme cases:
- If `alpha = 1` and `beta = 1`, the synthesized speech sounds the most dissimilar to the reference speaker, but it is also the most diverse (each time you synthesize a speech it will be totally different).
- If `alpha = 0` and `beta = 0`, the synthesized speech sounds the most siimlar to the reference speaker, but it is deterministic (i.e., the sampled style is not used for speech synthesis).

`diffusion_steps` - Increasing this value makes the speech more clearer on the cost of time
`embedding_scale` - Increasing this value makes the speech more expressive on the cost of time


### SpanishTTS API

This API is used to generarate speech in Spanish using the CoquiTTS model. CoquiTTS is an open-source, deep-learning-based text-to-speech (TTS) system designed to provide high-quality speech synthesis with flexibility and customization. One of the notable pre-trained models available in CoquiTTS is the CSS10 Spanish model, which is specifically trained to generate natural-sounding Spanish speech.

#### Endpoint
- `POST /predictions/STTS2Spanish` (Spanish TTS)

#### Request
e.g.
- **URL**: `{inference api url}/predictions/STTS2Spanish` # Replace `{inference api url}` with the Inference API URL (e.g., `http://localhost:8080`, `https://example.com`)
- **Method**: `POST`
- **Headers**: `Content-Type: application/json`
- **Body**: JSON object containing the following fields:
    - `text`: The text to generate speech.


#### Example Request
```python
import requests
import time
import base64
from IPython.display import Audio

# Define the URL (ensure this matches your actual model name)
url = "http://127.0.0.1:8080/predictions/STTS2Spanish"

# Define the data (input text)
data = {
    "text": "En un pequeño pueblo rodeado de montañas y ríos cristalinos, vivía una joven llamada Elena. Desde pequeña, había sentido una profunda fascinación por las estrellas y los misterios del universo. Todas las noches, subía a la colina más alta y, con su viejo telescopio, observaba el cielo estrellado, imaginando historias sobre los mundos lejanos que tal vez nunca podría visitar. Una noche, mientras contemplaba la luna llena, escuchó un sonido extraño proveniente del bosque. Sin pensarlo dos veces, decidió investigar. Caminó entre los árboles altos y frondosos, siguiendo el eco de una melodía lejana. Finalmente, llegó a un claro iluminado por una luz tenue y misteriosa. En el centro, había un anciano de cabello plateado tocando una flauta de madera. Su música era hipnotizante, como si cada nota contara una historia secreta del universo. El anciano la miró y sonrió. 'Elena', dijo con voz serena, 'las estrellas no solo están en el cielo; también viven en los sueños de quienes se atreven a imaginar'. En ese momento, Elena comprendió que su destino no estaba limitado por la distancia o la realidad, sino por los límites de su propia imaginación."
}

# Send the POST request with JSON data
start_time = time.time()

try:
    response = requests.post(url, json=data)
    response.raise_for_status()  # Will raise an HTTPError for bad responses (4xx, 5xx)
    print(f"Time Taken: {time.time() - start_time:.2f} seconds")

    # Preview the audio file (assuming response.content is audio data)
    speech = base64.b64decode(response.content)
    with open("output_speech.wav", "wb") as f:
        f.write(speech)
    Audio(speech, autoplay=True)

except requests.exceptions.RequestException as e:
    print(f"Error occurred: {e}")
```

#### Response
- **Status Code**: `200 OK`
- **Body**: Audio content in base64 string format.

#### Example Response
The response will be an base64-encoded audio file that can be played directly in a Jupyter notebook using the `base64.b64decode` funtion and `IPython.display.Audio` class.

#### Note: 



## Deployment Guide

### Prerequisites
- Install [Docker](https://docs.docker.com/engine/install/ubuntu/#:~:text=This%20example%20downloads%20the,sh%20get%2Ddocker.sh)
- Install `awscli` [here](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html#:~:text=%24-,curl%20%22https%3A//awscli.amazonaws.com/awscli%2Dexe%2Dlinux%2Dx86_64.zip%22%20%2Do%20%22awscliv2.zip%22%0Aunzip%20awscliv2.zip%0Asudo%20./aws/install,-To%20update%20your)
- Login to AWS CLI
```bash
aws configure
```

### Step 1: Clone the Repository
```bash
git clone https://github.com/rosyteran/SpanishCatalanTTS
cd SpanishCatalanTTS
```

### Step 2: Download the Pretrained Models
- Download the pretrained models
```bash
mkdir model_store
aws s3 cp s3://optimized-stts2-model/STTS2Encoder.mar model-store/STTS2Encoder.mar
aws s3 cp s3://optimized-stts2-model/STTS2Catalan.mar model-store/STTS2Catalan.mar
aws s3 cp s3://optimized-stts2-model/STTS2SpanishEncoder.mar model-store/STTS2SpanishEncoder.mar
aws s3 cp s3://optimized-stts2-model/STTS2Spanish.mar model-store/STTS2Spanish.mar
```

### Step 3: Build the Docker Image
```bash
docker build -t stts2-spanish-catalan .
```

### Step 4: Run the Docker Container
```bash
docker run -d --rm -it --gpus all --name stts   -p 8080:8080 -p 8081:8081 -p 8082:8082 -p 7070:7070 -p 7071:7071   -v $(pwd)/model_store:/home/model-server/model-store   --entrypoint "sh"   stts2-spanish-catalan -c "echo max_response_size=52428800 >> /home/model-server/config.properties && torchserve --start --model-store /home/model-server/model-store --models /home/model-server/model-store/STTS2Encoder.mar,/home/model-server/model-store/STTS2Catalan.mar,/home/model-server/model-store/STTS2SpanishEncoder.mar,/home/model-server/model-store/STTS2Spanish.mar --disable-token-auth --enable-model-api --foreground --no-config-snapshots"
```
> Note: CoquiTTS model can have response up to 50MB by default. If you need to test with long text or receive above 50 MB of speech by response, you will need to manually modify this option in above command. `echo max_response_size=52428800`

### Step 5: Scale workers (Optional)

Use Management API Endpoint to scale the workers
```bash
# Scale the StyleCatalan model to more than 1 workers (depending on the GPU Size) (1 worker takes roughly 2 GB of GPU Memory)
curl -X PUT "{management_api_url}/models/STTS2Spanish?min_worker=1&max_worker=8"
# Replace {management_api_url} with the Management API URL (e.g., http://localhost:8081)
```
> Default port for Inference API is 8080 and for Management API is 8081 (can be changed in the docker run command)

Refer to [TorchServe Documentation](https://pytorch.org/serve/management_api.html) for more details on Management API

### Step 6: Test the API
- Follow the API Docs to test the API

> You can check the logs using `docker logs stts` and stop the container using `docker stop stts`
