import torch
from ts.torch_handler.base_handler import BaseHandler
from TTS.api import TTS
import torchaudio
from io import BytesIO
import base64

class CoquiTTSHandler(BaseHandler):
    def __init__(self):
        super(CoquiTTSHandler, self).__init__()
        self.model = None

    def initialize(self, ctx):
        # Load Coqui TTS model from local files
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            self.model = TTS(model_name="tts_models/es/css10/vits", gpu=True)
        else:
            self.model = TTS(model_name="tts_models/es/css10/vits", gpu=False)
        self.initialized = True

    def preprocess(self, data):
        if isinstance(data, list):
            data = data[0]
        text = data.get("data") or data.get("body")
        return text["text"]

    def inference(self, text):
        if not self.model:
            raise RuntimeError("Model is not initialized")

        # Clear GPU cache before running inference
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # Generate speech
        wav = self.model.tts(text=text)

        # Convert to tensor
        wav_tensor = torch.tensor(wav, dtype=torch.float32)

        # Save to BytesIO buffer
        buffer = BytesIO()
        torchaudio.save(buffer, wav_tensor.unsqueeze(0), 22050, format="wav")
        buffer.seek(0)
        return buffer.read()

    def postprocess(self, output):
        return [base64.b64encode(output).decode("utf-8")]

    def handle(self, data, ctx):
        text = self.preprocess(data)
        wav_output = self.inference(text)
        res = self.postprocess(wav_output)
        return res
