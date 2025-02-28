"""
ModelHandler defines a custom model handler.
"""
import os
from ts.torch_handler.base_handler import BaseHandler
from models import StyleTTS2CatalanConfig, StyleTTS2Encoder
import json
import torch
import librosa
import tempfile
import hashlib


class EncoderHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        self._context = None
        self.initialized = False
        self.explain = False
        self.target = 0
        print("Model handler initialized")

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self._context = context
        self.manifest = context.manifest
        self.initialized = True
        properties = context.system_properties
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id")
            is not None else "cpu")
        print("Device is: {}".format(self.device))
        #  load the model, refer 'custom handler class' above for details
        with open("config.json") as f:
            config = json.load(f)

        ASR_config = config.pop('ASR_config')
        bert = config.pop('bert')
        slm = config.pop('slm')

        config = StyleTTS2CatalanConfig(
            asr_config=ASR_config,
            bert=bert,
            slm=slm,
            **config
        )
        os.makedirs("/tmp/tts/cache", exist_ok=True)

        self.encoder = StyleTTS2Encoder(config)
        self.encoder.load_state_dict(torch.load("StyleTTS2Encoder.pth",
                                                weights_only=True))
        self.encoder = self.encoder.to(self.device)
        print("Encoder loaded")

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of
        prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        if isinstance(data, list):
            data = data[0]
        audio = data.pop('audio')
        hashed = data.pop('hash', None)
        fp = tempfile.TemporaryFile()
        fp.write(audio)
        fp.seek(0)
        with torch.no_grad():
            if hashed:
                encoded = torch.load(f"/tmp/tts/cache/{hashed}.pt").to(self.device)
            else:
                hashed = hashlib.md5(fp.read()).hexdigest()
                fp.seek(0)
                wave, sr = librosa.load(fp, sr=24000)
                wave, index = librosa.effects.trim(wave, top_db=30)
                wave_tensor = torch.from_numpy(wave).float().to(self.device)
                encoded = self.encoder(wave_tensor)
        cwd = os.getcwd()
        torch.save(encoded, os.path.join("/tmp/tts/cache", f"{hashed}.pt"))
        print("Hashed", os.path.join("/tmp/tts/cache", f"{hashed}.pt"))
        data['hash'] = hashed
        data['ref_s'] = encoded.cpu().numpy().tolist()
        return [data]