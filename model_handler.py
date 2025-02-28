# custom handler file

# model_handler.py

"""
ModelHandler defines a custom model handler.
"""
import os
from ts.torch_handler.base_handler import BaseHandler
from models import StyleTTS2Catalan, StyleTTS2CatalanConfig, TextCleaner
import json, torch
from nltk.tokenize import word_tokenize
import phonemizer
import io
import soundfile as sf
text_cleaner = TextCleaner()
global_phonemizer = phonemizer.backend.EspeakBackend(language='ca', preserve_punctuation=True,  with_stress=True)

class ModelHandler(BaseHandler):
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
        model_dir = properties.get("model_dir")
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu")
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
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)

        self.model = StyleTTS2Catalan(config)
        self.model.load_state_dict(torch.load(model_pt_path, weights_only=True))
        self.model = self.model.to(self.device)
        print("Model loaded")

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        print(data, "data")
        text = data.pop("text")
        phonemized_text = global_phonemizer.phonemize([text])
        phonemized_text = word_tokenize(phonemized_text[0])
        phonemized_text = ' '.join(phonemized_text)
        tokens = text_cleaner(phonemized_text)
        tokens.insert(0, 0)
        data['tokens'] = tokens
        data['device'] = self.device
        if 'hash' in data:
            data['ref_s'] = torch.load(f"/tmp/tts/cache/{data.pop('hash')}.pt")
        elif 'ref_s' in data:
            data['ref_s'] = torch.tensor(data['ref_s'])
        else:
            with open("default_ref_audio.json") as f:
                data['ref_s'] = torch.tensor(json.load(f))
        return data


    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        print(model_input, "model_input")
        with torch.no_grad():
            tokens = model_input.pop('tokens')
            print(self.device, "device")
            print(model_input['ref_s'], "ref_s", type(model_input['ref_s']))
            tokens = torch.LongTensor(tokens).unsqueeze(0).to(self.device)
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
            model_input['tokens'] = tokens
            model_input['input_lengths'] = input_lengths
            model_output = self.model(**model_input)
        return model_output

    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        print(inference_output, "inference_output")
        postprocess_output = inference_output.cpu().numpy()[..., :-50]
        with io.BytesIO() as buffer:
            sf.write(buffer, postprocess_output, samplerate=24000, format='WAV')
            buffer.seek(0)  # Reset the buffer's position to the start
            data = buffer.read()
        return [data]

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        print(data, "data")
        if isinstance(data, list):
            data = data[0]
        data = data.get("data") or data.get("body")
        model_input = self.preprocess(data)
        print(model_input, "model_input")
        model_output = self.inference(model_input)
        print(model_output, "model_output")
        return self.postprocess(model_output)