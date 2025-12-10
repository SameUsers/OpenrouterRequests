from vosk import Model, KaldiRecognizer
import json


class VoskService:
    _instance = None
    _model = None
    _recognizer = None
    _initialized = False

    def __new__(cls, model_path=None):
        if cls._instance is not None:
            return cls._instance

        if model_path is None:
            raise ValueError("model_path required for first initialization")

        cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_path=None):
        if not self.__class__._initialized and model_path:
            self.__class__._model = Model(model_path)
            self.__class__._recognizer = KaldiRecognizer(self.__class__._model, 16000)
            self.__class__._initialized = True

    @classmethod
    async def recognize(cls, chunk):
        if cls._recognizer is None:
            cls()

        if len(chunk) == 0:
            return None
        if cls._recognizer.AcceptWaveform(chunk):
            result_json = cls._recognizer.FinalResult()
            result = json.loads(result_json)
            text = result.get("text", "").strip()
            return text

        return None