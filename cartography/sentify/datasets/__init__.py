from sentify.datasets.imdb import IMDBDataModule
from sentify.datasets.measuring_hate_speech import (
    MeasuringHateSpeechDataModule,
    SentimentMHSDataModule,
    HateSpeechMHSDataModule,
    ViolenceMHSDataModule,
    InsultMHSDataModule,
    HumiliateMHSDataModule,
)
from sentify.datasets.sentiment140 import Sentiment140DataModule

DATASETS = {
    'sentiment140': Sentiment140DataModule,
    'imdb': IMDBDataModule,
    'MHS_sentiment': SentimentMHSDataModule,
    'MHS_hatespeech': HateSpeechMHSDataModule,
    'MHS_insult': InsultMHSDataModule,
    'MHS_violence': ViolenceMHSDataModule,
    'MHS_humiliate': HumiliateMHSDataModule,
}
