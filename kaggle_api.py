from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

api.competition_submit('submission.csv', message=
                       'test api sub', competition='digit-recognizer')