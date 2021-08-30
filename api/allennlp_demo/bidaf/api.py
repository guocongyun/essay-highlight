import os
from allennlp_demo.common import config
from allennlp_demo.common import config, http

class RobertaModelEndpoint(http.MyModelEndpoint):
    def __init__(self):
        # pass
        model_path = "/app/allennlp_demo/common/qa_models/roberta/deepsettosed_kf2/checkpoint-14800"
        c = config.Model.from_file(os.path.join(os.path.dirname(__file__), "model.json"))
        super().__init__(model_path, c)

if __name__ == "__main__":
    endpoint = RobertaModelEndpoint()
    endpoint.run()
    # inputs = {
    #     "passage": "A reusable launch system (RLS, or reusable launch vehicle, RLV) is a launch system which is capable of launching a payload into space more than once. This contrasts with expendable launch systems, where each launch vehicle is launched once and then discarded. No completely reusable orbital launch system has ever been created. Two partially reusable launch systems were developed, the Space Shuttle and Falcon 9. The Space Shuttle was partially reusable: the orbiter (which included the Space Shuttle main engines and the Orbital Maneuvering System engines), and the two solid rocket boosters were reused after several months of refitting work for each launch. The external tank was discarded after each flight.",
    #     "question": "How many partially reusable launch systems were developed?",
    # }
    # prediction = endpoint.predict(inputs)
    # print(prediction)
