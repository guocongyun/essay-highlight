## Getting Started

By default the demo will send requests to API endpoints running in production. If you'd like to 
run an endpoint locally, just run the start command with the ID of the endpoint:

```
./demo start ui
./demo start bidaf
```

Note1: Please unzip qa model in api/bidaf/common/qa_models directory. The unzipped directory should be
    -api
        -bidaf_elmo
        -...
        -common
            -qa_model
                -deepsettosed
                    -checkpoint-14800
                        -pytorch_model.bin

Note2: The current server was tested on windows. If Run On Linux, Please set the global variable posix in ./demo.py, on line 43, to True.

