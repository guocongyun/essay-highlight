## Getting Started

By default the demo will send requests to API endpoints running in production. If you'd like to 
run an endpoint locally, just run the start command with the ID of the endpoint:

```
./demo start ui
./demo start bidaf (Starts qa model in Reading Comprehension tab)
./demo start bidaf_elmo (Starts textual similarity model in Evaluate Reading Comprehension)
./demo start transformer-qa (Starts summarization model in Named Entity Recognition)
```

The website is hosted on localhost:8080
Please unzip qa model in api/bidaf/common/qa_models directory. 
Please unzip similarity model in api/bidaf/common/similarity_models directory. 
Please unzip summarization model in api/bidaf/common/summarization_models directory. 
The unzipped directory should be
```
+--api
   +--bidaf_elmo
   +--...
   +--common
      +--qa_model
          +--deepsettosed
              +--checkpoint-14800
                  +--pytorch_model.bin
                  +--...
      +--similarity_model
        +--pytorch_model.bin
        +--...
      +--summarization_models
        +--pytorch_model.bin
        +--...
```
                  

Note2: The current server was tested on windows. If Run On Linux, Please set the global variable posix in ./demo.py, on line 43, to True.


Known issue:

    1) Too large zip file may cause the server to crash

