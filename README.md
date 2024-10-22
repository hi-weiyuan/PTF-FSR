# PTF-FedSeqRec
This is the implementation of our paper "PTF-FedSeqRec: A Parameter Transmission-Free Federated Sequential Recommender System"

# Things to know
- The server.py includes all the server model training logistics and the federated learning coordination.
- The client.py includes all the client-side algorithms.
- model.py includes the base model for GRU4Rec and SASRec, MoRec.py includes the MoRec code which originated from [MoRec](https://github.com/westlake-repl/IDvs.MoRec)
- All the important hyperparameters are in parse.py.
- main.py is used to run the code.
- The usage of GPU memory is positively related to the number of users and items of the specific dataset. If out of memory, consider moving server and client models on several different GPUs.
- **Please cite the paper if the code is helpful. Thanks!**
