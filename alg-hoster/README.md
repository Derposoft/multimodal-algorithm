# How to Use This

In the event that the inferencing container requires instance types that are very large for hosting models or running batch transform jobs, the contents of this folder can be used to host a model yourself.

## Steps

1. Install gunicorn
2. Install nginx
3. Copy the nginx server configuration file "flask" to /etc/nginx/sites-enabled. This will throw requests sent over http to the server when we run it. This file can be edited if wanted. Restart the nginx service.
4. Copy the model.tar.gz file from the train container output to alg-hoster/model and extract it there.
5. Run the following command, while in this directory: **gunicorn app:app --timeout 9999 --daemon --capture-output --log-file SERVER.OUT**. This will start the server in daemon mode, allowing you to drop ssh connection with the server if wanted. All server logs will be outputted to SERVER.OUT. Currently, a 9999 timeout is required in order to allow time for the server to initially load model files.

## Additional Rules

- Currently, the machine running this needs enough memory to hold all of the model files. In other words, if you have a 300 GB model, you need 300 GB of memory. This is due to the fact that reading a model from the disk is much slower, and hosted model servers for inferencing are often meant to be responsive.

- To use the model, send a POST request to the IP address of the machine where the server is hosted, at the /infer endpoint. Ensure that the firewall is set to allow incoming traffic at port 80. The post request body should be a JSON set of key-value pairs { k1: v1, k1: v2, ...} such that v1, v2, ... are the values of the features *in the same order* as the columns of the train dataset. The values of the keys don't matter.

- The response to the POST request should look something like this:

{

    'ID': [id entered with row],

    'label': { 'class': predicted class },

    'nrow': The row inputted, spat back out at you, but with the labels. Can be used for debugging to ensure that the ordering of the request was correct.
    
}
