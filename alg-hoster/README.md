# How to Use This

In the event that the inferencing container requires instance types that are very large for hosting models or running batch transform jobs, the contents of this folder can be used to host a model yourself.

## Steps

1. Install gunicorn
2. Install nginx
3. Copy the nginx server configuration file "flask" to /etc/nginx/sites-enabled. This will throw requests sent over http to the server when we run it. This file can be edited if wanted.
4. Copy the model.tar.gz file from the train container output to alg-hoster/model and extract it there.
5. Run the following command, while in this directory: **gunicorn app:app --timeout 9999 --daemon --capture-output --log-file SERVER.OUT**
