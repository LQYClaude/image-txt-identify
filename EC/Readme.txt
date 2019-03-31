Author: Qianyi Lu 2018/9/24

All the packages and modules should be installed are listed in requirements file.
This system is better lunched in a virtual environment so please find a python programmer to install it.

Use "python server.py" to lunch the server, no other commands required on command window.

All the html files are better be put in app/templates folder. All the css files are better be put in app/static folder. They are presently be used as default.

Trainned neural network matrix is presently saved in app/my_net folder.

Data collected from user is save in app/mat folder.

--------------------------------------------------------------------------------
This system is Written in python.

The natural language classification is based on NLTK. Include words distribution and stemming.

The neural network is based on tensorflow. Include two kinds of network: RNN and CNN. RNN is better for those case which has a logic from start to end, such as playing games and writting. CNN is better for those cases contain small continue parts, such as a picture contains graphics, a movie has cuts and a song has climax.

The web server is based on flask. Include index page (home page), answer page (showing analysis result), contact page and a base page for the structure. 