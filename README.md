# Daily-Mood-Tracker
Trained a custom LSTM model that classifies a person's emotions based on the text entered. An application of this could be a personal diary that detects your emotions automatically.

# File Description:
* train, test, val: These are the datasets that contain the text and the associated emotion separated with a semicolon (;).
* label_encoder.pkl: Contains the encoded labels for the different emotions. The encoder was trained on the training data and stored as .pkl file. The code can be found in the jupyter notebook.
* word2idx.pkl: Contains unique IDs for the different words in the text. While training, we first make a vocabulary of all the words present in the dataset. Then we make a dictionary that attaches a unique number to each word. These word indices can be used to vectorize the sentences.
* mood_db.py: Contains the SQL functions to create the table, insert entries and then list all the entries.
* mood_tracker.db: The table where entries are stored.
* sentiment_code.ipynb: Contains all the preprocessing code. Contains the architecture of the LSTM model and the training of the model.
* model_1.pth: This file contains the trained model. This was used in the streamlit app to access the functionality of the model.
* main.py: This is the streamlit file, which can be run through the command `streamlit run main.py`. Contains the UI for the web app. Makes it easier for the user to enter the text, see the result, and store the entry in the modd_tracker.db .
