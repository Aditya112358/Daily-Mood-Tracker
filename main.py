
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import pickle
import mood_db

st.title("Your Personal Mood Tracker!")

mood_db.create_table()
with open('label_encoder.pkl','rb') as file:
    label_encoder = pickle.load(file)

with open('word2idx.pkl','rb') as file:
    word2idx = pickle.load(file)



class EmtionalClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(EmtionalClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=word2idx['<PAD>'])
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self,x):
        x = self.embedding(x)
        _, (hidden,_) = self.lstm(x)
        out = self.fc(self.dropout(hidden[-1]))
        return out
    
model_1 = EmtionalClassifier(vocab_size=len(word2idx), embed_dim=100,hidden_dim=128,output_dim=len(label_encoder.classes_))
model_1.load_state_dict(torch.load("model_1.pth",weights_only=True))

import torch
from nltk.tokenize import word_tokenize

MAX_LEN = 50  # or whatever you used during training

def predict_emotion(text, model=model_1, word2idx=word2idx, label_encoder=label_encoder, device='cpu'):
    model.eval()

    # Clean and tokenize
    text = text.lower()
    tokens = word_tokenize(text)
    input_ids = [word2idx.get(token, word2idx['<UNK>']) for token in tokens]

    # Pad or truncate
    if len(input_ids) < MAX_LEN:
        input_ids += [word2idx['<PAD>']] * (MAX_LEN - len(input_ids))
    else:
        input_ids = input_ids[:MAX_LEN]

    # Convert to tensor
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        predicted_idx = torch.argmax(output, dim=1).item()

    # Map to label
    predicted_emotion = label_encoder.inverse_transform([predicted_idx])[0]
    return predicted_emotion

st.subheader("Welcome")
option = st.selectbox(
    "What would you like to do:",
    ("To make an entry", "To retrieve all the entries"),
)

if option == "To make an entry":
    
    title = st.text_input("Type your text here:")
    emotion = predict_emotion(text=title)
    st.markdown("**Your Entry:**")
    st.write(title)
    st.markdown("**According to us your mood is:**")
    st.write(emotion)
    confidence = st.number_input("How much do you agree with this? (1-10)")
    # print(emotion)
    
    if st.button("Submit",type="primary"):
        mood_db.insert_entry(text=title,mood=emotion, confidence=confidence)
        st.write("Your entry has been submitted.")
        
        
    

elif option == "To retrieve all the entries":
    db_list = mood_db.get_all_entries()
    df = pd.DataFrame(db_list,columns=["Entry No.", "Entry","Mood","Confidence","Date"],index=range(1,len(db_list)+1))
    st.dataframe(df)
else:
    print("Invalid Entry")


