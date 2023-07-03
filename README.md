# Voice-Chat-With-Research-Paper
This repo contains python code that was created via Databutton, Cohere, OpenAI and Streamlit. It is a multi-lingual chatbot that reads your text or PDF files, and is prompt-engineered specifically for medical data.
The app alows you to pass voice input, which then gets transcribet to text via OpenAI's Whisper. That text gets passed into a prompt using LangChain and finally, the prompt is fed into an LLM by Cohere.
This work builds upon https://medium.com/databutton/multilingual-chat-bot-for-personal-documents-using-coheres-multilingual-models-langchain-2b4e1c8cdab! (Thanks for sharing!)

- The app can be accessed at: https://databutton.com/v/dczfx71a

**How to use**
Click on "Click to record" to start recording. When finished you have to click on "Recording" again, to stop the recording and to initiate speech to text transcription.
Then you wait for a few second for the LLM to process your prompt, after which you will recieve the answer!

![image](https://github.com/vrda23/Voice-Chat-With-Research-Paper/assets/93191867/444baffa-992d-4757-a146-646f324ba731)

