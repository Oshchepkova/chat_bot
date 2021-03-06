FROM python:3.7.3-slim
WORKDIR /chat_bot
RUN pip3 install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
COPY . .
RUN pip3 install -r requirements.txt
RUN python3 -m nltk.downloader punkt 
CMD ["chat.py"]
ENTRYPOINT ["python3", "-u"]
