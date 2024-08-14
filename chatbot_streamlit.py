import streamlit as st

# Your chatbot code imports
import re
import nltk
import numpy as np
import PyPDF2
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
from pinecone import Pinecone
from langchain.chains import SequentialChain
from langchain.chains.base import Chain
from langchain_community.llms import HuggingFaceHub
import warnings
warnings.filterwarnings("ignore")

nltk.download('punkt')

# Initialize Pinecone
pc = Pinecone(api_key="9a8868f5-c5be-4129-bc8d-6d553e29a5cf")
index = pc.Index("bankpdf2")

# Initialize HuggingFace Model for responses
model_id = "HuggingFaceH4/zephyr-7b-beta"
huggingfacehub_api_token =  "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"
conv_model = HuggingFaceHub(
    huggingfacehub_api_token=huggingfacehub_api_token,
    repo_id=model_id,
    model_kwargs={"temperature": 0.1, "max_new_tokens": 100}
)

# Initialize BERT model and tokenizer
model_name = "sentence-transformers/all-mpnet-base-v2"
tokenizer_bert = AutoTokenizer.from_pretrained(model_name)
model_bert = TFAutoModel.from_pretrained(model_name)

# Define your chatbot functions
def preprocess_text(text):
    return re.sub(r'[^\w\s]', '', text).lower()

def get_embeddings(text):
    inputs = tokenizer_bert(text, return_tensors="tf", padding=True, truncation=True)
    outputs = model_bert(inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()  # CLS token
    return embeddings

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text

def segment_sentences(text):
    return nltk.sent_tokenize(text)

def get_best_segments(user_input, index, top_k=3):
    user_input_embedding = get_embeddings(preprocess_text(user_input))[0]
    result = index.query(
        namespace="Default",
        vector=user_input_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )
    best_segment_ids = []
    if 'matches' in result and result['matches']:
        best_segment_ids = [match["id"] for match in result["matches"]]
    return best_segment_ids

def process_response(response):
    return response.strip()

def split_into_segments(text):
    segments = re.split(r'(?i)\band\b|(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return [segment.strip() for segment in segments if segment.strip()]

# Define the Chains
class RephraseQuestionChain(Chain):
    @property
    def input_keys(self):
        return ["user_input"]
    @property
    def output_keys(self):
        return ["rephrased_question"]
    def _call(self, inputs):
        user_input = inputs["user_input"]
        rephrase_prompt = f"Rephrase the following question for better understanding:\n\nQuestion: {user_input}\n\nRephrased question:"
        response = conv_model(rephrase_prompt, parameters={"temperature": 0.1, "max_new_tokens": 50})
        rephrased_question = process_response(response)
        return {"rephrased_question": rephrased_question}

class ExtractSegmentsChain(Chain):
    @property
    def input_keys(self):
        return ["user_input"]
    @property
    def output_keys(self):
        return ["extracted_segments"]
    def _call(self, inputs):
        user_input = inputs["user_input"]
        segments = split_into_segments(user_input)
        return {"extracted_segments": segments}

class RetrieveBestSegmentsChain(Chain):
    @property
    def input_keys(self):
        return ["rephrased_question"]
    @property
    def output_keys(self):
        return ["retrieved_segments"]
    def _call(self, inputs):
        rephrased_question = inputs["rephrased_question"]
        best_segment_ids = get_best_segments(rephrased_question, index, top_k=3)
        all_best_segments = []
        if best_segment_ids:
            corpus_segments = [sentences[int(segment_id.split("_")[1])] for segment_id in best_segment_ids]
            all_best_segments.append(corpus_segments)
        else:
            all_best_segments.append(["I'm sorry, but I can only respond to questions related to the corpus."])
        return {"retrieved_segments": all_best_segments}

class GenerateResponseChain(Chain):
    @property
    def input_keys(self):
        return ["retrieved_segments", "user_input"]
    @property
    def output_keys(self):
        return ["response"]
    def _call(self, inputs):
        user_input = inputs["user_input"]
        segments = inputs["retrieved_segments"]
        all_responses = []
        for segment, corpus_segments in zip(split_into_segments(user_input), segments):
            if isinstance(corpus_segments, list):
                concatenated_segments = " ".join(corpus_segments)
                input_text = (
                    f"The following information was retrieved from the corpus:\n\n{concatenated_segments}\n\n"
                    f"Question: {segment}\n\n"
                    f"Answer:"
                )
                response = conv_model(input_text, parameters={"temperature": 0.1, "max_new_tokens": 100})
                response = response.strip()
                # Remove repeated information
                response_lines = response.split('\n')
                filtered_response_lines = []
                for line in response_lines:
                    if line not in concatenated_segments:
                        filtered_response_lines.append(line)
                filtered_response = '\n'.join(filtered_response_lines).strip()
                all_responses.append(filtered_response)
            else:
                all_responses.append(corpus_segments[0])
        final_response = "\n".join(all_responses)
        return {"response": final_response}

# Combine the Chains
chain = SequentialChain(chains=[
    RephraseQuestionChain(),
    ExtractSegmentsChain(),
    RetrieveBestSegmentsChain(),
    GenerateResponseChain()
], input_variables=["user_input"], output_variables=["response"])

# Load the PDF
pdf_path =  r"C:\Users\bhettiarachchi.ext\Downloads\reformattingbankpdf.pdf"
pdf_text = extract_text_from_pdf(pdf_path)
sentences = segment_sentences(pdf_text)

# Streamlit App Layout
st.title("Chatbot Interface")

st.write("Type your question below:")

# Input from the user
user_input = st.text_input("You:")

# Process the input and get the response
if user_input:
    response = chain.run({"user_input": user_input})
    if isinstance(response, dict) and "response" in response:
        st.write(f"Chatbot: {response['response']}")
