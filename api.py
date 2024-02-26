from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from flask import Flask, jsonify, request
import os
import filecmp
 
 
app = Flask(__name__)
 
def answer_question(question):
    pdf = PdfReader("sample.pdf")
    raw_text = ""
    for i,page in enumerate(pdf.pages):
        content = page.extract_text()
        if content:
            raw_text+=content
 
 
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 800,
        chunk_overlap = 200,
        length_function = len
    )
    text = text_splitter.split_text(raw_text)
    embeddings = OpenAIEmbeddings(
        openai_api_key="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"
    )
    print(embeddings)
    documet_search = FAISS.from_texts(text,embeddings)
 
 
    chain = load_qa_chain(OpenAI(
        openai_api_key="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"
        ),chain_type="stuff")
 
    docs = documet_search.similarity_search(question)
    data = chain.run(input_documents=docs,question=question)
    return data
   
   
   
@app.route("/answer", methods=['POST'])
def custom_qna():
    data = request.data.decode("utf-8")
    responce = answer_question(data)
    return jsonify({
            "answer":responce
        })
   

@app.route('/success', methods = ['POST'])   
def success():   
    if request.method == 'POST':   
        f = request.files['file'] 
        f.save("sample.pdf")   
        return jsonify({
            "answer":True
        })
   
app.run(debug=True)