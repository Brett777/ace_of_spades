import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict
import faiss
import json
import textwrap
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets.openai_api_key)

# Set page configuration
st.set_page_config(page_title="Ace of Spades", layout="centered")

# Add these constants at the top of the file, after the imports
INDEX_FILE_PATH_CHAPTERS = "book_vector_index_chapters.faiss"
INDEX_FILE_PATH_PAGES = "book_vector_index_pages.faiss"
METADATA_FILE_PATH_CHAPTERS = "book_vector_metadata_chapters.json"
METADATA_FILE_PATH_PAGES = "book_vector_metadata_pages.json"

def load_vector_indices():
    """Load FAISS indices and metadata"""
    # Load FAISS indices
    chapter_index = faiss.read_index(INDEX_FILE_PATH_CHAPTERS)
    page_index = faiss.read_index(INDEX_FILE_PATH_PAGES)
    
    # Load metadata
    with open(METADATA_FILE_PATH_CHAPTERS, 'r') as f:
        chapter_metadata = json.load(f)
    with open(METADATA_FILE_PATH_PAGES, 'r') as f:
        page_metadata = json.load(f)
        
    return chapter_index, page_index, chapter_metadata, page_metadata

def create_embedding(text: str, client: OpenAI) -> list:
    """Create embedding for input text"""
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def fetch_relevant_content(pages: list, chapters: list, chapter_metadata: dict, page_metadata: dict) -> dict:
    """Fetch relevant content from metadata files based on specified pages and chapters"""
    context = {
        'chapter_results': [item for item in chapter_metadata if item.get('chapter_number') in chapters],
        'page_results': [item for item in page_metadata if item.get('page_number') in pages]
    }
    return context

def determine_relevant_pages(question: str, chapter_summaries: Dict, book_summary: str):    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": f"""
            ROLE:                        
            Your job is to find the most relevant pages in the book that support the user's question.
            Review the summary of the book to understand the main themes, characters, and plot.
            Use the summary to guide your search for relevant pages.
            Review the chapter summaries. The chapter summaries will also help you understand specifically which pages are most relevant.
            
            FULL BOOK SUMMARY:
            {book_summary}

            CHAPTER SUMMARIES:
            {chapter_summaries}
             
            RESPONSE:
            Respond with json. 
            Your response should be a list of the chapters and pages that are most relevant for answering the user's questions.
            Do not provide a range. Provide a specific chapters. Within those chapters, provide a specific list of page numbers.
            
            Example response:
            chapters: [1]
            pages: [1, 2, 3, 4, 5]
            
               """},
            
            {"role": "user", "content": f"The user has the following question. Determine which pages would be helpful to read in order to answer this question:\n {question}"}
        ]
    )

    result = json.loads(response.choices[0].message.content)
    return {
        "pages": result.get("pages")               
    }

def answer_user_question(question: str, context: Dict, book_summary: str):
    """Get AI response to user question"""
    with open('entire_book_summary.txt', 'r', encoding='utf-8') as f:
        book_summary = f.read()
    
    response = client.chat.completions.create(
        model="gpt-4o",       
        temperature=0.7,
        messages=[
            {"role": "system", "content": f"""
            ROLE:
            Your job is to answer a user's question based on the book Ace of Spades.
            Your answer should be detailed and provide enough specific examples with chapter and page references
            such that the reader of your response can turn to the relevant pages in the book and understand the context of your answer.
            CONTEXT:
             You will be provided with a summative report about the book to help you understand the question. 
             The summative report should also help you find support facts in the detailed context provided. 
             The detailed context includes summaries of the chapters and full text of the pages that are most relevant to the question.
             You should carefully consider the detailed context and use it to support your answer. 

            Chapter Summaries:
             In the context, chapter summaries look like this. Notice that the a chapter number, name, start page, end page, are provided. Use this information to find the most relevant pages in the book.
                "chapter_results":[
                0:
                "type":"chapter_summary"
                "chapter_number":39
                "chapter_name":"Chiamaka"
                "start_page":344
                "end_page":347
                "content":[
                0:"Chiamaka and Devon arrive at the back entrance of the school wearing masks."
                1:"They plan to meet journalist Ms. Donovan and her crew in Morgan Library."
                2:"Chiamaka retrieves a 1965 yearbook as proof for later."
                3:"Devon is nervous and feels watched in the library."
                4:"Chiamaka sends Devon to the back door to look out for the journalist."
                5:"Devon observes students engaging in typical senior antics of drinking and smoking."
                6:"Devon receives a call from Terrell, who surprises him by showing up in person."
                7:"Terrell apologizes and reveals he helped the school spy on Devon, causing a tense emotional moment."
                ]
               
            
             In the context, full text pages look like this. Notice that the a page number, start line, and end line are provided. Use this information to find details and direct quotes to support your answer.
             "type":"page"
             "page_number":179
            "chapter_number":"CHAPTER 19"
            "chapter_name":"Part Two: X Marks the Spot - CHAPTER 19"
            "content":"I ’ m lost. Reason being, I decided to listen to Chiamaka fucking Adebayo. After detention on Friday she attacked me again and forced her number into my phone, and then sent me a message this morning to meet her in lab 201 – wherever the fuck that is. A hand grabs my arm and I almost scream. My heart's near to exploding as I swing around, only to see an Chiamaka. “ You ’ late. ” Think I don t know that? “ I didn t know where lab 201 was. ” She doesn t seem impressed, and I don t think I really care. I want Aces to stop, I want Dre to speak to me again, and I just want to get into Juilliard and be done with Niveus. She pulls me into a room – lab 201, I guess and I ’ met with a guy seated at a desk with a laptop opened up. She hits my arm. Give him your phone. ” I look at her, hoping she feels the dagger I ’ m mentally throwing. “ Why my phone? Why not his or yours? Chiamaka gives me the look my ma gives me when I give her lip."

            RESPONSE:
             Your response should provide a consise and direct answer to the user's question.
             Support your answer with direct quotes and page references. 
             
            Summary of the book Ace of Spades:
            {book_summary}
            """},            
            {"role": "user", "content": f"Detailed Context - Chapter Summaries and Full Text Pages: {context}\n\n User Question: {question}"}
        ]
    )
    return response.choices[0].message.content

def chat_interface():
    """Create and handle chat interface"""
    # Load metadata files and book summary at the start of the function
    with open(METADATA_FILE_PATH_CHAPTERS, 'r') as f:
        chapter_metadata = json.load(f)
    with open(METADATA_FILE_PATH_PAGES, 'r') as f:
        page_metadata = json.load(f)
    with open('entire_book_summary.txt', 'r', encoding='utf-8') as f:
        book_summary = f.read()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages with consistent file extensions
    for message in st.session_state.messages:
        avatar_img = "assistant_avatar.jpg" if message["role"] == "assistant" else "user_avatar.jpg"
        with st.chat_message(message["role"], avatar=avatar_img):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about Ace of Spades"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="user_avatar.jpg"):
            st.markdown(prompt)

        # Get response
        with st.chat_message("assistant", avatar="assistant_avatar.jpg"):
            with st.spinner("Thinking..."):
                # Get relevant pages - Updated to include the required parameters
                relevant_content = determine_relevant_pages(prompt, chapter_metadata, book_summary)
                
                # Fetch the actual content
                context = fetch_relevant_content(
                    pages=relevant_content['pages'],
                    chapters=relevant_content.get('chapters', []),
                    chapter_metadata=chapter_metadata,
                    page_metadata=page_metadata
                )
                                
                response = answer_user_question(prompt, context, book_summary)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

def main_page():
    """Main page layout and functionality"""
    # Add custom CSS for fixed chat container
    st.markdown("""
    <style>
    .chat-container {
        position: fixed;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 100%;
        max-width: 800px;
        background-color: white;
        padding: 1rem;
        z-index: 1000;
    }
    
    /* Add padding to prevent content from being hidden behind chat */
    .main-content {
        padding-bottom: 200px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Wrap header content in a div with main-content class
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    # Load metadata files
    with open(METADATA_FILE_PATH_CHAPTERS, 'r') as f:
        chapter_metadata = json.load(f)
    with open(METADATA_FILE_PATH_PAGES, 'r') as f:
        page_metadata = json.load(f)
    
    # Load book summary
    with open('entire_book_summary.txt', 'r', encoding='utf-8') as f:
        book_summary = f.read()

    # Header container
    header_container = st.container()
    with header_container:
        left_col, right_col = st.columns([1, 5])
        
        with left_col:
            st.image("ace_of_spades_cover.jpg", width=500)
            
        with right_col:
            st.title("Ace of Spades")
            st.caption("By Faridah Àbíké-Íyímídé")
            st.subheader("Welcome to Niveus Private Academy, where power and privilege always win.")
            st.write("""
            Ask any questions about this thrilling debut novel, and I'll provide detailed answers 
            with specific chapter and page references. Whether you're curious about the characters, 
            plot twists, themes, or symbolism, I'm here to help you explore this gripping story of 
            institutional racism and dark academia.
            """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat container with fixed positioning
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    chat_interface()
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main_page()
