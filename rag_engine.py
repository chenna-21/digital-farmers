import os
import logging
import pandas as pd
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEngine:
    """Hybrid RAG Engine - API (Online) + Local Dataset (Offline) + Multi-language"""
    
    def __init__(self):
        logger.info("=" * 60)
        logger.info("AgriBot - Hybrid RAG Mode")
        logger.info("=" * 60)
        
        # 1. Initialize API Clients
        self.has_groq = self._check_api_key("GROQ_API_KEY")
        self.has_openai = self._check_api_key("OPENAI_API_KEY")
        self.has_gemini = self._check_api_key("GOOGLE_API_KEY")
        
        if not (self.has_groq or self.has_openai or self.has_gemini):
            logger.warning("No Online LLM keys found. Running in OFFLINE-ONLY mode.")

        # 2. Check for Dataset
        self.dataset_path = "../dataset.csv"
        self.has_dataset = os.path.exists(self.dataset_path)
        
        # Initialize encoder and index as None
        self.encoder = None
        self.index = None
        self.df = None
        
        if self.has_dataset:
            logger.info(f"Loading dataset from {self.dataset_path}...")
            try:
                # Try reading with different encodings if default utf-8 fails
                try:
                    self.df = pd.read_csv(self.dataset_path)
                except UnicodeDecodeError:
                    self.df = pd.read_csv(self.dataset_path, encoding='latin1')
                    
                cols = self.df.columns
                # Flexible column detection
                self.q_col = next((c for c in cols if 'question' in c.lower()), cols[0])
                self.a_col = next((c for c in cols if 'answer' in c.lower()), cols[1] if len(cols) > 1 else cols[0])
                
                logger.info(f"Dataset Loaded: {len(self.df)} records. Using Question: '{self.q_col}', Answer: '{self.a_col}'")
                
                # 3. Initialize Embeddings & Index
                try:
                    logger.info("Loading Sentence Transformer model (this may take a moment)...")
                    self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
                    
                    logger.info("Generating/Loading embeddings...")
                    # Generate embeddings
                    questions = self.df[self.q_col].astype(str).tolist()
                    self.custom_embeddings = self.encoder.encode(questions)
                    
                    self.dimension = self.custom_embeddings.shape[1]
                    self.index = faiss.IndexFlatL2(self.dimension)
                    self.index.add(np.array(self.custom_embeddings).astype('float32'))
                    logger.info("FAISS Index created successfully.")
                    
                except Exception as e:
                    logger.error(f"Failed to initialize Offline RAG embeddings: {e}")
                    # Don't disable dataset, we can still use it for logging or future use, but RAG won't work
                    self.index = None
            except Exception as e:
                logger.error(f"Error loading dataset: {e}")
                self.has_dataset = False
        else:
            logger.warning("dataset.csv not found! Offline mode unavailable.")

    def _check_api_key(self, key_name):
        key = os.getenv(key_name)
        return key and key not in [f"your_{key_name.lower().split('_api')[0]}_api_key_here", ""]

    def _translate(self, text, target_lang):
        """Translate text to target language"""
        if not text: return ""
        if target_lang == 'en': return text
        try:
            # deep-translator uses 'hi' for Hindi, 'te' for Telugu
            return GoogleTranslator(source='auto', target=target_lang).translate(text)
        except Exception as e:
            logger.error(f"Translation Error ({target_lang}): {e}")
            return text

    def _search_local(self, query):
        """Search local dataset using FAISS"""
        if not self.has_dataset or self.index is None: return None
        
        try:
            query_embedding = self.encoder.encode([query])
            D, I = self.index.search(np.array(query_embedding).astype('float32'), k=3)
            
            results = []
            for idx in I[0]:
                if idx < len(self.df) and idx >= 0:
                    results.append({
                        "question": self.df.iloc[idx][self.q_col],
                        "answer": self.df.iloc[idx][self.a_col]
                    })
            return results
        except Exception as e:
            logger.error(f"Local Search Error: {e}")
            return None

    def _call_llm(self, query, context_text=""):
        """Call Online LLM (Groq > OpenAI > Gemini)"""
        
        system_prompt = f"""You are AgriBot, an agricultural expert. 
        Context information from our database is below. Use it to answer the farmer's question if relevant.
        If the context doesn't help, use your general knowledge about Indian agriculture.
        
        Context:
        {context_text}
        
        Question: {query}
        """

        # Groq
        if self.has_groq:
            try:
                from groq import Groq
                client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                resp = client.chat.completions.create(
                    model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                    messages=[{"role": "user", "content": system_prompt}]
                )
                return resp.choices[0].message.content
            except Exception as e:
                logger.error(f"Groq Error: {e}")

        # OpenAI
        if self.has_openai:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                resp = client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                    messages=[{"role": "user", "content": system_prompt}]
                )
                return resp.choices[0].message.content
            except Exception as e: 
                logger.error(f"OpenAI Error: {e}")

        # Gemini
        if self.has_gemini:
            try:
                import google.generativeai as genai
                genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
                model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
                resp = model.generate_content(system_prompt)
                return resp.text
            except Exception as e:
                logger.error(f"Gemini Error: {e}")

        return None

    def generate_response(self, query: str, lang: str = "en"):
        logger.info(f"Processing query: '{query}' [Lang: {lang}]")
        
        # 1. Translate to English if needed (assuming dataset/LLMs optimize for English)
        english_query = query
        if lang != 'en':
            english_query = self._translate(query, 'en')
            logger.info(f"Translated to English: {english_query}")

        # 2. Local RAG Search
        local_results = self._search_local(english_query)
        sources = []
        context_text = ""
        
        if local_results:
            sources = local_results
            context_text = "\\n".join([f"Q: {r['question']}\\nA: {r['answer']}" for r in local_results])

        # 3. Try Online LLM with Context
        response_text = self._call_llm(english_query, context_text)
        
        # 4. Fallback: If Online Failed, use Local Answer directly (Validation for Offline Mode)
        if not response_text:
            logger.info("Online LLM failed or not configured. Attempting local fallback.")
            if local_results:
                response_text = f"Based on our database: {local_results[0]['answer']}"
            else:
                response_text = "I'm sorry, I couldn't find an answer in my offline database and couldn't connect to online services."

        # 5. Translate back to Target Language
        if lang != 'en' and response_text:
            # deep-translator mapping: 'hi' for Hindi, 'te' for Telugu
            target_map = {'hi': 'hi', 'te': 'te', 'en': 'en'}
            target_code = target_map.get(lang, lang)
            response_text = self._translate(response_text, target_code)
            
        return {
            "response": response_text,
            "sources": sources
        }

rag_engine = RAGEngine()
