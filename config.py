"""Environment-driven settings for PaperPilot."""

import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
QUERY_PREFIX = os.getenv("QUERY_PREFIX", "")
TOP_K = int(os.getenv("TOP_K", "5"))
SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", "0.8"))
KEYWORD_WEIGHT = float(os.getenv("KEYWORD_WEIGHT", "0.2"))
MIN_FINAL_SCORE = float(os.getenv("MIN_FINAL_SCORE", "0.15"))

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "20"))
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "10"))
RERANK_THRESHOLD = float(os.getenv("RERANK_THRESHOLD", "-8.0"))

SECTION_AWARE_RETRIEVAL = os.getenv("SECTION_AWARE_RETRIEVAL", "on").lower() == "on"
SECTION_BOOST_MAX_EXTRA = int(os.getenv("SECTION_BOOST_MAX_EXTRA", "12"))
MIN_SEMANTIC_SCORE = float(os.getenv("MIN_SEMANTIC_SCORE", "0.15"))
MAX_PAGES = int(os.getenv("MAX_PAGES", "300"))
