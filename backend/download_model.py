"""Script to pre-download the embedding model before FastAPI starts"""

import os
import sys
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model():
    """Download the embedding model synchronously and ensure all files are cached"""
    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Always disable SSL verification for Docker builds (corporate proxy issues)
    # This is safe in a controlled Docker environment
    logger.info("Configuring SSL settings for HuggingFace downloads...")
    os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["REQUESTS_CA_BUNDLE"] = ""
    
    # Disable SSL verification at Python level
    import ssl
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    ssl._create_default_https_context = ssl._create_unverified_context
    
    # Clear any stale httpx clients before starting
    try:
        import huggingface_hub
        if hasattr(huggingface_hub, '_CACHED_CLIENTS'):
            huggingface_hub._CACHED_CLIENTS.clear()
    except:
        pass
    
    logger.info(f"Downloading and caching embedding model: {model_name}")
    
    # Set cache directory explicitly
    cache_dir = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    logger.info(f"Using cache directory: {cache_dir}")
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries}: Loading model...")
            
            # Clear httpx clients before each attempt
            try:
                import huggingface_hub
                if hasattr(huggingface_hub, '_CACHED_CLIENTS'):
                    huggingface_hub._CACHED_CLIENTS.clear()
            except:
                pass
            
            from sentence_transformers import SentenceTransformer
            
            # Load model - this will download ALL files if needed
            # Using cache_folder ensures files are saved to a persistent location
            model = SentenceTransformer(model_name, cache_folder=cache_dir)
            
            # Force model to load all components by generating a dummy embedding
            # This ensures all files (including adapter configs) are downloaded
            logger.info("Generating test embedding to ensure all model files are cached...")
            _ = model.encode(["test"], normalize_embeddings=True)
            
            logger.info(f"Model downloaded and fully cached successfully: {model_name}")
            logger.info(f"Model dimension: {model.get_sentence_embedding_dimension()}")
            return True
            
        except RuntimeError as e:
            error_str = str(e).lower()
            if "client has been closed" in error_str and attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s, 6s, 8s
                logger.warning(f"httpx client closed error (attempt {attempt + 1}/{max_retries}), waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                # Clear clients again before retry
                try:
                    import huggingface_hub
                    if hasattr(huggingface_hub, '_CACHED_CLIENTS'):
                        huggingface_hub._CACHED_CLIENTS.clear()
                except:
                    pass
                continue
            else:
                logger.error(f"Failed to download model after {max_retries} attempts: {e}", exc_info=True)
                return False
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                logger.warning(f"Error downloading model (attempt {attempt + 1}/{max_retries}): {e}")
                logger.info(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"Failed to download model after {max_retries} attempts: {e}", exc_info=True)
                return False
    
    return False

if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)

