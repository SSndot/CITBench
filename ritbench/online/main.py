import os
import sys
import multiprocessing
import warnings
import logging
from typing import List, Tuple
from tqdm import tqdm

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('online_run.log', encoding='utf-8'),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from online.gen import OnlineTaskGenerator

def load_config():
    config_dir = os.path.dirname(__file__)
    
    yaml_path = os.path.join(config_dir, "config.yaml")
    if os.path.exists(yaml_path):
        try:
            import yaml
            with open(yaml_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
            
            if config_dict:
                api_key = config_dict.get('api_key', '')
                base_url = config_dict.get('base_url', '')
                
                def replace_vars(obj):
                    if isinstance(obj, dict):
                        return {k: replace_vars(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [replace_vars(item) for item in obj]
                    elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
                        var_name = obj[2:-1]
                        if var_name == 'api_key':
                            return api_key
                        elif var_name == 'base_url':
                            return base_url
                        return obj
                    return obj
                
                config_dict = replace_vars(config_dict)
            
            class ConfigObject:
                def __init__(self, d):
                    for k, v in d.items():
                        if isinstance(v, dict):
                            setattr(self, k, ConfigObject(v))
                        elif isinstance(v, list):
                            setattr(self, k, [ConfigObject(item) if isinstance(item, dict) else item for item in v])
                        else:
                            setattr(self, k, v)
                
                def get(self, key, default=None):
                    return getattr(self, key, default)
            
            return ConfigObject(config_dict)
        except ImportError:
            pass
        except Exception as e:
            pass
    
    py_path = os.path.join(config_dir, "config.py")
    if os.path.exists(py_path):
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("config", py_path)
            config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config)
            return config
        except Exception as e:
            pass
    
    return None


def process_folder(folder_args: Tuple[str, str, str, str, str, float, int]) -> bool:
    try:
        folder, K, perturbation_prob, api_key, base_url, model_name, difficulty_level = folder_args
        
        generator = OnlineTaskGenerator(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            difficulty_level=difficulty_level
        )
        
        return generator.run(folder, K=int(K), perturbation_prob=perturbation_prob)
    except Exception as e:
        logger.error(f"Failed to process folder {folder}: {str(e)}")
        return False


def main():
    
    config = load_config()
    
    if config is None:
        logger.error("Failed to load configuration file")
        return
    
    api_key = config.get("api_key", "")
    base_url = config.get("base_url", "")
    model_name = config.get("generator_config", {}).get("model_name", "gpt-4")
    
    if not api_key:
        logger.error("Please configure api_key in config.yaml")
        return
    
    online_config = config.get("online_config", {})
    K = online_config.get("K", 5)
    perturbation_prob = online_config.get("perturbation_prob", 0.3)
    max_workers = online_config.get("max_workers", min(multiprocessing.cpu_count(), 10))
    difficulty_level = online_config.get("difficulty_level", 1)
    
    logger.info("=" * 60)
    logger.info("Online Task Generator")
    logger.info("=" * 60)
    logger.info(f"Model: {model_name}")
    logger.info(f"API: {base_url}")
    logger.info(f"K: {K}")
    logger.info(f"Perturbation probability: {perturbation_prob}")
    logger.info(f"Difficulty level: Level{difficulty_level}")
    logger.info(f"Max concurrency: {max_workers}")
    logger.info("=" * 60)
    
    generator = OnlineTaskGenerator(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url
    )
    
    task_folders = generator.get_task_folders()
    logger.info(f"Found {len(task_folders)} task folders")
    
    if len(task_folders) == 0:
        logger.warning("No task folders found")
        return
    
    success_count = 0
    failed_count = 0
    
    with multiprocessing.Pool(processes=max_workers) as pool:
        folder_args = [(folder, K, perturbation_prob, api_key, base_url, model_name, difficulty_level) for folder in task_folders]
        results = list(tqdm(
            pool.imap(process_folder, folder_args),
            total=len(folder_args),
            desc="Processing task folders"
        ))
        
        for result in results:
            if result:
                success_count += 1
            else:
                failed_count += 1
    
    logger.info("=" * 60)
    logger.info(f"Processing completed: Success {success_count}, Failed {failed_count}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
