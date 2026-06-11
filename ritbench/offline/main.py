
import os
import sys
import multiprocessing
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from offline.gen import TaskAgent, OpenAIClient, load_config, get_datasets_from_config, get_task_templates_from_config

def process_task(task_args):
    try:
        dataset_path, template_path, generator_config, validator_configs, output_dir, api_key, base_url = task_args
        
        model_name = generator_config.get('model_name', 'gpt-4') if isinstance(generator_config, dict) else getattr(generator_config, 'model_name', 'gpt-4')
        generator = OpenAIClient(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url
        )
        
        validators = []
        for v_config in validator_configs:
            validators.append(OpenAIClient(
                model_name=v_config.get('model_name', 'gpt-3.5-turbo') if isinstance(v_config, dict) else getattr(v_config, 'model_name', 'gpt-3.5-turbo'),
                api_key=v_config.get('api_key', api_key) if isinstance(v_config, dict) else getattr(v_config, 'api_key', api_key),
                base_url=v_config.get('base_url', base_url) if isinstance(v_config, dict) else getattr(v_config, 'base_url', base_url)
            ))
        
        while len(validators) < 6:
            validators.append(OpenAIClient(
                model_name='gpt-3.5-turbo',
                api_key=api_key,
                base_url=base_url
            ))

        validators = validators[:6]

        agent = TaskAgent(
            generator_model=generator,
            validator_models=validators,
            task_template_path=template_path,
            output_dir=output_dir
        )
        
        dataset_name = os.path.basename(dataset_path)
        template_name = os.path.splitext(os.path.basename(template_path))[0]
        
        agent.run(dataset_path)
        
        return True
    except Exception as e:
        if len(task_args) >= 2:
            dataset_name = os.path.basename(task_args[0])
            template_name = os.path.splitext(os.path.basename(task_args[1]))[0]
        else:
            dataset_name = "unknown"
            template_name = "unknown"
        return False

def main():
    
    config = load_config()
    
    if config:
        config_type = "config.yaml" if os.path.exists(os.path.join(os.path.dirname(__file__), "config.yaml")) else "config.py"
        
        datasets = get_datasets_from_config(config)
        
        generator_config = None
        if hasattr(config, 'generator_config'):
            gc = config.generator_config
            generator_config = {
                'model_name': getattr(gc, 'model_name', 'gpt-4'),
                'api_key': getattr(gc, 'api_key', ''),
                'base_url': getattr(gc, 'base_url', '')
            }
        elif hasattr(config, 'GENERATOR_CONFIG'):
            generator_config = config.GENERATOR_CONFIG
        
        validator_configs = []
        if hasattr(config, 'validator_configs') and config.validator_configs:
            for vc in config.validator_configs:
                validator_configs.append({
                    'model_name': getattr(vc, 'model_name', 'gpt-3.5-turbo'),
                    'api_key': getattr(vc, 'api_key', ''),
                    'base_url': getattr(vc, 'base_url', '')
                })
        elif hasattr(config, 'VALIDATOR_CONFIGS') and config.VALIDATOR_CONFIGS:
            validator_configs = config.VALIDATOR_CONFIGS
        
        output_dir = None
        if hasattr(config, 'output_dir'):
            output_dir = config.output_dir
        elif hasattr(config, 'OUTPUT_DIR'):
            output_dir = config.OUTPUT_DIR
        output_dir = output_dir or "output"
        
        task_templates = get_task_templates_from_config(config)
        
        api_key = None
        if generator_config:
            api_key = generator_config.get('api_key', '') if isinstance(generator_config, dict) else getattr(generator_config, 'api_key', '')
        if not api_key:
            if hasattr(config, 'api_key'):
                api_key = config.api_key
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        
        base_url = None
        if generator_config:
            base_url = generator_config.get('base_url', '') if isinstance(generator_config, dict) else getattr(generator_config, 'base_url', '')
        if not base_url:
            if hasattr(config, 'base_url'):
                base_url = config.base_url
        if not base_url:
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        if not api_key:
            return
        
        model_name = 'gpt-4'
        if generator_config:
            model_name = generator_config.get('model_name', 'gpt-4') if isinstance(generator_config, dict) else getattr(generator_config, 'model_name', 'gpt-4')
        
        generator = OpenAIClient(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url
        )
        
        validators = []
        for v_config in validator_configs:
            validators.append(OpenAIClient(
                model_name=v_config.get('model_name', 'gpt-3.5-turbo') if isinstance(v_config, dict) else getattr(v_config, 'model_name', 'gpt-3.5-turbo'),
                api_key=v_config.get('api_key', api_key) if isinstance(v_config, dict) else getattr(v_config, 'api_key', api_key),
                base_url=v_config.get('base_url', base_url) if isinstance(v_config, dict) else getattr(v_config, 'base_url', base_url)
            ))
        
        while len(validators) < 6:
            validators.append(OpenAIClient(
                model_name='gpt-3.5-turbo',
                api_key=api_key,
                base_url=base_url
            ))

        validators = validators[:6]
        
    else:
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return
        
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        generator = OpenAIClient(
            model_name="gpt-4",
            api_key=api_key,
            base_url=base_url
        )
        
        generator_config = {
            'model_name': "gpt-4",
            'api_key': api_key,
            'base_url': base_url
        }
        
        validator_configs = [
            {
                'model_name': "gpt-3.5-turbo",
                'api_key': api_key,
                'base_url': base_url
            } for _ in range(6)
        ]
        
        validators = [
            OpenAIClient(
                model_name="gpt-3.5-turbo",
                api_key=api_key,
                base_url=base_url
            ) for _ in range(6)
        ]
        
        datasets = ["data/Standard/airbnb.csv"]
        task_templates = ["scripts/single_task/sm-s-e.txt"]
        output_dir = "output"
    
    if not datasets:
        return
    
    if not task_templates:
        return
    
    
    dataset_counts = {
        'Large': 3,
        'Standard': 8,
        'Modest': 3,
        'Wide': 2
    }
    
    datasets_by_category = {'Large': [], 'Standard': [], 'Modest': [], 'Wide': []}
    for dataset_path in datasets:
        if not os.path.exists(dataset_path):
            continue
        
        path_parts = dataset_path.split(os.sep)
        for category in ['Large', 'Standard', 'Modest', 'Wide']:
            if category in path_parts:
                datasets_by_category[category].append(dataset_path)
                break
    
    for category, dataset_list in datasets_by_category.items():
        pass
    
    tasks = []
    skipped_tasks = 0
    
    for template_path in task_templates:
        if not os.path.exists(template_path):
            continue
        
        template_name = os.path.splitext(os.path.basename(template_path))[0]
        if '-' in template_name:
            template_category = template_name.split('-')[0]
        else:
            template_category = template_name
        
        task_root_dir = os.path.join(output_dir, "task", template_category, template_name)
        existing_datasets_by_category = {'Large': set(), 'Standard': set(), 'Modest': set(), 'Wide': set()}
        
        if os.path.exists(task_root_dir):
            for dir_name in os.listdir(task_root_dir):
                if f"_{template_name}_" in dir_name or dir_name.endswith(f"_{template_name}"):
                    parts = dir_name.split('_')
                    if len(parts) >= 2:
                        dataset_name = parts[0]
                        for category in ['Large', 'Standard', 'Modest', 'Wide']:
                            if dataset_name in [os.path.splitext(os.path.basename(d))[0] for d in datasets_by_category[category]]:
                                existing_datasets_by_category[category].add(dataset_name)
                                break
        
        import random
        for category, target_count in dataset_counts.items():
            existing_count = len(existing_datasets_by_category[category])
            
            if existing_count >= target_count:
                continue
            
            needed_count = target_count - existing_count
            
            available_datasets = datasets_by_category[category]
            if not available_datasets:
                continue
            
            unused_datasets = []
            for dataset_path in available_datasets:
                dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
                if dataset_name not in existing_datasets_by_category[category]:
                    unused_datasets.append(dataset_path)
            
            if len(unused_datasets) < needed_count:
                selected_datasets = unused_datasets
            else:
                selected_datasets = random.sample(unused_datasets, needed_count)
            
            for dataset_path in selected_datasets:
                dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
                tasks.append((dataset_path, template_path))
    
    if not tasks:
        return
    
    if skipped_tasks > 0:
        pass
    else:
        pass

    
    max_workers = min(multiprocessing.cpu_count(), 4)
    
    with multiprocessing.Pool(processes=max_workers) as pool:
        task_args = []
        for dataset_path, template_path in tasks:
            task_args.append((
                dataset_path,
                template_path,
                generator_config,
                validator_configs,
                output_dir,
                api_key,
                base_url
            ))
        
        results = []
        with tqdm(total=len(task_args), desc="Processing tasks", unit="task") as pbar:
            for result in pool.imap_unordered(process_task, task_args):
                results.append(result)
                pbar.update(1)
    
    success_count = sum(results)
    total_count = len(results)
    


if __name__ == "__main__":
    main()
