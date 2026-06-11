import os
import json
import logging
import sys
import random
import multiprocessing
from typing import Dict, List, Optional
from datetime import datetime
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from offline.gen import OpenAIClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OnlineTaskGenerator:
    
    def __init__(self, 
                 model_name: str,
                 api_key: str,
                 base_url: str,
                 output_dir: str = "output/online_task",
                 scripts_dir: str = "scripts/online_task",
                 difficulty_level: int = 1):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.output_dir = output_dir
        self.scripts_dir = scripts_dir
        self.difficulty_level = difficulty_level
        
        self.llm_client = OpenAIClient(model_name=model_name, api_key=api_key, base_url=base_url)
        
        self.organize_template_path = os.path.join(scripts_dir, "organize.txt")
        self.colloquialism_template_path = os.path.join(scripts_dir, "colloquialism.txt")
        self.redundant_template_path = os.path.join(scripts_dir, "redundant.txt")
        self.modify_template_path = os.path.join(scripts_dir, "modify.txt")
    
    def load_organize_template(self) -> str:
        with open(self.organize_template_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def load_colloquialism_template(self) -> str:
        with open(self.colloquialism_template_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def load_redundant_template(self) -> str:
        with open(self.redundant_template_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def load_modify_template(self) -> str:
        with open(self.modify_template_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def get_task_folders(self) -> List[str]:
        task_folders = []
        if os.path.exists(self.output_dir):
            for item in os.listdir(self.output_dir):
                item_path = os.path.join(self.output_dir, item)
                if os.path.isdir(item_path):
                    task_folders.append(item_path)
        return sorted(task_folders)
    
    def generate_organize_json(self, task_folder: str, max_retries: int = 5) -> bool:
        import time
        
        task_json_path = os.path.join(task_folder, "task.json")
        ans_py_path = os.path.join(task_folder, "ans.py")
        
        if not os.path.exists(task_json_path):
            logger.warning(f"task.json does not exist in task folder: {task_folder}")
            return False
        
        if not os.path.exists(ans_py_path):
            logger.warning(f"ans.py does not exist in task folder: {task_folder}")
            return False
        
        try:
            with open(task_json_path, 'r', encoding='utf-8') as f:
                task_data = json.load(f)
            
            with open(ans_py_path, 'r', encoding='utf-8') as f:
                ans_code = f.read()
            
            task_desc = task_data.get("task", {}).get("target_en", "")
            
            template = self.load_organize_template()
            prompt = template.format(task_desc, ans_code)
            
            for attempt in range(max_retries):
                try:
                    response, _ = self.llm_client.call(prompt)
                    
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    
                    if json_start >= 0 and json_end > json_start:
                        json_str = response[json_start:json_end]
                        organize_data = json.loads(json_str)
                        
                        organize_json_path = os.path.join(task_folder, "organize.json")
                        with open(organize_json_path, 'w', encoding='utf-8') as f:
                            json.dump(organize_data, f, ensure_ascii=False, indent=2)
                        
                        logger.info(f"Successfully generated organize.json: {task_folder}")
                        return True
                    else:
                        logger.warning(f"Attempt {attempt+1}/{max_retries}: Cannot extract JSON from response")
                        if attempt < max_retries - 1:
                            prompt += "\n\nError feedback: Must return valid JSON format. Please regenerate."
                
                except json.JSONDecodeError as e:
                    logger.warning(f"Attempt {attempt+1}/{max_retries}: JSON parsing failed - {str(e)}")
                    if attempt < max_retries - 1:
                        prompt += "\n\nError feedback: JSON format error. Please regenerate."
                        wait_time = min(2 ** attempt, 10)
                        logger.info(f"Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                except Exception as e:
                    logger.warning(f"Attempt {attempt+1}/{max_retries}: Call failed - {str(e)}")
                    if attempt < max_retries - 1:
                        prompt += f"\n\nError feedback: {str(e)}\nPlease regenerate."
                        wait_time = min(2 ** attempt, 10)
                        logger.info(f"Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
            
            logger.error(f"Failed to generate organize.json: {task_folder}")
            return False
        
        except Exception as e:
            logger.error(f"Error processing task folder {task_folder}: {str(e)}")
            return False
    
    def apply_fuzzy_perturbation(self, tasks: List[Dict], task_index: int) -> List[Dict]:
        if task_index >= len(tasks):
            return tasks
        
        original_task = tasks[task_index]
        
        clarification_task = {
            "category": "Clarification",
            "summary": f"Task clarification: {original_task.get('summary', '')}",
            "rules": [f"Original task rules: {', '.join(original_task.get('rules', []))}"],
            "is_protected": True,
            "clarified_task_summary": original_task.get("summary", "")
        }
        
        fuzzy_task = {
            "category": original_task.get("category", ""),
            "summary": original_task.get("summary", ""),
            "rules": ["Please execute operations according to task description"]
        }
        
        new_tasks = [clarification_task] + tasks[:task_index] + [fuzzy_task] + tasks[task_index+1:]
        return new_tasks
    
    def apply_bias_perturbation(self, tasks: List[Dict], task_index: int, max_retries: int = 5) -> List[Dict]:
        import time
        
        if task_index >= len(tasks):
            return tasks
        
        original_task = tasks[task_index]
        
        modification_task = {
            "category": "Modification",
            "summary": f"Task modification: {original_task.get('summary', '')}",
            "rules": [f"Original task rules: {', '.join(original_task.get('rules', []))}"],
            "is_protected": True,
            "modified_task_summary": original_task.get("summary", "")
        }
        
        task_desc = json.dumps(original_task, ensure_ascii=False, indent=2)
        template = self.load_modify_template()
        prompt = template.format(task_desc)
        
        modified_task = None
        for attempt in range(max_retries):
            try:
                response, _ = self.llm_client.call(prompt)
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    modified_task = json.loads(json_str)
                    break
            except Exception as e:
                if attempt < max_retries - 1:
                    prompt += f"\n\nError feedback: {str(e)}\nPlease regenerate."
                    wait_time = min(2 ** attempt, 10)
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
        
        if modified_task is None:
            modified_task = original_task
        
        new_tasks = [modification_task] + tasks[:task_index] + [modified_task] + tasks[task_index+1:]
        return new_tasks
    
    def apply_redundant_perturbation(self, tasks: List[Dict], insert_index: int, max_retries: int = 5) -> List[Dict]:
        import time
        
        if insert_index > len(tasks):
            insert_index = len(tasks)
        
        prev_task = tasks[insert_index - 1] if insert_index > 0 else tasks[0]
        task_desc = json.dumps(prev_task, ensure_ascii=False, indent=2)
        template = self.load_redundant_template()
        prompt = template.format(task_desc)
        
        new_task = None
        for attempt in range(max_retries):
            try:
                response, _ = self.llm_client.call(prompt)
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    new_task = json.loads(json_str)
                    break
            except Exception as e:
                if attempt < max_retries - 1:
                    prompt += f"\n\nError feedback: {str(e)}\nPlease regenerate."
                    wait_time = min(2 ** attempt, 10)
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
        
        if new_task is None:
            return tasks
        
        new_tasks = tasks[:insert_index] + [new_task] + tasks[insert_index:]
        return new_tasks
    
    def apply_order_perturbation(self, tasks: List[Dict], index1: int, index2: int) -> List[Dict]:
        if index1 >= len(tasks) or index2 >= len(tasks) or index1 == index2:
            return tasks
        
        task1 = tasks[index1]
        task2 = tasks[index2]
        
        order_adjustment_task = {
            "category": "OrderAdjustment",
            "summary": "Task order adjustment",
            "rules": [f"Original order: Task{index1+1}({task1.get('summary', '')}) -> Task{index2+1}({task2.get('summary', '')})"],
            "is_protected": True,
            "swapped_task1_summary": task1.get("summary", ""),
            "swapped_task2_summary": task2.get("summary", "")
        }
        
        new_tasks = tasks.copy()
        new_tasks[index1], new_tasks[index2] = new_tasks[index2], new_tasks[index1]
        
        new_tasks = [order_adjustment_task] + new_tasks
        return new_tasks
    
    def apply_random_perturbation(self, tasks: List[Dict], max_retries: int = 5) -> List[Dict]:
        if len(tasks) < 1:
            return tasks
        
        perturbation_types = ["fuzzy", "bias", "redundant", "order"]
        perturbation_type = random.choice(perturbation_types)
        
        non_protected_indices = [i for i, task in enumerate(tasks) if not task.get("is_protected", False)]
        
        if not non_protected_indices:
            return tasks
        
        task_index = random.choice(non_protected_indices)
        
        if perturbation_type == "fuzzy":
            return self.apply_fuzzy_perturbation(tasks, task_index)
        elif perturbation_type == "bias":
            return self.apply_bias_perturbation(tasks, task_index, max_retries)
        elif perturbation_type == "redundant":
            insert_index = random.randint(0, len(tasks))
            return self.apply_redundant_perturbation(tasks, insert_index, max_retries)
        elif perturbation_type == "order":
            if len(non_protected_indices) >= 2:
                index1 = random.choice(non_protected_indices)
                remaining_indices = [i for i in non_protected_indices if i != index1]
                index2 = random.choice(remaining_indices)
                return self.apply_order_perturbation(tasks, index1, index2)
        
        return tasks
    
    def get_perturbation_types_by_level(self) -> List[str]:
        if self.difficulty_level == 1:
            return []
        elif self.difficulty_level == 2:
            return ["fuzzy", "bias"]
        elif self.difficulty_level in [3, 4, 5]:
            return ["fuzzy", "bias", "redundant", "order"]
        else:
            return []
    
    def should_apply_perturbation_by_level(self, batch: List[Dict], perturbation_prob: float) -> bool:
        if self.difficulty_level == 1:
            return False
        
        if random.random() >= perturbation_prob:
            return False
        
        if len(batch) < 1:
            return False
        
        return True
    
    def should_apply_perturbation(self, batch: List[Dict], perturbation_prob: float) -> bool:
        if self.difficulty_level == 1:
            return False
        
        if random.random() >= perturbation_prob:
            return False
        
        if len(batch) < 1:
            return False
        
        return True
    
    def get_perturbation_task_index(self, batch: List[Dict]) -> Optional[int]:
        non_protected_indices = [i for i, task in enumerate(batch) if not task.get("is_protected", False)]
        
        if not non_protected_indices:
            return None
        
        if self.difficulty_level in [2, 3]:
            return non_protected_indices[-1]
        elif self.difficulty_level in [4, 5]:
            return random.choice(non_protected_indices)
        
        return None
    
    def apply_perturbation_by_level(self, batch: List[Dict], max_retries: int = 5) -> Tuple[List[Dict], List[Dict]]:
        perturbation_types = self.get_perturbation_types_by_level()
        
        if not perturbation_types:
            return batch, []
        
        task_index = self.get_perturbation_task_index(batch)
        
        if task_index is None:
            return batch, []
        
        perturbation_type = random.choice(perturbation_types)
        
        if perturbation_type == "fuzzy":
            perturbed_batch = self.apply_fuzzy_perturbation(batch, task_index)
            clarification_tasks = [perturbed_batch[0]] if perturbed_batch and perturbed_batch[0].get("category") == "Clarification" else []
            return perturbed_batch, clarification_tasks
        elif perturbation_type == "bias":
            perturbed_batch = self.apply_bias_perturbation(batch, task_index, max_retries)
            clarification_tasks = [perturbed_batch[0]] if perturbed_batch and perturbed_batch[0].get("category") == "Modification" else []
            return perturbed_batch, clarification_tasks
        elif perturbation_type == "redundant":
            insert_index = random.randint(0, len(batch))
            perturbed_batch = self.apply_redundant_perturbation(batch, insert_index, max_retries)
            clarification_tasks = [perturbed_batch[0]] if perturbed_batch and perturbed_batch[0].get("category") in ["Clarification", "Modification"] else []
            return perturbed_batch, clarification_tasks
        elif perturbation_type == "order":
            non_protected_indices = [i for i, task in enumerate(batch) if not task.get("is_protected", False)]
            if len(non_protected_indices) >= 2:
                if self.difficulty_level in [3, 4, 5]:
                    last_two = non_protected_indices[-2:] if len(non_protected_indices) >= 2 else non_protected_indices
                    if len(last_two) >= 2:
                        index1, index2 = random.sample(last_two, 2)
                        perturbed_batch = self.apply_order_perturbation(batch, index1, index2)
                        clarification_tasks = [perturbed_batch[0]] if perturbed_batch and perturbed_batch[0].get("category") == "OrderAdjustment" else []
                        return perturbed_batch, clarification_tasks
        
        return batch, []
    
    def should_add_extra_task(self, batch: List[Dict]) -> bool:
        if self.difficulty_level != 5:
            return False
        
        if not batch:
            return False
        
        first_task = batch[0]
        if first_task.get("category") in ["Clarification", "Modification", "OrderAdjustment"]:
            return True
        
        return False
    
    def add_extra_task_with_perturbation(self, batch: List[Dict], all_tasks: List[Dict], max_retries: int = 5) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        if not self.should_add_extra_task(batch):
            return batch, [], all_tasks
        
        remaining_tasks = all_tasks
        
        if not remaining_tasks:
            return batch, [], all_tasks
        
        return self._add_single_extra_task_with_perturbation(batch, remaining_tasks, max_retries)
    
    def _add_single_extra_task_with_perturbation(self, batch: List[Dict], remaining_tasks: List[Dict], max_retries: int = 5) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        next_task_index = 0
        for i, task in enumerate(remaining_tasks):
            if not task.get("is_protected", False):
                next_task_index = i
                break
        
        if next_task_index >= len(remaining_tasks):
            return batch, [], remaining_tasks
        
        extra_task = remaining_tasks[next_task_index]
        
        perturbation_types = self.get_perturbation_types_by_level()
        if perturbation_types:
            perturbation_type = random.choice(perturbation_types)
            
            if perturbation_type == "fuzzy":
                perturbed_batch = self.apply_fuzzy_perturbation(batch + [extra_task], len(batch))
                clarification_tasks = [perturbed_batch[0]] if perturbed_batch and perturbed_batch[0].get("category") == "Clarification" else []
                updated_remaining = remaining_tasks[:next_task_index] + remaining_tasks[next_task_index+1:]
                return perturbed_batch, clarification_tasks, updated_remaining
            elif perturbation_type == "bias":
                perturbed_batch = self.apply_bias_perturbation(batch + [extra_task], len(batch), max_retries)
                clarification_tasks = [perturbed_batch[0]] if perturbed_batch and perturbed_batch[0].get("category") == "Modification" else []
                updated_remaining = remaining_tasks[:next_task_index] + remaining_tasks[next_task_index+1:]
                return perturbed_batch, clarification_tasks, updated_remaining
            elif perturbation_type == "redundant":
                insert_index = random.randint(0, len(batch))
                perturbed_batch = self.apply_redundant_perturbation(batch + [extra_task], insert_index, max_retries)
                clarification_tasks = [perturbed_batch[0]] if perturbed_batch and perturbed_batch[0].get("category") in ["Clarification", "Modification"] else []
                updated_remaining = remaining_tasks[:next_task_index] + remaining_tasks[next_task_index+1:]
                return perturbed_batch, clarification_tasks, updated_remaining
            elif perturbation_type == "order":
                non_protected_indices = [i for i, task in enumerate(batch + [extra_task]) if not task.get("is_protected", False)]
                if len(non_protected_indices) >= 2:
                    last_two = non_protected_indices[-2:] if len(non_protected_indices) >= 2 else non_protected_indices
                    if len(last_two) >= 2:
                        index1, index2 = random.sample(last_two, 2)
                        perturbed_batch = self.apply_order_perturbation(batch + [extra_task], index1, index2)
                        clarification_tasks = [perturbed_batch[0]] if perturbed_batch and perturbed_batch[0].get("category") == "OrderAdjustment" else []
                        updated_remaining = remaining_tasks[:next_task_index] + remaining_tasks[next_task_index+1:]
                        return perturbed_batch, clarification_tasks, updated_remaining
        
        updated_remaining = remaining_tasks[:next_task_index] + remaining_tasks[next_task_index+1:]
        return batch + [extra_task], [], updated_remaining
    
    def generate_online_task_json(self, folder: str, K: int = 2, max_retries: int = 5, perturbation_prob: float = 0.3) -> bool:
        online_task_filename = f"online_task_l{self.difficulty_level}.json"
        online_task_json_path = os.path.join(folder, online_task_filename)
        
        if os.path.exists(online_task_json_path):
            logger.info(f"File already exists, skipping generation: {online_task_filename}")
            return True
        
        organize_json_path = os.path.join(folder, "organize.json")
        
        if not os.path.exists(organize_json_path):
            logger.error(f"organize.json does not exist: {folder}")
            return False
        
        try:
            with open(organize_json_path, 'r', encoding='utf-8') as f:
                organize_data = json.load(f)
            
            tasks = organize_data.get("target_en", [])
            
            if not tasks:
                logger.warning(f"target_en is empty: {folder}")
                return False
            
            template = self.load_colloquialism_template()
            turns = []
            
            task_queue = tasks.copy()
            window = []
            
            while len(task_queue) > 0:
                if self.difficulty_level in [2, 3, 4] and task_queue and task_queue[0].get("category") in ["Clarification", "Modification", "OrderAdjustment"]:
                    num_to_take = 1
                    should_perturb = False
                else:
                    k = random.randint(2,3)
                    num_to_take = min(k, len(task_queue))
                    taken_tasks = task_queue[:num_to_take]
                    should_perturb = self.should_apply_perturbation(taken_tasks, perturbation_prob)
                
                taken_tasks = task_queue[:num_to_take]
                task_queue = task_queue[num_to_take:]
                
                window.extend(taken_tasks)
                
                if should_perturb:
                    perturbed_window, clarification_tasks = self.apply_perturbation_by_level(window, max_retries)
                    
                    if clarification_tasks:
                        for clarification_task in reversed(clarification_tasks):
                            task_queue.insert(0, clarification_task)
                else:
                    perturbed_window = window
                
                task_content = "\n\n".join([self._format_task_for_prompt(task) for task in perturbed_window])
                prompt = template.format(task_content)
                
                for attempt in range(max_retries):
                    try:
                        response, _ = self.llm_client.call(prompt)
                        
                        json_start = response.find('{')
                        json_end = response.rfind('}') + 1
                        
                        if json_start >= 0 and json_end > json_start:
                            json_str = response[json_start:json_end]
                            result = json.loads(json_str)
                            turns.append(result)
                            break
                        else:
                            logger.warning(f"Attempt {attempt+1}/{max_retries}: Cannot extract JSON from response")
                            if attempt < max_retries - 1:
                                prompt += "\n\nError feedback: Must return valid JSON format. Please regenerate."
                    
                    except json.JSONDecodeError as e:
                        logger.warning(f"Attempt {attempt+1}/{max_retries}: JSON parsing failed - {str(e)}")
                        if attempt < max_retries - 1:
                            prompt += "\n\nError feedback: JSON format error. Please regenerate."
                    except Exception as e:
                        logger.warning(f"Attempt {attempt+1}/{max_retries}: Call failed - {str(e)}")
                        if attempt < max_retries - 1:
                            prompt += f"\n\nError feedback: {str(e)}\nPlease regenerate."
                
                window = []
            
            online_task_data = {"task": turns}
            
            with open(online_task_json_path, 'w', encoding='utf-8') as f:
                json.dump(online_task_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Successfully generated {online_task_filename}: {folder}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to generate online_task.json {folder}: {str(e)}")
            return False
    
    def _format_task_for_prompt(self, task: Dict, full_tasks: Optional[List[Dict]] = None) -> str:
        category = task.get("category", "")
        summary = task.get("summary", "")
        rules = task.get("rules", [])
        
        task_str = f"Category: {category}\n"
        task_str += f"Summary: {summary}\n"
        task_str += f"Rules:\n"
        for rule in rules:
            task_str += f"  - {rule}\n"
        
        if "clarified_task_summary" in task:
            task_str += f"Clarified task summary: {task['clarified_task_summary']}\n"
        if "modified_task_summary" in task:
            task_str += f"Modified task summary: {task['modified_task_summary']}\n"
        if "swapped_task1_summary" in task:
            task_str += f"Swapped task 1 summary: {task['swapped_task1_summary']}\n"
        if "swapped_task2_summary" in task:
            task_str += f"Swapped task 2 summary: {task['swapped_task2_summary']}\n"
        
        return task_str.strip()
    
    def run(self, folder: str, K: int = 5, perturbation_prob: float = 0.3, difficulty_level: int = None) -> bool:
        logger.info("=" * 50)
        logger.info(f"Starting to process folder: {folder}")
        logger.info("=" * 50)
        
        if difficulty_level is not None:
            self.difficulty_level = difficulty_level
        
        logger.info(f"Difficulty level: Level{self.difficulty_level}")
        
        success = self.generate_online_task_json(folder, K=K, perturbation_prob=perturbation_prob)
        
        return success
