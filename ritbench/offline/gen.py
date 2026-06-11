import os
import json
import pandas as pd
import subprocess
import sys
import warnings
from typing import Dict, List, Tuple, Optional
from io import StringIO
import traceback
from datetime import datetime
import re

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import analyze_csv_file


class LLMClient:
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
    
    def call(self, prompt: str, max_retries: int = 3) -> Tuple[str, int]:
        raise NotImplementedError("Subclass must implement the call method")


class OpenAIClient(LLMClient):
    
    def call(self, prompt: str, max_retries: int = 3) -> Tuple[str, int]:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a professional code generation assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10000
            )
            
            if not response or not response.choices or not response.choices[0].message:
                raise Exception(f"API returned empty response")
            
            content = response.choices[0].message.content
            if not content:
                raise Exception(f"API returned empty content")
            
            token_count = response.usage.total_tokens if response.usage else 0
            
            return content, token_count
        except Exception as e:
            raise


class DatasetAnalyzer:
    
    @staticmethod
    def generate_description(csv_path: str, encoding: str = 'utf-8') -> str:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File not found: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(csv_path, encoding='gbk')
            except:
                df = pd.read_csv(csv_path, encoding='latin-1')
        except Exception as e:
            raise Exception(f"Failed to read file: {str(e)}")
        
        description = []
        description.append(f"Dataset file: {os.path.basename(csv_path)}")
        description.append(f"Total rows: {len(df)}")
        description.append(f"Total columns: {len(df.columns)}")
        description.append(f"\nColumn names: {list(df.columns)}")

        description.append("\nData sample (row 1):")
        description.append(df.iloc[0].to_string())
        
        return "\n".join(description)


class CodeExecutor:
    
    @staticmethod
    def execute_code(code: str, working_dir: str, input_files: List[str] = None, original_dataset: str = None) -> Tuple[bool, str, Dict[str, pd.DataFrame]]:
        old_cwd = os.getcwd()
        
        try:
            if not os.path.isabs(working_dir):
                working_dir = os.path.abspath(working_dir)
            
            os.makedirs(working_dir, exist_ok=True)
            os.chdir(working_dir)
            
            exec_globals = {
                '__builtins__': __builtins__,
                'pd': pd,
                'np': __import__('numpy'),
                'numpy': __import__('numpy'),
                'os': os,
                'json': json,
                're': __import__('re'),
                'sklearn': __import__('sklearn'),
                'tqdm': __import__('tqdm'),
                'datetime': __import__('datetime'),
                'scipy': __import__('scipy'),
                'math': __import__('math'),
                'random': __import__('random'),
                'collections': __import__('collections'),
                'seaborn': __import__('seaborn'),
                '__file__': os.path.join(working_dir, '_temp_exec.py')
            }
            exec_locals = {}
            
            if original_dataset:
                if os.path.exists(original_dataset):
                    try:
                        df_original = pd.read_csv(original_dataset)
                        exec_globals['df'] = df_original
                        exec_globals['df_original'] = df_original
                    except:
                        pass
                elif os.path.exists(os.path.join(working_dir, os.path.basename(original_dataset))):
                    try:
                        rel_path = os.path.join(working_dir, os.path.basename(original_dataset))
                        df_original = pd.read_csv(rel_path)
                        exec_globals['df'] = df_original
                        exec_globals['df_original'] = df_original
                    except:
                        pass
            
            if input_files:
                for i, file_path in enumerate(input_files):
                    if os.path.exists(file_path):
                        df_name = f"df_input_{i+1}"
                        try:
                            exec_globals[df_name] = pd.read_csv(file_path)
                        except:
                            pass
                    rel_path = os.path.basename(file_path)
                    if os.path.exists(rel_path):
                        df_name = f"df_input_{i+1}"
                        try:
                            exec_globals[df_name] = pd.read_csv(rel_path)
                        except:
                            pass
            
            exec(code, exec_globals, exec_locals)
            
            if not os.path.exists(working_dir):
                os.makedirs(working_dir, exist_ok=True)
            os.chdir(working_dir)
            
            output_files = {}
            current_dir = os.getcwd()
            if not os.path.exists(current_dir):
                current_dir = working_dir
            
            for filename in os.listdir(current_dir):
                if filename.endswith('.csv') and not filename.startswith('_'):
                    is_input = False
                    if input_files:
                        for inp_file in input_files:
                            if os.path.basename(inp_file) == filename:
                                is_input = True
                                break
                    if not is_input:
                        filepath = os.path.join(current_dir, filename)
                        try:
                            df = pd.read_csv(filepath)
                            if not df.empty:
                                output_files[filename] = df
                        except:
                            pass
            
            for namespace in [exec_locals, exec_globals]:
                for key, value in namespace.items():
                    if isinstance(value, pd.DataFrame) and not value.empty:
                        csv_name = f"{key}.csv"
                        if csv_name not in output_files:
                            output_files[csv_name] = value
            
            if output_files:
                return True, "", output_files
            else:
                return False, "Code executed successfully but no CSV files were generated", {}
                
        except Exception as e:
            error_msg = f"Code execution failed: {str(e)}\n{traceback.format_exc()}"
            return False, error_msg, {}
        finally:
            os.chdir(old_cwd)


class TaskAgent:
    
    def __init__(self, 
                 generator_model: LLMClient,
                 validator_models: List[LLMClient],
                 task_template_path: str,
                 output_dir: str = "output"):
        self.generator_model = generator_model
        self.validator_models = validator_models
        self.task_template_path = task_template_path
        self.output_dir = output_dir
        self.code_executor = CodeExecutor()
        
        self.tmp_dir = os.path.join(output_dir, "tmp")
        self.task_dir = os.path.join(output_dir, "task")
        os.makedirs(self.tmp_dir, exist_ok=True)
        os.makedirs(self.task_dir, exist_ok=True)

        self.current_tmp_input_dir = None
    
    def get_task_template_name(self) -> str:
        template_name = os.path.splitext(os.path.basename(self.task_template_path))[0]
        return template_name
    
    def load_task_template(self) -> str:
        if os.path.exists(self.task_template_path):
            with open(self.task_template_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            return """# Role
You are a master of table preprocessing task construction, skilled at constructing tasks that meet requirements based on task descriptions and specifications, ensuring the table that meets the task requirements is unique
# Task Description
Based on the uploaded file, construct a table processing task, providing task description, Python code for constructing the gt table, and Python code for constructing the input table
# Task Construction Requirements
1. Provide a task description that includes the task input, output (requiring output of a table named gt.csv), and processing rules
2. The task must not contain any random operations, and must strictly specify the format or order of rows and columns in the final output, ensuring the table that meets the task requirements is unique
3. The number of processing rules must not exceed 3
# Input Table Construction Requirements
Provide Python code to split the original table into input tables, where the code input is the given input table and the output is the input table
# GT Table Construction Requirements
Provide Python code that fully complies with the task description to generate the gt table, where the code input is the input table and the output is gt.csv
# Task Output
The task description outputs content in JSON format:
```json
{
    "task":{
        "target_zh": "",
        "target_en": ""
    }
}
```
Code output in Python"""
    
    def generate_task(self, dataset_description: str, original_dataset_path: str = None, max_retries: int = 5) -> Tuple[Optional[Dict], Optional[str], Optional[str], Optional[Dict[str, pd.DataFrame]]]:
        template = self.load_task_template()
        original_path_info = ""
        if original_dataset_path:
            abs_path = os.path.abspath(original_dataset_path)
            original_path_info = f"\nOriginal dataset file path: {abs_path}\nWhen generating input table construction code, please use pd.read_csv('{abs_path}') to read the original dataset, or directly use the variable df (auto-loaded)."
        
        prompt = f"""{template}

# Dataset Description
{dataset_description}
{original_path_info}

Note: All generated CSV files should be saved to the current working directory"""
        
        for attempt in range(max_retries):
            try:
                response, token_count = self.generator_model.call(prompt)
                
                task_desc, input_code, gt_code = self._parse_response(response)
                
                if not task_desc or not gt_code:
                    if attempt < max_retries - 1:
                        prompt += "\n\nPlease ensure the output format is correct, including the task description JSON and complete Python code (at least gt table construction code is required)."
                    continue
                
                input_dfs = {}
                if input_code and input_code.strip():
                    success, error, input_dfs = self.code_executor.execute_code(
                        input_code, 
                        self.current_tmp_input_dir,
                        input_files=None,
                        original_dataset=original_dataset_path
                    )
                    
                    if not success or not input_dfs:
                        if attempt < max_retries - 1:
                            prompt += f"\n\nError feedback: {error}\nPlease regenerate the input table construction code."
                        continue
                else:
                    if original_dataset_path and os.path.exists(original_dataset_path):
                        try:
                            df_original = pd.read_csv(original_dataset_path)
                            original_filename = os.path.basename(original_dataset_path)
                            if 'input' not in original_filename.lower():
                                input_filename = 'input.csv'
                            else:
                                input_filename = original_filename
                            input_dfs[input_filename] = df_original
                        except Exception as e:
                            if attempt < max_retries - 1:
                                prompt += f"\n\nError feedback: Unable to read original dataset {original_dataset_path}. Please regenerate."
                            continue
                    else:
                        if attempt < max_retries - 1:
                            prompt += f"\n\nError feedback: Original dataset path is invalid. Please regenerate."
                        continue
                
                input_file_paths = [os.path.join(self.current_tmp_input_dir, fname) for fname in input_dfs.keys()]
                success, error, gt_dfs = self.code_executor.execute_code(
                    gt_code,
                    self.current_tmp_input_dir,
                    input_files=input_file_paths
                )
                
                if not success or not gt_dfs:
                    if attempt < max_retries - 1:
                        prompt += f"\n\nError feedback: {error}\nPlease regenerate the gt table construction code."
                    continue
                
                if 'gt.csv' not in gt_dfs and 'gt' not in [f.replace('.csv', '') for f in gt_dfs.keys()]:
                    if attempt < max_retries - 1:
                        prompt += "\n\nError feedback: Must generate a file named gt.csv. Please regenerate the code."
                    continue
                
                return task_desc, input_code, gt_code, {**input_dfs, **gt_dfs}
                
            except Exception as e:
                if attempt < max_retries - 1:
                    prompt += f"\n\nError feedback: {str(e)}\nPlease regenerate."
        
        return None, None, None, None
    
    def _parse_response(self, response: str) -> Tuple[Optional[Dict], Optional[str], Optional[str]]:
        task_desc = None
        input_code = None
        gt_code = None
        
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                task_desc = json.loads(json_str)
        except:
            try:
                json_start = response.find('```json')
                if json_start >= 0:
                    json_start = response.find('\n', json_start) + 1
                    json_end = response.find('```', json_start)
                    if json_end > json_start:
                        json_str = response[json_start:json_end].strip()
                        task_desc = json.loads(json_str)
            except:
                pass
        
        code_blocks = []
        in_code_block = False
        code_language = None
        current_code = []
        
        for line in response.split('\n'):
            if line.strip().startswith('```'):
                if in_code_block:
                    if current_code:
                        code_blocks.append('\n'.join(current_code))
                    current_code = []
                    in_code_block = False
                    code_language = None
                else:
                    in_code_block = True
                    if 'python' in line.lower():
                        code_language = 'python'
                    elif 'json' not in line.lower():
                        code_language = 'other'
            elif in_code_block and (code_language == 'python' or code_language == 'other'):
                current_code.append(line)
        
        if current_code:
            code_blocks.append('\n'.join(current_code))
        
        if len(code_blocks) >= 2:
            input_code = code_blocks[0]
            gt_code = code_blocks[1]
        elif len(code_blocks) == 1:
            code = code_blocks[0]
            separators = ['# input', '# construct input', '# generate input', 'def generate_input',
                         '# gt', '# construct gt', '# generate gt', 'def generate_gt']
            found_sep = None
            for sep in separators:
                if sep.lower() in code.lower():
                    found_sep = sep
                    break
            
            if found_sep:
                parts = code.split(found_sep, 1)
                if len(parts) >= 2:
                    input_code = parts[0].strip()
                    gt_code = found_sep + parts[1].strip()
                else:
                    input_code = code
                    gt_code = code
            else:
                code_lower = code.lower()
                if 'input1.csv' in code_lower or 'input2.csv' in code_lower:
                    input_code = code
                    gt_code = None
                elif 'gt.csv' in code_lower:
                    gt_code = code
                    input_code = None
                else:
                    input_code = code
                    gt_code = code
        
        return task_desc, input_code, gt_code
    
    def validate_with_models(self, task_desc: Dict, input_dfs: Dict[str, pd.DataFrame]) -> Tuple[List[str], List[Dict[str, pd.DataFrame]], List[Dict]]:
        task_prompt = f"""Task description (Chinese):
{task_desc.get('task', {}).get('target_zh', '')}

Task description (English):
{task_desc.get('task', {}).get('target_en', '')}

Input table information:
"""
        for fname, df in input_dfs.items():
            if 'gt' not in fname.lower():
                task_prompt += f"\nTable {fname}:\nColumns: {list(df.columns)}\nRows: {len(df)}\nFirst 3 rows:\n{df.head(3).to_string()}\n"
        
        task_prompt += "\nBased on the task description and input tables, generate Python code to complete the task. The code should read the input tables, process the data, and output a gt.csv file."
        
        codes = []
        results = []
        exec_infos = []
        
        for i, model in enumerate(self.validator_models):
            exec_info = {
                "model_index": i + 1,
                "model_name": model.model_name,
                "has_code": False,
                "code_executable": False,
                "execution_success": False,
                "execution_error": "",
                "result_generated": False,
                "matches_gt": False
            }
            
            try:
                response, token_count = model.call(task_prompt)
                
                code = self._extract_code(response)
                codes.append(code)
                exec_info["has_code"] = bool(code and code.strip())
                
                if not code or not code.strip():
                    results.append({})
                    exec_infos.append(exec_info)
                    continue
                
                input_file_paths = []
                tmp_input_dir = self.current_tmp_input_dir or self.tmp_dir
                for fname in input_dfs.keys():
                    if 'gt' not in fname.lower():
                        filepath = os.path.join(tmp_input_dir, fname)
                        if os.path.exists(filepath):
                            input_file_paths.append(filepath)
                        else:
                            os.makedirs(tmp_input_dir, exist_ok=True)
                            input_dfs[fname].to_csv(filepath, index=False, encoding='utf-8')
                            input_file_paths.append(filepath)
                
                success, error, output_dfs = self.code_executor.execute_code(
                    code,
                    self.current_tmp_input_dir,
                    input_files=input_file_paths
                )
                
                exec_info["code_executable"] = True
                exec_info["execution_success"] = success
                exec_info["execution_error"] = error if not success else ""
                exec_info["result_generated"] = bool(output_dfs)
                
                if success and output_dfs:
                    results.append(output_dfs)
                else:
                    results.append({})
                    
            except Exception as e:
                codes.append("")
                results.append({})
                exec_info["execution_error"] = str(e)
            
            exec_infos.append(exec_info)
        
        return codes, results, exec_infos
    
    def _extract_code(self, response: str) -> str:
        code_blocks = []
        in_code_block = False
        current_code = []
        
        for line in response.split('\n'):
            if '```python' in line or '```' in line:
                if in_code_block:
                    code_blocks.append('\n'.join(current_code))
                    current_code = []
                    in_code_block = False
                else:
                    in_code_block = True
            elif in_code_block:
                current_code.append(line)
        
        if current_code:
            code_blocks.append('\n'.join(current_code))
        
        if code_blocks:
            return code_blocks[0]
        return response
    
    def compare_results(self, gt_df: pd.DataFrame, results: List[Dict[str, pd.DataFrame]]) -> Tuple[bool, List[bool]]:
        matches = []
        
        try:
            gt_cols_sorted = sorted(gt_df.columns)
            gt_normalized = gt_df[gt_cols_sorted].copy()
            if len(gt_normalized) > 0:
                gt_normalized = gt_normalized.sort_values(by=gt_cols_sorted).reset_index(drop=True)
        except:
            gt_normalized = gt_df.copy()
        
        for i, result in enumerate(results):
            gt_result = None
            for fname, df in result.items():
                if 'gt' in fname.lower() or fname.endswith('gt.csv'):
                    gt_result = df
                    break
            
            if gt_result is None or gt_result.empty:
                matches.append(False)
                continue
            
            try:
                if set(gt_normalized.columns) != set(gt_result.columns):
                    matches.append(False)
                    continue
                
                result_normalized = gt_result[sorted(gt_result.columns)].copy()
                
                if len(result_normalized) > 0:
                    result_normalized = result_normalized.sort_values(by=sorted(result_normalized.columns)).reset_index(drop=True)
                
                if len(gt_normalized) != len(result_normalized):
                    matches.append(False)
                    continue
                
                if gt_normalized.equals(result_normalized):
                    matches.append(True)
                else:
                    numeric_cols = gt_normalized.select_dtypes(include=['float64', 'float32']).columns
                    if len(numeric_cols) > 0:
                        diff = (gt_normalized[numeric_cols] - result_normalized[numeric_cols]).abs()
                        numeric_match = (diff < 1e-6).all().all()
                        
                        non_numeric_cols = [c for c in gt_normalized.columns if c not in numeric_cols]
                        if non_numeric_cols:
                            non_numeric_match = gt_normalized[non_numeric_cols].equals(result_normalized[non_numeric_cols])
                        else:
                            non_numeric_match = True
                        
                        if numeric_match and non_numeric_match:
                            matches.append(True)
                        else:
                            matches.append(False)
                    else:
                        matches.append(False)
            except Exception as e:
                matches.append(False)
        
        total_matches = sum(matches) + 1
        return total_matches > 3, matches
    
    def save_results(self, 
                    dataset_name: str,
                    task_desc: Dict,
                    input_code: str,
                    gt_code: str,
                    input_dfs: Dict[str, pd.DataFrame],
                    gt_df: pd.DataFrame,
                    validator_codes: List[str],
                    validator_results: List[Dict[str, pd.DataFrame]],
                    exec_infos: List[Dict]):
        template_name = self.get_task_template_name()
        
        if '-' in template_name:
            template_category = template_name.split('-')[0]
        else:
            template_category = template_name
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_folder = os.path.join(self.task_dir, template_category, template_name, f"{dataset_name}_{template_name}_{timestamp}")
        os.makedirs(task_folder, exist_ok=True)
        
        src_folder = os.path.join(task_folder, "src")
        os.makedirs(src_folder, exist_ok=True)
        
        with open(os.path.join(task_folder, "task.json"), 'w', encoding='utf-8') as f:
            json.dump(task_desc, f, ensure_ascii=False, indent=2)
        
        if input_code and input_code.strip():
            with open(os.path.join(task_folder, "gen.py"), 'w', encoding='utf-8') as f:
                f.write(input_code)
        
        with open(os.path.join(task_folder, "ans.py"), 'w', encoding='utf-8') as f:
            f.write(gt_code)
        
        input_pattern = re.compile(r'^input(\d+)?\.csv$', re.IGNORECASE)
    
        filtered_input_dfs = {
            fname: df for fname, df in input_dfs.items()
            if 'gt' not in fname.lower() and input_pattern.match(fname)
        }

        for fname, df in filtered_input_dfs.items():
            save_path = os.path.join(src_folder, fname)
            df.to_csv(save_path, index=False, encoding='utf-8')
        
        gt_df.to_csv(os.path.join(src_folder, "gt.csv"), index=False, encoding='utf-8')
        
        exec_json = {
            "validation_results": exec_infos
        }
        with open(os.path.join(task_folder, "exec.json"), 'w', encoding='utf-8') as f:
            json.dump(exec_json, f, ensure_ascii=False, indent=2)
        
        return task_folder
    
    def run(self, csv_path: str, dataset_name: Optional[str] = None):
        if dataset_name is None:
            dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
        
        tmp_subfolder = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.current_tmp_input_dir = os.path.join(self.tmp_dir, tmp_subfolder)
        os.makedirs(self.current_tmp_input_dir, exist_ok=True)
        
        try:
            analyzer = DatasetAnalyzer()
            description = analyzer.generate_description(csv_path)
            
            task_desc, input_code, gt_code, all_dfs = self.generate_task(description, csv_path)
            
            if task_desc is None:
                return
            
            input_dfs = {fname: df for fname, df in all_dfs.items() if 'gt' not in fname.lower()}
            gt_df = None
            for fname, df in all_dfs.items():
                if 'gt' in fname.lower():
                    gt_df = df
                    break
            
            if gt_df is None:
                return
            
            for fname, df in input_dfs.items():
                filepath = os.path.join(self.current_tmp_input_dir, fname)
                df.to_csv(filepath, index=False, encoding='utf-8')
            
            validator_codes, validator_results, exec_infos = self.validate_with_models(task_desc, input_dfs)
            
            passed, match_flags = self.compare_results(gt_df, validator_results)
            
            for i, match in enumerate(match_flags):
                if i < len(exec_infos):
                    exec_infos[i]["matches_gt"] = match
            
            if passed:
                task_folder = self.save_results(
                    dataset_name,
                    task_desc,
                    input_code,
                    gt_code,
                    input_dfs,
                    gt_df,
                    validator_codes,
                    validator_results,
                    exec_infos
                )
                if task_folder:
                    pass
                else:
                    pass
            else:
                pass
        except Exception as e:
            import traceback
            traceback.print_exc()


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


def list_available_datasets():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    
    if not os.path.exists(data_dir):
        return
    
    
    categories = ["Standard", "Large", "Modest", "Wide"]
    for category in categories:
        category_dir = os.path.join(data_dir, category)
        if os.path.exists(category_dir):
            csv_files = [f for f in os.listdir(category_dir) if f.endswith('.csv')]
            if csv_files:
                for csv_file in sorted(csv_files):
                    pass
    


def get_datasets_from_config(config=None):
    if config is None:
        config = load_config()
    
    if config is None:
        return []
    
    datasets = []
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    
    category = None
    if hasattr(config, 'dataset_category') and config.dataset_category:
        category = config.dataset_category
    elif hasattr(config, 'DATASET_CATEGORY') and config.DATASET_CATEGORY:
        category = config.DATASET_CATEGORY
    
    if category:
        if category == "all":
            for category_name in ["Standard", "Large", "Modest", "Wide"]:
                category_dir = os.path.join(data_dir, category_name)
                if os.path.exists(category_dir):
                    for filename in os.listdir(category_dir):
                        if filename.endswith('.csv'):
                            datasets.append(os.path.join(category_dir, filename))
        else:
            category_dir = os.path.join(data_dir, category)
            if os.path.exists(category_dir):
                for filename in os.listdir(category_dir):
                    if filename.endswith('.csv'):
                        datasets.append(os.path.join(category_dir, filename))
        return datasets
    
    paths = None
    if hasattr(config, 'dataset_paths') and config.dataset_paths:
        paths = config.dataset_paths
    elif hasattr(config, 'DATASET_PATHS') and config.DATASET_PATHS:
        paths = config.DATASET_PATHS
    
    if paths:
        for path in paths:
            if not os.path.isabs(path):
                path = os.path.join(base_dir, path)
            if os.path.exists(path):
                datasets.append(path)
        return datasets
    
    path = None
    if hasattr(config, 'dataset_path') and config.dataset_path:
        path = config.dataset_path
    elif hasattr(config, 'DATASET_PATH') and config.DATASET_PATH:
        path = config.DATASET_PATH
    
    if path:
        if not os.path.isabs(path):
            path = os.path.join(base_dir, path)
        if os.path.exists(path):
            datasets.append(path)
        return datasets
    
    return []


def get_task_templates_from_config(config=None):
    if config is None:
        config = load_config()
    
    if config is None:
        return []
    
    templates = []
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scripts_dir = os.path.join(base_dir, "scripts", "single_task")
    
    category = None
    if hasattr(config, 'task_template_category') and config.task_template_category:
        category = config.task_template_category
    elif hasattr(config, 'TASK_TEMPLATE_CATEGORY') and config.TASK_TEMPLATE_CATEGORY:
        category = config.TASK_TEMPLATE_CATEGORY
    
    if category:
        if category == "all":
            for category_name in ["DI", "EC", "EM", "LT", "RP", "RT", "SA", "SM"]:
                category_dir = os.path.join(scripts_dir, category_name)
                if os.path.exists(category_dir):
                    for root, dirs, files in os.walk(category_dir):
                        for filename in files:
                            if filename.endswith('.txt'):
                                templates.append(os.path.join(root, filename))
        else:
            category_dir = os.path.join(scripts_dir, category)
            if os.path.exists(category_dir):
                for root, dirs, files in os.walk(category_dir):
                    for filename in files:
                        if filename.endswith('.txt'):
                            templates.append(os.path.join(root, filename))
        return sorted(templates)
    
    paths = None
    if hasattr(config, 'task_template_paths') and config.task_template_paths:
        paths = config.task_template_paths
    elif hasattr(config, 'TASK_TEMPLATE_PATHS') and config.TASK_TEMPLATE_PATHS:
        paths = config.TASK_TEMPLATE_PATHS
    
    if paths:
        for path in paths:
            if not os.path.isabs(path):
                path = os.path.join(base_dir, path)
            if os.path.exists(path):
                templates.append(path)
        return templates
    
    path = None
    if hasattr(config, 'task_template_path') and config.task_template_path:
        path = config.task_template_path
    elif hasattr(config, 'TASK_TEMPLATE_PATH') and config.TASK_TEMPLATE_PATH:
        path = config.TASK_TEMPLATE_PATH
    
    if path:
        if not os.path.isabs(path):
            path = os.path.join(base_dir, path)
        if os.path.exists(path):
            templates.append(path)
        return templates
    
    return []
