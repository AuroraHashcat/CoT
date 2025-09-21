import os
import argparse
import json
import re
import time
import random
from datetime import datetime
from typing import List, Dict, Any, Tuple
from urllib.parse import quote_plus

from tqdm import tqdm

from common.config_utils import load_dataset_config
from data_processing.data_loader import load_data
from causal_cot.llm_handler import create_llm_handler

# 数据集类型映射
DATASET_TYPE_MAP = {
    "math": "fill_in_blank",
    "causalnet": "true_or_false",
    "cladder": "multiple_choice",
    "commonsenseqa": "multiple_choice",
    "corr2cause": "true_or_false",
    "gpqa": "multiple_choice",
    "aqua": "multiple_choice"
}

def get_dataset_type(dataset_name):
    """根据数据集名称获取题型"""
    for key in DATASET_TYPE_MAP:
        if key in dataset_name.lower():
            return DATASET_TYPE_MAP[key]
    return "fill_in_blank"  # 默认为填空题

def get_cot_prompt(item, dataset_type=None):
    q = item['question']
    if dataset_type == "multiple_choice":
        if 'choices' in item and item['choices']:
            q += '\nChoices:\n' + '\n'.join(item['choices'])
        prompt = (
            "You are a meticulous logical reasoner.\n"
            "Solve the following MULTIPLE CHOICE question by creating a step-by-step Chain of Thought. Each step must be causally justified, not just correlated or associated.\n\n"
            f"{q}\n\n"
            "IMPORTANT: This is a multiple choice question. Your final answer MUST be exactly one of the choice letters (A, B, C, or D) or numbers (0, 1, 2, or 3), nothing else.\n\n"
            "Output Format:\n"
            "Step 1: [First causal deduction or analysis]\n"
            "Step 2: [Second deduction, building upon Step 1]\n"
            "...\n"
            "Step N: [Final deduction that directly leads to the answer]\n"
            "===FINAL_ANSWER_START===\n"
            "Conclusion: [Single letter/number representing your choice: A, B, C, D or 0, 1, 2, 3]\n"
            "===FINAL_ANSWER_END===\n"
        )
    elif dataset_type == "fill_in_blank":
        prompt = (
            "You are a meticulous logical reasoner.\n"
            "Solve the following question by creating a step-by-step Chain of Thought. Each step must be causally justified, not just correlated or associated.\n\n"
            f"{q}\n\n"
            "Output Format:\n"
            "Step 1: [First causal deduction or analysis]\n"
            "Step 2: [Second deduction, building upon Step 1]\n"
            "...\n"
            "Step N: [Final deduction that directly leads to the answer]\n"
            "===FINAL_ANSWER_START===\n"
            "Conclusion: [Your final answer, only a single number or expression, nothing else. based strictly on the above causal reasoning]\n"
            "===FINAL_ANSWER_END===\n"
        )
    elif dataset_type == "true_or_false":
        prompt = (
            "You are a meticulous logical reasoner.\n"
            "Solve the following TRUE or FALSE question by creating a step-by-step Chain of Thought. Each step must be causally justified, not just correlated or associated.\n\n"
            f"{q}\n\n"
            "IMPORTANT: This is a TRUE or FALSE question. Your final answer MUST be exactly 'true' or 'false', nothing else.\n\n"
            "Output Format:\n"
            "Step 1: [First causal deduction or analysis]\n"
            "Step 2: [Second deduction, building upon Step 1]\n"
            "...\n"
            "Step N: [Final deduction that directly leads to the answer]\n"
            "===FINAL_ANSWER_START===\n"
            "Conclusion: [true or false]\n"
            "===FINAL_ANSWER_END===\n"
        )
    return prompt


def extract_cot_and_answer(text: str) -> Tuple[List[str], Any]:
    import re
    print("=== LLM OUTPUT ===")
    print(text)
    steps = re.findall(r"Step \d+: (.*)", text)
    # 只提取 Conclusion: 后的内容
    conclusion = None
    match = re.search(r"Conclusion:\s*(.*?)(?:\n|===FINAL_ANSWER_END===|$)", text)
    if match:
        conclusion = match.group(1).strip()
        # 去除多余符号
        conclusion = re.sub(r'[\*#]+', '', conclusion).strip()
    print(f"Extracted steps: {steps}")
    print(f"Extracted conclusion: {conclusion}")
    return steps, conclusion


def get_answer_key(item: Dict[str, Any]) -> Any:
    return item.get('answerKey') or item.get('Answer') or item.get('answer')


class BaiduSearcher:
    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        self.session = None
        # 用户代理池，轮换使用
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15'
        ]
        self._init_session()

    def _init_session(self):
        """初始化会话，保持Cookie"""
        try:
            import requests
            self.session = requests.Session()
            # 先访问百度首页，获取Cookie
            self.session.get('https://www.baidu.com', timeout=10, headers={
                'User-Agent': random.choice(self.user_agents)
            })
        except Exception as e:
            print(f"[WARN] Failed to initialize Baidu session: {e}")

    def search(self, query: str) -> List[Dict[str, str]]:
        """使用百度搜索（改进的反爬虫绕过版本）"""
        results: List[Dict[str, str]] = []
        try:
            import requests
            from bs4 import BeautifulSoup  # type: ignore
        except Exception as e:
            print(f"[ERROR] Baidu search requires requests and bs4: {e}")
            return results
        
        # 添加随机延时，模拟人类行为
        time.sleep(random.uniform(1.0, 3.0))
        
        try:
            # 百度搜索URL
            url = f"https://www.baidu.com/s?wd={quote_plus(query)}&pn=0&rn={self.max_results}"
            headers = {
                'User-Agent': random.choice(self.user_agents),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'same-origin',
                'Cache-Control': 'max-age=0',
            }
            
            # 使用会话或直接请求
            client = self.session if self.session else requests
            resp = client.get(url, timeout=15, headers=headers)
            resp.raise_for_status()
            resp.encoding = 'utf-8'
            
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            # 百度搜索结果的多种选择器（适应页面变化）
            result_selectors = [
                'div.result.c-container',  # 主要选择器
                'div[class*="result"]',     # 备选选择器
                'div.c-container',          # 简化选择器
            ]
            
            result_items = []
            for selector in result_selectors:
                result_items = soup.select(selector)
                if result_items:
                    break
            
            if not result_items:
                print("[WARN] No Baidu search results found with any selector")
                return results

            for item in result_items[:self.max_results]:
                try:
                    # 提取标题（多种选择器）
                    title_selectors = ['h3.t a', 'h3 a', 'h3.c-title a', 'a[data-click]']
                    title_elem = None
                    for sel in title_selectors:
                        title_elem = item.select_one(sel)
                        if title_elem:
                            break
                    
                    title = title_elem.get_text(strip=True) if title_elem else ''
                    
                    # 提取URL
                    url_elem = title_elem
                    url = url_elem.get('href', '') if url_elem else ''
                    
                    # 提取摘要（多种选择器）
                    snippet_selectors = [
                        'span.content-right_8Zs40',
                        'div.c-abstract', 
                        'div.c-span-last',
                        'span[class*="content"]',
                        'div[class*="abstract"]'
                    ]
                    
                    snippet_elem = None
                    for sel in snippet_selectors:
                        snippet_elem = item.select_one(sel)
                        if snippet_elem:
                            break
                    
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ''
                    
                    if title and url:
                        results.append({
                            'title': title,
                            'url': url,
                            'snippet': snippet
                        })
                except Exception as e:
                    print(f"[WARN] Error parsing Baidu result: {e}")
                    continue
                    
        except Exception as e:
            print(f"[ERROR] Baidu search failed: {e}")
            # 如果会话失败，尝试重新初始化
            if self.session:
                self._init_session()
        
        return results


class DuckDuckGoSearcher:
    def __init__(self, max_results: int = 5, region: str = 'wt-wt', safesearch: str = 'moderate', timelimit: str = None, allow_html_fallback: bool = False):
        self.max_results = max_results
        self.region = region
        self.safesearch = safesearch
        self.timelimit = timelimit
        self.allow_html_fallback = allow_html_fallback
        self._init_backend()

    def _init_backend(self) -> None:
        self._use_ddgs = False
        try:
            from ddgs import DDGS  # 新版 DuckDuckGo 搜索包
            self.DDGS = DDGS
            self._use_ddgs = True
        except ImportError:
            print("[警告] 未安装 ddgs 包，请先 pip install ddgs")
            self.DDGS = None
        except Exception as e:
            self.DDGS = None
            if not self.allow_html_fallback:
                raise RuntimeError("ddgs is not installed. Please `pip install ddgs`. Or set allow_html_fallback=True to use HTML scraping fallback.")

    def search(self, query: str) -> List[Dict[str, str]]:
        if self._use_ddgs and self.DDGS is not None:
            return self._search_ddgs(query)
        return self._search_html_fallback(query) if self.allow_html_fallback else []

    def _search_ddgs(self, query: str) -> List[Dict[str, str]]:
        results: List[Dict[str, str]] = []
        try:
            with self.DDGS() as ddgs:
                for r in ddgs.text(query, max_results=self.max_results):
                    title = r.get('title', '')
                    url = r.get('href', '')
                    snippet = r.get('body', '')
                    if url:
                        results.append({'title': title, 'url': url, 'snippet': snippet})
        except Exception as e:
            print(f"[WARN] DDGS search failed: {e}")
        return results

    def _search_html_fallback(self, query: str) -> List[Dict[str, str]]:
        results: List[Dict[str, str]] = []
        try:
            import requests
            from bs4 import BeautifulSoup  # type: ignore
        except Exception as e:
            print(f"[ERROR] Fallback search requires requests and bs4: {e}")
            return results
        try:
            url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
            resp = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116 Safari/537.36'
            })
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            links = soup.select('a.result__a')
            snippets = soup.select('a.result__snippet')
            for i, a in enumerate(links[:self.max_results]):
                href = a.get('href') or ''
                title = a.get_text(strip=True)
                snippet = snippets[i].get_text(strip=True) if i < len(snippets) else ''
                if href:
                    results.append({'title': title, 'url': href, 'snippet': snippet})
        except Exception as e:
            print(f"[ERROR] HTML scraping failed: {e}")
        return results


class MultiSearcher:
    """多引擎搜索器，支持自动回退"""
    def __init__(self, max_results: int = 5, **kwargs):
        self.max_results = max_results
        self.searchers = []
        
        # 尝试初始化DuckDuckGo搜索器
        try:
            ddg_searcher = DuckDuckGoSearcher(max_results=max_results, **kwargs)
            self.searchers.append(('DuckDuckGo', ddg_searcher))
            print("[INFO] DuckDuckGo searcher initialized")
        except Exception as e:
            print(f"[WARN] Failed to initialize DuckDuckGo searcher: {e}")
        
        # 初始化百度搜索器作为备用
        try:
            baidu_searcher = BaiduSearcher(max_results=max_results)
            self.searchers.append(('Baidu', baidu_searcher))
            print("[INFO] Baidu searcher initialized as backup")
        except Exception as e:
            print(f"[WARN] Failed to initialize Baidu searcher: {e}")
        
        if not self.searchers:
            raise RuntimeError("No search engines available!")

    def search(self, query: str) -> List[Dict[str, str]]:
        """依次尝试每个搜索引擎"""
        for engine_name, searcher in self.searchers:
            try:
                print(f"[INFO] Trying search with {engine_name}...")
                results = searcher.search(query)
                if results:
                    print(f"[INFO] Found {len(results)} results using {engine_name}")
                    return results
                else:
                    print(f"[WARN] No results from {engine_name}")
            except Exception as e:
                print(f"[WARN] {engine_name} search failed: {e}")
                continue
        
        print("[WARN] All search engines failed")
        return []


def build_query_generation_prompt(question: str, step: str) -> str:
    return (
        "You are generating targeted web search queries to fact-check and causally validate a reasoning step.\n"
        "Given the question and the specific step, propose 2-4 concise web search queries that will surface authoritative evidence relevant to the causal claims in the step.\n"
        "Return ONLY a JSON list of strings.\n\n"
        f"Question:\n{question}\n\n"
        f"Reasoning Step:\n{step}\n\n"
        "JSON list of queries:"
    )


def parse_queries_from_llm(text: str) -> List[str]:
    try:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return []
        arr = json.loads(match.group(0))
        return [str(x).strip() for x in arr if isinstance(x, str)]
    except Exception:
        return []


def build_step_validation_prompt(question: str, step: str, web_evidence: List[Dict[str, str]]) -> str:
    evidence_str_parts = []
    for i, e in enumerate(web_evidence, 1):
        evidence_str_parts.append(
            f"[{i}] Title: {e.get('title','')}\nURL: {e.get('url','')}\nSnippet: {e.get('snippet','')}\n"
        )
    evidence_block = "\n".join(evidence_str_parts) if evidence_str_parts else "No relevant evidence found."
    return (
        "You are an expert in causal reasoning. Assess whether the following reasoning step is causally and logically valid, using the provided web evidence as support.\n\n"
        f"Question:\n{question}\n\n"
        f"Reasoning Step:\n{step}\n\n"
        f"Web Evidence:\n{evidence_block}\n\n"
        "Your task: Output one of the following as the FIRST line: ACCEPT / REJECT_CAUSAL / REJECT_LOGICAL.\n"
        "Then provide a brief explanation referencing the evidence indices when possible."
    )


def is_multiple_choice_question(item: Dict[str, Any]) -> bool:
    """检测是否为选择题"""
    question_text = item['question']
    return ('Choices:' in question_text) or ('choices' in item and item['choices'])


def build_reflection_prompt(question: str, validated_steps: List[str], failed_step: str, failure_reason: str, step_index: int, dataset_type: str = "multiple_choice") -> str:
    validated_text = "\n".join(validated_steps) if validated_steps else "(none)"
    
    mc_instruction = ""
    if dataset_type == "multiple_choice":
        mc_instruction = "\nIMPORTANT: This is a multiple choice question. Your final conclusion MUST be exactly one choice letter (A, B, C, D) or number (0, 1, 2, 3).\n"
    elif dataset_type == "true_or_false":
        mc_instruction = "\nIMPORTANT: This is a TRUE or FALSE question. Your final conclusion MUST be exactly 1 for 'TRUE' or 0 for 'FALSE'.\n"
    elif dataset_type == "fill_in_blank":
        mc_instruction = "\nIMPORTANT: This is a fill-in-the-blank question. Your final conclusion should be a concise answer based strictly on the reasoning steps.\n"
    return (
        "You are a self-correcting reasoning agent. The following reasoning step failed causal/logical validation.\n\n"
        f"Question:\n{question}\n\n"
        f"Validated Steps So Far:\n{validated_text}\n\n"
        f"Failed Step:\n{failed_step}\n\n"
        f"Failure Reason:\n{failure_reason}\n\n"
        f"{mc_instruction}"
        "Your task: Regenerate the reasoning chain from the failed step onward, correcting the error. Number each step as before and end with 'Conclusion:'.\n"
        f"Step {step_index}: ...\nStep {step_index+1}: ...\nConclusion: ...\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Causal-CoT with Web Search Validation (Multi-Engine)")
    parser.add_argument('--model_config', type=str, required=True, help='Path to model config JSON.')
    parser.add_argument('--dataset_config', type=str, required=True, help='Path to dataset config JSON.')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--max_results', type=int, default=5, help='Max web results per query')
    parser.add_argument('--max_queries_per_step', type=int, default=3, help='Max search queries per step')
    parser.add_argument('--max_rounds', type=int, default=1, help='Max reflection rounds when a step fails')
    # Search engine selection
    parser.add_argument('--search_engine', type=str, default='auto', choices=['auto', 'ddg', 'baidu'], 
                        help='Search engine to use: auto (try DDG first, fallback to Baidu), ddg (DuckDuckGo only), baidu (Baidu only)')
    # DDGS params
    parser.add_argument('--ddgs_region', type=str, default='wt-wt', help='DDGS region, e.g., wt-wt, us-en, uk-en')
    parser.add_argument('--ddgs_safesearch', type=str, default='moderate', help='DDGS safesearch: off, moderate, strict')
    parser.add_argument('--ddgs_timelimit', type=str, default=None, help='DDGS timelimit: d, w, m, y, or None')
    parser.add_argument('--allow_html_fallback', action='store_true', help='Allow fallback to HTML scraping if duckduckgo_search is unavailable')
    args = parser.parse_args()

    print(f"[INFO] Loading model config from: {args.model_config}")
    print(f"[INFO] Loading dataset config from: {args.dataset_config}")
    try:
        dataset_config = load_dataset_config(args.dataset_config)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset config: {e}")
        return

    print("[INFO] Initializing model handler...")
    try:
        # 加载并转换模型配置
        with open(args.model_config, 'r', encoding='utf-8') as f:
            old_config = json.load(f)
        
        # 转换配置格式以适应新的llm_handler
        model_info = old_config.get("model_info", {})
        api_key_info = old_config.get("api_key_info", {})
        params = old_config.get("params", {})
        
        # 从环境变量获取API密钥
        api_key_env = api_key_info.get("api_key_env")
        api_key = os.getenv(api_key_env) if api_key_env else None
        
        model_config = {
            "type": "api",
            "provider": model_info.get("provider", "openai"),
            "model": model_info.get("name", ""),
            "api_key": api_key,
            "base_url": api_key_info.get("api_url"),
            "temperature": params.get("temperature", 0.7),
            "max_tokens": params.get("max_output_tokens", 2000),
            "max_retries": 3,
            "retry_delay": 1.0
        }
        
        llm = create_llm_handler(model_config)
    except Exception as e:
        print(f"[ERROR] Failed to initialize model handler: {e}")
        return

    print(f"[INFO] Loading data...")
    try:
        data = load_data(dataset_config)
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return

    model_name = os.path.basename(args.model_config).replace('.json', '')
    dataset_name = os.path.basename(args.dataset_config).replace('.json', '')
    
    # 添加时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = os.path.join("results", dataset_name, "web_search")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_name}_{timestamp}.json")
    if args.output:
        output_path = args.output

    # Initialize searcher: 只用 DuckDuckGo
    print(f"[INFO] Initializing DuckDuckGo search engine...")
    searcher = DuckDuckGoSearcher(
        max_results=args.max_results,
        region=args.ddgs_region,
        safesearch=args.ddgs_safesearch,
        timelimit=args.ddgs_timelimit,
        allow_html_fallback=args.allow_html_fallback,
    )

    results: List[Dict[str, Any]] = []
    correct = 0
    total_elapsed = 0.0
    total_tokens = 0

    for idx, item in enumerate(tqdm(data, desc=f"Processing {dataset_name}")):
        start_time = time.time()
        sample_id = item.get('id', '')
        question_text = item['question']
        dataset_type = get_dataset_type(dataset_name)
        tokens_used = None
        gold = None
        conclusion = None
        final_cot = []
        is_correct = False
        step_validation_records = []
        reflection_used = 0
        try:
            prompt = get_cot_prompt(item,dataset_type=dataset_type)
            cot_output = llm.query(prompt)
            # 参考baseline统计token方法
            if isinstance(cot_output, dict) and "tokens_used" in cot_output:
                tokens_used = cot_output["tokens_used"]
                cot_output_text = cot_output.get("text", str(cot_output))
            else:
                tokens_used = None
                cot_output_text = cot_output if isinstance(cot_output, str) else str(cot_output)
            steps, pred = extract_cot_and_answer(cot_output_text)
        except Exception as e:
            print(f"[ERROR] LLM failed for sample id {sample_id}: {e}")
            steps, pred = [], None
            cot_output_text = str(e)
            tokens_used = None
        # Validate each step using web search
        validated_steps: List[str] = []
        final_cot = []
        reflection_used = 0
        conclusion = pred
        for s_idx, step in enumerate(steps, start=1):
            try:
                q_prompt = build_query_generation_prompt(question_text, step)
                q_resp = llm.query(q_prompt)
                queries = parse_queries_from_llm(q_resp)
                if not queries:
                    queries = [f"{question_text} {step}"]
                queries = queries[: args.max_queries_per_step]
            except Exception:
                queries = [f"{question_text} {step}"]
            evidence: List[Dict[str, str]] = []
            for q in queries:
                ev = searcher.search(q)
                for e in ev:
                    if e['url'] not in {x['url'] for x in evidence}:
                        evidence.append(e)
                if len(evidence) >= args.max_results:
                    evidence = evidence[: args.max_results]
                    break
            v_prompt = build_step_validation_prompt(question_text, step, evidence)
            try:
                v_resp = llm.query(v_prompt)
            except Exception as e:
                v_resp = f"[ERROR] {e}"
            decision = v_resp.strip().split("\n")[0]
            step_validation_records.append({
                'step': step,
                'decision': decision,
                'validation_output': v_resp,
                'queries': queries,
                'evidence': evidence
            })
            if decision.startswith("ACCEPT"):
                validated_steps.append(step)
                final_cot.append(step)
                continue
            for _ in range(args.max_rounds):
                reflection_used += 1
                refl_prompt = build_reflection_prompt(question_text, validated_steps, step, v_resp, s_idx, dataset_type=dataset_type)
                try:
                    refl_resp = llm.query(refl_prompt)
                    new_steps, new_concl = extract_cot_and_answer(refl_resp)
                    if new_steps:
                        final_cot = validated_steps + new_steps
                        conclusion = new_concl
                        break
                except Exception as e:
                    step_validation_records.append({'reflection_error': str(e)})
            break
        if not final_cot:
            final_cot = validated_steps if validated_steps else steps
        gold = get_answer_key(item)
        is_correct = False
        if gold is not None and conclusion is not None:
            gold_str = str(gold).strip()
            conclusion_str = str(conclusion).strip()
            letter_to_number = {'A': '0', 'B': '1', 'C': '2', 'D': '3'}
            number_to_letter = {'0': 'A', '1': 'B', '2': 'C', '3': 'D'}
            if gold_str == conclusion_str:
                is_correct = True
            elif gold_str in letter_to_number and letter_to_number[gold_str] == conclusion_str:
                is_correct = True
            elif gold_str in number_to_letter and number_to_letter[gold_str] == conclusion_str:
                is_correct = True
            elif gold_str.upper() == conclusion_str.upper():
                is_correct = True
            else:
                gold_choice = None
                gold_match = re.search(r'([A-D]|[0-3])', gold_str.upper())
                if gold_match:
                    gold_choice = gold_match.group(1)
                conclusion_choice = None
                conclusion_match = re.search(r'([A-D]|[0-3])', conclusion_str.upper())
                if conclusion_match:
                    conclusion_choice = conclusion_match.group(1)
                if gold_choice and conclusion_choice:
                    if gold_choice == conclusion_choice:
                        is_correct = True
                    elif (gold_choice in letter_to_number and 
                          letter_to_number[gold_choice] == conclusion_choice):
                        is_correct = True
                    elif (gold_choice in number_to_letter and 
                          number_to_letter[gold_choice] == conclusion_choice):
                        is_correct = True
        print(f"Gold: '{gold}' vs Prediction: '{conclusion}' -> Correct: {is_correct}")
        if is_correct:
            correct += 1
        elapsed = time.time() - start_time
        total_elapsed += elapsed
        if tokens_used is not None:
            total_tokens += tokens_used
        rec: Dict[str, Any] = {
            'id': sample_id,
            'question': question_text,
            'answerKey': gold,
            'cot_output': cot_output_text,
            'cot_steps': steps,
            'pred': conclusion,
            'final_cot': final_cot,
            'is_correct': is_correct,
            'elapsed_time': elapsed,
            'tokens_used': tokens_used,
            'web_validation': step_validation_records,
            'reflection_rounds': reflection_used
        }
        results.append(rec)

        # Save progressive results
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({"results": results}, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"[ERROR] Failed to save results after sample {idx+1}: {e}")

    acc = correct / len(data) if data else 0.0
    avg_elapsed_time = total_elapsed / len(results) if results else 0.0
    avg_tokens_used = total_tokens / len(results) if results and total_tokens > 0 else None
    print(f'Accuracy: {acc:.2%}')
    print(f'Average elapsed time: {avg_elapsed_time:.2f}s')
    if avg_tokens_used is not None:
        print(f'Average tokens used: {avg_tokens_used:.2f}')

    metrics = {
        "overall_summary": {
            "total_questions_evaluated": len(results),
            "final_accuracy": f"{acc:.2%}",
            "total_successful_pipelines": correct,
            "avg_elapsed_time": avg_elapsed_time,
            "avg_tokens_used": avg_tokens_used
        },
        "detailed_metrics": {
            "web_validation": {
                "avg_reflection_rounds": sum(r.get('reflection_rounds', 0) for r in results) / max(1, len(results))
            }
        }
    }

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "metrics": metrics,
                "results": results
            }, f, indent=4, ensure_ascii=False)
        print(f"[INFO] Full results saved to: {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save results: {e}")


if __name__ == '__main__':
    main()