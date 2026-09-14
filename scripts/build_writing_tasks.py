"""Build the writing-loop task file at ``data/raw/writing_tasks.jsonl``.

The default 30/40/30 mixture contains three subsets:
  a) 议论文改进 essay: 内置固定种子题目(可复现, 不依赖网络)。
  b) 摘要大众化改写 abstract: 经 arXiv API 拉取近期摘要(默认 cs.LG/cs.CL/cs.AI/stat.ML),
     记录 arXiv id 保证可溯源; 摘要按长度过滤(500–1500 字符)。
  c) 技术解释 technical: 内置固定概念表, 面向非专家写 300–450 词解释。

Output record:
  {"task_id": "...", "assignment": "...", "subset": "essay"|"abstract"|"technical", "source": ...}

Example (use ``--skip-arxiv`` when running offline):
  python scripts/build_writing_tasks.py --out data/raw/writing_tasks.jsonl
Existing output is not overwritten unless ``--force`` is supplied.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Subset A: fixed argumentative essay topics.
# ---------------------------------------------------------------------------
ESSAY_TOPICS = [
    "Social media platforms should be legally responsible for misinformation spread by their users.",
    "Universities should abolish standardized test scores in admissions.",
    "Remote work should become the default mode for knowledge workers.",
    "Governments should impose a carbon tax on all fossil-fuel consumption.",
    "Artificial intelligence systems should be banned from making final decisions in criminal sentencing.",
    "School curricula should make programming a compulsory subject from primary school.",
    "Professional athletes are overpaid relative to their contribution to society.",
    "Animal testing for cosmetics should be banned worldwide.",
    "Public transportation should be free in all major cities.",
    "Voting should be mandatory for all eligible citizens.",
    "Space exploration funding would be better spent on solving problems on Earth.",
    "Zoos do more harm than good and should be phased out.",
    "The four-day work week should become the legal standard.",
    "Cash should be eliminated in favor of fully digital payments.",
    "Homework does more harm than good in primary education.",
    "Genetically modified crops are essential for global food security.",
    "Autonomous vehicles should be allowed on public roads before achieving perfect safety records.",
    "Museums should return culturally significant artifacts to their countries of origin.",
    "Nuclear energy is the most realistic path to decarbonization.",
    "Advertising targeted at children under twelve should be illegal.",
    "College education should be free for all citizens.",
    "The gig economy exploits workers and requires strict regulation.",
    "Violent video games contribute meaningfully to real-world aggression.",
    "Governments should provide a universal basic income.",
    "Single-use plastics should be banned globally within five years.",
    "Facial recognition technology should be banned in public spaces.",
    "The voting age should be lowered to sixteen.",
    "Fast fashion brands should be taxed for their environmental impact.",
    "Organ donation should operate on an opt-out rather than opt-in basis.",
    "Homeschooling provides a better education than traditional schooling.",
    "The Olympic Games cause more harm than benefit to host cities.",
    "Employers should be forbidden from checking applicants' social media accounts.",
    "All scientific research funded by taxpayers should be freely accessible.",
    "Wild animals should never be kept as private pets.",
    "News organizations should be required to distinguish clearly between reporting and opinion.",
    "Manned missions to Mars are worth the enormous cost and risk.",
    "Junk food advertising should be regulated as strictly as tobacco advertising.",
    "The private ownership of firearms should require licensing comparable to driving.",
    "Cultural appropriation in fashion and art causes genuine harm.",
    "Automation-driven job losses justify taxing robots.",
    "Endangered languages deserve government funding for preservation.",
    "Streaming services have improved the overall quality of television and film.",
    "Bottled water should be banned where safe tap water exists.",
    "Extreme sports with high fatality rates should require special insurance and licensing.",
    "Cities should prioritize bicycles over cars in urban planning.",
    "Grade inflation undermines the value of university degrees.",
    "Influencer marketing should be regulated as traditional advertising.",
    "The doping ban in professional sports should be reconsidered for medically supervised use.",
    "Public libraries remain essential infrastructure in the digital age.",
    "Companies should be required to publish gender and ethnicity pay-gap data.",
]

ESSAY_ASSIGNMENT_TMPL = (
    "Write a well-argued essay of 400-600 words taking a clear position on the "
    "following statement. Support your position with concrete reasons and evidence, "
    "address at least one counterargument, and end with a decisive conclusion.\n\n"
    "Statement: {topic}"
)

ABSTRACT_ASSIGNMENT_TMPL = (
    "Rewrite the following research abstract as a clear, engaging explanation for a "
    "general audience with no technical background (200-350 words). Preserve the key "
    "findings and their significance, avoid jargon, and use accessible analogies "
    "where helpful.\n\nAbstract (arXiv:{arxiv_id}):\n{abstract}"
)

# ---------------------------------------------------------------------------
# Subset C: fixed technical-explanation topics.
# ---------------------------------------------------------------------------
TECH_CONCEPTS = [
    "how public-key cryptography lets strangers exchange secrets securely",
    "why neural networks need an activation function",
    "how a database index speeds up queries",
    "what causes deadlock in concurrent programs and how to avoid it",
    "how the TCP handshake establishes a reliable connection",
    "why floating-point arithmetic produces rounding errors",
    "how garbage collection reclaims unused memory",
    "what a hash function is and why collisions matter",
    "how gradient descent finds the minimum of a loss function",
    "why overfitting happens in machine learning and how to detect it",
    "how a compiler turns source code into machine instructions",
    "what a race condition is and why it is hard to reproduce",
    "how DNS translates a domain name into an IP address",
    "why quantum computers can factor large numbers faster",
    "how the attention mechanism works in a transformer model",
    "what the CAP theorem says about distributed databases",
    "how a Bloom filter tests set membership with little memory",
    "why HTTPS is more secure than HTTP",
    "how a load balancer distributes traffic across servers",
    "what backpropagation computes during neural-network training",
    "how a Merkle tree lets you verify data without downloading all of it",
    "why caching improves performance and when it goes wrong",
    "how reinforcement learning agents learn from rewards",
    "what a container is and how it differs from a virtual machine",
    "how digital signatures prove a message was not tampered with",
    "why sorting algorithms have different time complexities",
    "how a recommendation system predicts what you might like",
    "what causes memory leaks and how profilers find them",
    "how error-correcting codes recover data from noisy channels",
    "why distributed systems need consensus algorithms like Raft",
    "how a GPU accelerates parallel computation",
    "what differential privacy guarantees when sharing statistics",
    "how tokenization breaks text into units a language model can process",
    "why B-trees are used for on-disk data structures",
    "how a virtual memory system maps addresses to physical RAM",
]

TECH_ASSIGNMENT_TMPL = (
    "Write a clear technical explanation (300-450 words) of the following topic for a "
    "reader who is technically curious but not an expert. Be accurate, define any term "
    "you introduce, use a concrete example or analogy, and stay focused on building "
    "correct intuition.\n\nTopic: Explain {concept}."
)

ARXIV_API = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"


def fetch_arxiv_abstracts(categories: list[str], n: int, min_len: int = 500, max_len: int = 1500) -> list[dict]:
    """Fetch recent abstracts, filter by length, and retain their arXiv IDs."""
    query = " OR ".join(f"cat:{c}" for c in categories)
    out, start = [], 0
    while len(out) < n and start < 400:
        resp = requests.get(
            ARXIV_API,
            params={
                "search_query": query,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
                "start": start,
                "max_results": 100,
            },
            timeout=60,
        )
        resp.raise_for_status()
        entries = ET.fromstring(resp.text).findall(f"{ATOM}entry")
        if not entries:
            break
        for e in entries:
            abstract = " ".join((e.findtext(f"{ATOM}summary") or "").split())
            arxiv_id = (e.findtext(f"{ATOM}id") or "").rsplit("/", 1)[-1]
            if min_len <= len(abstract) <= max_len and arxiv_id:
                out.append({"arxiv_id": arxiv_id, "abstract": abstract})
                if len(out) >= n:
                    break
        start += 100
        time.sleep(3)  # arXiv API 礼仪: 请求间隔 >= 3s
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/writing_tasks.jsonl")
    ap.add_argument("--n-essay", type=int, default=30, help="Number of argumentative essay tasks")
    ap.add_argument("--n-abstract", type=int, default=40, help="摘要改写数")
    ap.add_argument("--n-tech", type=int, default=30, help="技术解释数")
    ap.add_argument("--categories", nargs="+", default=["cs.LG", "cs.CL", "cs.AI", "stat.ML"])
    ap.add_argument("--skip-arxiv", action="store_true", help="纯离线: 跳过摘要子集")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的输出文件")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and not args.force:
        sys.exit(f"{out} already exists — refusing to overwrite (use --force). "
                 "已采轨迹依赖任务文件不变, 覆盖前请确认。")

    tasks: list[dict] = []
    for i, topic in enumerate(ESSAY_TOPICS[: args.n_essay]):
        tasks.append(
            {
                "task_id": f"essay_{i:03d}",
                "assignment": ESSAY_ASSIGNMENT_TMPL.format(topic=topic),
                "subset": "essay",
                "source": "builtin_seed_list_v1",
            }
        )
    print(f"essay subset: {len(tasks)} tasks")

    if not args.skip_arxiv:
        abstracts = fetch_arxiv_abstracts(args.categories, args.n_abstract)
        for a in abstracts:
            tasks.append(
                {
                    "task_id": f"abs_{a['arxiv_id'].replace('.', '_')}",
                    "assignment": ABSTRACT_ASSIGNMENT_TMPL.format(**a),
                    "subset": "abstract",
                    "source": f"arxiv:{a['arxiv_id']}",
                }
            )
        print(f"abstract subset: {len(abstracts)} tasks "
              f"(categories={args.categories})")
        if len(abstracts) < args.n_abstract:
            print(f"[warn] only fetched {len(abstracts)}/{args.n_abstract} abstracts")

    for i, concept in enumerate(TECH_CONCEPTS[: args.n_tech]):
        tasks.append(
            {
                "task_id": f"tech_{i:03d}",
                "assignment": TECH_ASSIGNMENT_TMPL.format(concept=concept),
                "subset": "technical",
                "source": "builtin_concept_list_v1",
            }
        )
    print(f"technical subset: {min(args.n_tech, len(TECH_CONCEPTS))} tasks")

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"→ {out}  ({len(tasks)} tasks total)")


if __name__ == "__main__":
    main()
