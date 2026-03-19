#!/usr/bin/env python3
"""Example: Programming language detection from code snippets.

Trains character-level Markov chains on code samples per language.
Syntax patterns (braces, indentation, keywords) are very distinctive
at the character level — typically achieves >95% accuracy.
"""

from typing import Dict, List, Tuple

from markovonnx import MarkovChain, Vocabulary, char_tokenize

LANGUAGES = {
    "python": [
        "def hello_world():\n    print('hello world')\n",
        "for i in range(10):\n    if i % 2 == 0:\n        print(i)\n",
        "class MyClass:\n    def __init__(self, name):\n        self.name = name\n",
        "import os\nimport sys\nfrom pathlib import Path\n",
        "with open('file.txt') as f:\n    data = f.read()\n",
        "result = [x**2 for x in range(100) if x % 3 == 0]\n",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n",
        "try:\n    value = int(input())\nexcept ValueError:\n    print('invalid')\n",
        "names = {'alice': 30, 'bob': 25}\nfor name, age in names.items():\n    print(f'{name}: {age}')\n",
        "async def fetch(url):\n    async with session.get(url) as resp:\n        return await resp.json()\n",
    ],
    "javascript": [
        "function helloWorld() {\n  console.log('hello world');\n}\n",
        "for (let i = 0; i < 10; i++) {\n  if (i % 2 === 0) {\n    console.log(i);\n  }\n}\n",
        "class MyClass {\n  constructor(name) {\n    this.name = name;\n  }\n}\n",
        "const fs = require('fs');\nconst path = require('path');\n",
        "const data = fs.readFileSync('file.txt', 'utf-8');\n",
        "const result = Array.from({length: 100}, (_, i) => i).filter(x => x % 3 === 0).map(x => x**2);\n",
        "function fibonacci(n) {\n  if (n <= 1) return n;\n  return fibonacci(n-1) + fibonacci(n-2);\n}\n",
        "try {\n  const value = parseInt(prompt());\n} catch (e) {\n  console.log('invalid');\n}\n",
        "const names = {alice: 30, bob: 25};\nObject.entries(names).forEach(([name, age]) => console.log(`${name}: ${age}`));\n",
        "async function fetch(url) {\n  const resp = await fetch(url);\n  return resp.json();\n}\n",
    ],
    "rust": [
        "fn main() {\n    println!(\"hello world\");\n}\n",
        "for i in 0..10 {\n    if i % 2 == 0 {\n        println!(\"{}\", i);\n    }\n}\n",
        "struct MyStruct {\n    name: String,\n}\nimpl MyStruct {\n    fn new(name: &str) -> Self {\n        Self { name: name.to_string() }\n    }\n}\n",
        "use std::fs;\nuse std::path::Path;\n",
        "let data = fs::read_to_string(\"file.txt\").unwrap();\n",
        "let result: Vec<i32> = (0..100).filter(|x| x % 3 == 0).map(|x| x*x).collect();\n",
        "fn fibonacci(n: u64) -> u64 {\n    if n <= 1 { return n; }\n    fibonacci(n-1) + fibonacci(n-2)\n}\n",
        "match input.parse::<i32>() {\n    Ok(v) => println!(\"{}\", v),\n    Err(_) => println!(\"invalid\"),\n}\n",
        "let mut names = HashMap::new();\nnames.insert(\"alice\", 30);\nfor (name, age) in &names {\n    println!(\"{}: {}\", name, age);\n}\n",
        "async fn fetch(url: &str) -> Result<String, Box<dyn Error>> {\n    let resp = reqwest::get(url).await?;\n    Ok(resp.text().await?)\n}\n",
    ],
    "go": [
        "func main() {\n\tfmt.Println(\"hello world\")\n}\n",
        "for i := 0; i < 10; i++ {\n\tif i%2 == 0 {\n\t\tfmt.Println(i)\n\t}\n}\n",
        "type MyStruct struct {\n\tName string\n}\nfunc NewMyStruct(name string) *MyStruct {\n\treturn &MyStruct{Name: name}\n}\n",
        "import (\n\t\"fmt\"\n\t\"os\"\n\t\"path/filepath\"\n)\n",
        "data, err := os.ReadFile(\"file.txt\")\nif err != nil {\n\tlog.Fatal(err)\n}\n",
        "result := make([]int, 0)\nfor i := 0; i < 100; i++ {\n\tif i%3 == 0 {\n\t\tresult = append(result, i*i)\n\t}\n}\n",
        "func fibonacci(n int) int {\n\tif n <= 1 {\n\t\treturn n\n\t}\n\treturn fibonacci(n-1) + fibonacci(n-2)\n}\n",
        "value, err := strconv.Atoi(input)\nif err != nil {\n\tfmt.Println(\"invalid\")\n}\n",
        "names := map[string]int{\"alice\": 30, \"bob\": 25}\nfor name, age := range names {\n\tfmt.Printf(\"%s: %d\\n\", name, age)\n}\n",
        "func fetch(url string) (string, error) {\n\tresp, err := http.Get(url)\n\tif err != nil {\n\t\treturn \"\", err\n\t}\n\tdefer resp.Body.Close()\n}\n",
    ],
}


def main() -> None:
    # Train per-language char-level models
    models: Dict[str, tuple] = {}
    for lang, samples in LANGUAGES.items():
        corpus = [char_tokenize(s) for s in samples]
        vocab = Vocabulary()
        vocab.build_from_sequences(corpus)
        mc = MarkovChain(order=4, vocab=vocab, smoothing=1e-5)
        mc.fit(corpus)
        models[lang] = (mc, vocab)

    # Test snippets
    tests = [
        ("def greet(name):\n    return f'hello {name}'\n", "python"),
        ("const greet = (name) => `hello ${name}`;\n", "javascript"),
        ("fn greet(name: &str) -> String {\n    format!(\"hello {}\", name)\n}\n", "rust"),
        ("func greet(name string) string {\n\treturn fmt.Sprintf(\"hello %s\", name)\n}\n", "go"),
        ("numbers = [x for x in range(50) if x > 10]\n", "python"),
        ("const numbers = [...Array(50).keys()].filter(x => x > 10);\n", "javascript"),
        ("let numbers: Vec<_> = (0..50).filter(|&x| x > 10).collect();\n", "rust"),
        ("numbers := make([]int, 0)\nfor i := 0; i < 50; i++ {\n\tif i > 10 { numbers = append(numbers, i) }\n}\n", "go"),
    ]

    print("--- Code Language Detector ---\n")
    correct = 0
    for snippet, expected in tests:
        tokens = char_tokenize(snippet)
        scores = []
        for lang, (mc, _) in models.items():
            ppx = mc.perplexity([tokens])
            scores.append((lang, ppx))
        scores.sort(key=lambda x: x[1])
        predicted = scores[0][0]
        ok = "✓" if predicted == expected else "✗"
        if predicted == expected:
            correct += 1
        first_line = snippet.split("\n")[0][:55]
        ppx_str = "  ".join(f"{l}={p:.0f}" for l, p in scores)
        print(f"  {ok} [{predicted:>10s}] {first_line}")
        print(f"         {ppx_str}")

    print(f"\nAccuracy: {correct}/{len(tests)} ({100*correct/len(tests):.0f}%)")


if __name__ == "__main__":
    main()
