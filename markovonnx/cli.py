"""Command-line interface for markovonnx."""

import argparse
import sys

from markovonnx.version import __version__


def cmd_train(args: argparse.Namespace) -> None:
    """Train a Markov chain and save as .markov archive."""
    from markovonnx.archive import save_markov_archive
    from markovonnx.markov import MarkovChain
    from markovonnx.tokenizers import char_tokenize, corpus_iter, get_tokenize_fn, word_tokenize
    from markovonnx.vocabulary import Vocabulary

    tokenize_fn = get_tokenize_fn(args.mode)

    print(f"Building vocabulary from {args.corpus}...")
    vocab = Vocabulary(max_vocab=args.max_vocab)
    vocab.build_streaming(args.corpus, tokenize_fn=tokenize_fn, max_lines=args.max_lines)
    print(f"Vocabulary: {vocab.size} tokens")

    print(f"Training MarkovChain(order={args.order}, backoff={args.backoff})...")
    mc = MarkovChain(
        order=args.order,
        vocab=vocab,
        smoothing=args.smoothing,
        backoff=args.backoff,
    )
    mc.fit_streaming(args.corpus, tokenize_fn=tokenize_fn, max_lines=args.max_lines)

    save_markov_archive(mc, args.output)

    export_c = getattr(args, "export_c", "")
    if export_c:
        from markovonnx.c_export import export_markov_c_header
        no_quantize = getattr(args, "no_quantize", False)
        progmem = getattr(args, "progmem", False)
        export_markov_c_header(mc, export_c, quantize=not no_quantize, progmem=progmem)
        print(f"C header written: {export_c}")


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate text from a .markov archive."""
    from markovonnx.archive import load_markov_archive
    from markovonnx.generate import generate_markov

    loaded = load_markov_archive(args.archive)
    rt = loaded["runtime"]
    config = loaded["config"]
    order = config.get("order", 2)
    mode = args.mode

    text = generate_markov(
        rt, args.seed, args.length,
        temperature=args.temperature, mode=mode, order=order,
    )
    print(text)


def cmd_export(args: argparse.Namespace) -> None:
    """Export a saved model to a C header file.

    Accepts:
      - ``markov-c``: a JSON file saved via ``MarkovChain.save()``
      - ``hmm-c``:    a JSON file saved via ``HiddenMarkovModel.save()``

    To export directly after training without a separate save step, use
    ``markovonnx train --export-c`` instead.
    """
    src = args.source
    out = args.output
    fmt = args.format
    no_quantize = getattr(args, "no_quantize", False)
    progmem = getattr(args, "progmem", False)

    if fmt == "markov-c":
        from markovonnx.c_export import export_markov_c_header
        from markovonnx.markov import MarkovChain
        mc = MarkovChain.load(src)
        export_markov_c_header(mc, out, quantize=not no_quantize, progmem=progmem)
        print(f"C header written: {out}")
    elif fmt == "hmm-c":
        from markovonnx.c_export import export_hmm_c_header
        from markovonnx.hmm import HiddenMarkovModel
        hmm = HiddenMarkovModel.load(src)
        export_hmm_c_header(hmm, out, progmem=progmem)
        print(f"HMM C header written: {out}")
    else:
        print(f"Unknown format: {fmt}", file=sys.stderr)
        sys.exit(1)


def cmd_train_hmm(args: argparse.Namespace) -> None:
    """Train an HMM from a CoNLL-style tagged corpus and save as JSON.

    Corpus format: one ``obs TAB tag`` per line; blank lines separate sequences.
    """
    from markovonnx.hmm import HiddenMarkovModel
    from markovonnx.vocabulary import Vocabulary

    # Parse corpus
    obs_seqs = []
    tag_seqs = []
    cur_obs: list = []
    cur_tags: list = []
    n_lines = 0

    with open(args.corpus, encoding="utf-8") as fh:
        for raw in fh:
            if args.max_lines and n_lines >= args.max_lines:
                break
            line = raw.rstrip("\n")
            if line.strip() == "":
                if cur_obs:
                    obs_seqs.append(cur_obs)
                    tag_seqs.append(cur_tags)
                    cur_obs = []
                    cur_tags = []
            else:
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    continue  # skip malformed lines
                cur_obs.append(parts[0])
                cur_tags.append(parts[1])
                n_lines += 1

    if cur_obs:
        obs_seqs.append(cur_obs)
        tag_seqs.append(cur_tags)

    if not obs_seqs:
        print("No sequences found in corpus", file=sys.stderr)
        sys.exit(1)

    print(f"Corpus: {len(obs_seqs)} sequences, {n_lines} tokens")

    obs_vocab = Vocabulary(max_vocab=0)
    obs_vocab.build_from_sequences(obs_seqs)
    print(f"Observation vocabulary: {obs_vocab.size} tokens")

    n_states = args.n_states
    print(f"Training HiddenMarkovModel(n_states={n_states})...")
    hmm = HiddenMarkovModel(
        n_states=n_states,
        obs_vocab=obs_vocab,
        smoothing=args.smoothing,
    )
    hmm.fit_supervised(obs_seqs, tag_seqs)

    hmm.save(args.output)
    print(f"Model saved: {args.output}")

    export_c = getattr(args, "export_c", "")
    if export_c:
        from markovonnx.c_export import export_hmm_c_header
        progmem = getattr(args, "progmem", False)
        export_hmm_c_header(hmm, export_c, progmem=progmem)
        print(f"C header written: {export_c}")


def cmd_size_report(args: argparse.Namespace) -> None:
    """Print a human-readable C-header size report for a saved model."""
    from markovonnx.size_report import format_hmm_report, format_markov_report

    fmt = args.format
    no_quantize = getattr(args, "no_quantize", False)
    progmem = getattr(args, "progmem", False)

    if fmt == "markov":
        from markovonnx.markov import MarkovChain
        chain = MarkovChain.load(args.model)
        print(format_markov_report(chain, quantize=not no_quantize, progmem=progmem))
    elif fmt == "hmm":
        from markovonnx.hmm import HiddenMarkovModel
        hmm = HiddenMarkovModel.load(args.model)
        print(format_hmm_report(hmm, progmem=progmem))
    else:
        print(f"Unknown format: {fmt}", file=sys.stderr)
        sys.exit(1)


def cmd_info(args: argparse.Namespace) -> None:
    """Show metadata from a .markov archive."""
    import json
    import zipfile
    from pathlib import Path

    with zipfile.ZipFile(args.archive, "r") as zf:
        config = json.loads(zf.read("config.json"))

    size_kb = Path(args.archive).stat().st_size / 1024
    print(f"Archive:    {args.archive} ({size_kb:.1f} KB)")
    for key, val in config.items():
        print(f"  {key}: {val}")


def main() -> None:
    """Entry point for ``markovonnx`` CLI."""
    parser = argparse.ArgumentParser(
        prog="markovonnx",
        description="Markov chains and HMMs with ONNX export/inference",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")

    # -- train ----------------------------------------------------------------
    p_train = sub.add_parser("train", help="Train a Markov chain from a text corpus")
    p_train.add_argument("corpus", help="Path to text corpus (one line per sequence)")
    p_train.add_argument("-o", "--output", default="model.markov", help="Output .markov archive path")
    p_train.add_argument("--mode", choices=["char", "word"], default="char", help="Tokenization mode")
    p_train.add_argument("--order", type=int, default=3, help="N-gram order")
    p_train.add_argument("--smoothing", type=float, default=1e-5, help="Laplace smoothing alpha")
    p_train.add_argument("--max-vocab", type=int, default=0, help="Max vocabulary size (0=unlimited)")
    p_train.add_argument("--max-lines", type=int, default=0, help="Max corpus lines (0=unlimited)")
    p_train.add_argument("--backoff", action="store_true", help="Enable interpolated backoff")
    p_train.add_argument("--export-c", metavar="PATH", default="", help="Also export a C header (.h) for embedded/ESP32 use")
    p_train.add_argument("--no-quantize", action="store_true", help="Use float32 instead of uint8 in C header (larger)")
    p_train.add_argument("--progmem", action="store_true", help="Annotate C arrays with ESP32 .rodata section attribute")

    # -- generate -------------------------------------------------------------
    p_gen = sub.add_parser("generate", help="Generate text from a .markov archive")
    p_gen.add_argument("archive", help="Path to .markov archive")
    p_gen.add_argument("--seed", default="", help="Seed text")
    p_gen.add_argument("--length", type=int, default=100, help="Tokens to generate")
    p_gen.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    p_gen.add_argument("--mode", choices=["char", "word"], default="char", help="Tokenization mode")

    # -- export ---------------------------------------------------------------
    p_export = sub.add_parser("export", help="Export a saved model to C header")
    p_export.add_argument("source", help="Path to .markov archive (markov-c) or HMM JSON (hmm-c)")
    p_export.add_argument("-o", "--output", required=True, help="Output .h file path")
    p_export.add_argument(
        "--format", choices=["markov-c", "hmm-c"], default="markov-c",
        help="Output format: markov-c or hmm-c (default: markov-c)",
    )
    p_export.add_argument("--no-quantize", action="store_true", help="Use float32 instead of uint8 (markov-c only)")
    p_export.add_argument("--progmem", action="store_true", help="Add ESP32 .rodata section attribute")

    # -- info -----------------------------------------------------------------
    p_info = sub.add_parser("info", help="Show metadata from a .markov archive")
    p_info.add_argument("archive", help="Path to .markov archive")

    # -- train-hmm ------------------------------------------------------------
    p_thmm = sub.add_parser(
        "train-hmm", help="Train an HMM from a CoNLL-style tagged corpus"
    )
    p_thmm.add_argument("corpus", help="Path to CoNLL corpus (obs TAB tag, blank line = sequence boundary)")
    p_thmm.add_argument("-o", "--output", default="model.hmm.json", help="Output JSON path")
    p_thmm.add_argument("--n-states", type=int, default=8, help="Number of hidden states")
    p_thmm.add_argument("--smoothing", type=float, default=1e-5, help="Laplace smoothing alpha")
    p_thmm.add_argument("--export-c", metavar="PATH", default="", help="Also export a C header (.h)")
    p_thmm.add_argument("--progmem", action="store_true", help="Add ESP32 .rodata section attribute to C header")
    p_thmm.add_argument("--max-lines", type=int, default=0, help="Max corpus lines (0=unlimited)")

    # -- size-report ----------------------------------------------------------
    p_size = sub.add_parser("size-report", help="Show C header memory size breakdown for a model")
    p_size.add_argument("model", help="Path to MarkovChain JSON or HMM JSON")
    p_size.add_argument(
        "--format", choices=["markov", "hmm"], default="markov",
        help="Model format (default: markov)",
    )
    p_size.add_argument("--no-quantize", action="store_true", help="Assume float32 probs (markov only)")
    p_size.add_argument("--progmem", action="store_true", help="Show Flash-only fit check")

    args = parser.parse_args()
    if args.command == "train":
        cmd_train(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "info":
        cmd_info(args)
    elif args.command == "size-report":
        cmd_size_report(args)
    elif args.command == "train-hmm":
        cmd_train_hmm(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
